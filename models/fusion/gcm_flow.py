"""
GCM-Flow: global coarse-to-fine cross-modal flow matcher.

    GCMFlow(rgb_dim=128, depth_dims=(96, 192, 384), corr_dim=128, embed_dim=64)
        forward(F_I, [f1, f2, f3]) -> flow_embedding (B, embed_dim, H/4, W/4), [flow_3, flow_2, flow_1]

Matches the RGB feature F_I against the multi-scale depth features from coarse to fine:
    Scale 3 (H/16): global all-pairs correlation (softmax with learnable temperature)
                    -> flow_3 + feature
    Scale 2 (H/8) : warp depth feature with the up-sampled flow, local correlation
                    (radius 4, 9x9) -> residual flow_2 + feature
    Scale 1 (H/4) : warp + local correlation (radius 3, 7x7) -> residual flow_1 + feature
The three per-scale features are fused into one flow embedding that is pooled with the
heatmap for pose regression. Pure PyTorch (no CUDA correlation extension needed).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def bilinear_warp(feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Backward-warp ``feat`` using pixel-level ``flow`` displacement.

    Args:
        feat: (B, C, H, W)
        flow: (B, 2, H, W) — (dx, dy) pixel displacements
    Returns:
        warped: (B, C, H, W)
    """
    B, _, H, W = flow.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=flow.device, dtype=flow.dtype),
        torch.arange(W, device=flow.device, dtype=flow.dtype),
        indexing='ij',
    )
    # (B, 2, H, W): base grid
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    coords = grid + flow  # new sampling locations

    # Normalize to [-1, 1]
    coords_norm = torch.empty_like(coords)
    coords_norm[:, 0] = 2.0 * coords[:, 0] / max(W - 1, 1) - 1.0
    coords_norm[:, 1] = 2.0 * coords[:, 1] / max(H - 1, 1) - 1.0

    warped = F.grid_sample(
        feat,
        coords_norm.permute(0, 2, 3, 1),  # (B, H, W, 2)
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )
    return warped


def local_correlation(
    feat1: torch.Tensor,
    feat2: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """
    Compute local correlation volume (pure PyTorch, no CUDA extension).

    For each pixel in *feat1*, compute dot product with a
    (2r+1)x(2r+1) neighbourhood in *feat2*.

    Args:
        feat1: (B, C, H, W) — L2-normalized reference
        feat2: (B, C, H, W) — L2-normalized target
        radius: search radius
    Returns:
        corr: (B, (2r+1)^2, H, W)
    """
    B, C, H, W = feat1.shape
    feat2_pad = F.pad(feat2, [radius] * 4, mode='constant', value=0.0)

    corr_list = []
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            target = feat2_pad[:, :, dy:dy + H, dx:dx + W]
            corr_list.append((feat1 * target).sum(dim=1, keepdim=True))

    return torch.cat(corr_list, dim=1)  # (B, (2r+1)^2, H, W)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvGN(nn.Module):
    """Conv2d + GroupNorm + GELU."""
    def __init__(self, in_ch: int, out_ch: int, ks: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, padding=ks // 2, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# GCMFlow
# ---------------------------------------------------------------------------

class GCMFlow(nn.Module):
    """
    Args:
        rgb_dim:     channels of the projected RGB feature F_I (default 128).
        depth_dims:  channels of the depth features [f1 (H/4), f2 (H/8), f3 (H/16)].
        corr_dim:    unified channel dim used for correlation (default 128).
        embed_dim:   output flow-embedding channels (= heatmap channels it is pooled with).
        local_radius_L2 / local_radius_L1: local search radius at scale 2 / scale 1.
    """

    def __init__(
        self,
        rgb_dim: int = 128,
        depth_dims: tuple = (96, 192, 384),
        corr_dim: int = 128,
        embed_dim: int = 64,
        local_radius_L2: int = 4,
        local_radius_L1: int = 3,
    ):
        super().__init__()
        self.corr_dim = corr_dim
        self.embed_dim = embed_dim
        self.local_radius_L2 = local_radius_L2
        self.local_radius_L1 = local_radius_L1

        C1, C2, C3 = depth_dims

        # ---- Projection: unify channels to corr_dim ----
        self.proj_rgb_L3 = _ConvGN(rgb_dim, corr_dim, ks=1)
        self.proj_rgb_L2 = _ConvGN(rgb_dim, corr_dim, ks=1)
        self.proj_rgb_L1 = _ConvGN(rgb_dim, corr_dim, ks=1)

        self.proj_lid_L3 = _ConvGN(C3, corr_dim, ks=1)
        self.proj_lid_L2 = _ConvGN(C2, corr_dim, ks=1)
        self.proj_lid_L1 = _ConvGN(C1, corr_dim, ks=1)

        # ---- Scale 3: Global Correlation ----
        self.log_tau = nn.Parameter(torch.tensor(0.0))  # learnable temperature

        global_in = corr_dim * 3  # cat[F_I, F_agg, F_D]
        self.flow_pred_L3 = nn.Sequential(
            _ConvGN(global_in, corr_dim),
            _ConvGN(corr_dim, 64),
            nn.Conv2d(64, 2, 3, padding=1),
        )
        self.feat_ext_L3 = nn.Sequential(
            _ConvGN(global_in, corr_dim),
            _ConvGN(corr_dim, embed_dim, ks=1),
        )

        # ---- Scale 2: Local Correlation + Warp ----
        local_ch_L2 = (2 * local_radius_L2 + 1) ** 2  # 81
        local_in_L2 = local_ch_L2 + corr_dim + 2       # corr + rgb + flow_up

        self.flow_pred_L2 = nn.Sequential(
            _ConvGN(local_in_L2, corr_dim),
            _ConvGN(corr_dim, 64),
            nn.Conv2d(64, 2, 3, padding=1),
        )
        self.feat_ext_L2 = nn.Sequential(
            _ConvGN(local_in_L2, corr_dim),
            _ConvGN(corr_dim, embed_dim, ks=1),
        )

        # ---- Scale 1: Local Correlation + Warp ----
        local_ch_L1 = (2 * local_radius_L1 + 1) ** 2  # 49
        local_in_L1 = local_ch_L1 + corr_dim + 2

        self.flow_pred_L1 = nn.Sequential(
            _ConvGN(local_in_L1, corr_dim),
            _ConvGN(corr_dim, 64),
            nn.Conv2d(64, 2, 3, padding=1),
        )
        self.feat_ext_L1 = nn.Sequential(
            _ConvGN(local_in_L1, corr_dim),
            _ConvGN(corr_dim, embed_dim, ks=1),
        )

        # ---- Multi-scale fusion ----
        self.fusion = nn.Sequential(
            _ConvGN(embed_dim * 3, embed_dim * 2),
            _ConvGN(embed_dim * 2, embed_dim, ks=1),
        )

        self._init_weights()

    # -------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(min=1e-4)

    # -------------------------------------------------
    def forward(self, F_I: torch.Tensor, depth_feats: list):
        """
        Args:
            F_I: projected RGB feature  (B, rgb_dim, h, w)
            depth_feats: depth CNN multi-scale features [f1, f2, f3]
                f1 (B, C1, H/4,  W/4)  — finest
                f2 (B, C2, H/8,  W/8)
                f3 (B, C3, H/16, W/16) — coarsest
        Returns:
            flow_embedding (B, embed_dim, H/4, W/4)
            flows          [flow_3, flow_2, flow_1]
        """
        f1, f2, f3 = depth_feats[0], depth_feats[1], depth_feats[2]

        # ============ Scale 3 — Global Correlation ============
        H3, W3 = f3.shape[2:]
        F_I_L3 = self.proj_rgb_L3(
            F.interpolate(F_I, size=(H3, W3), mode='bilinear', align_corners=False)
        )
        F_D_L3 = self.proj_lid_L3(f3)

        B, C = F_I_L3.shape[:2]
        A_flat = F.normalize(F_I_L3.flatten(2), p=2, dim=1)  # (B, C, N3)
        B_flat = F.normalize(F_D_L3.flatten(2), p=2, dim=1)

        C_corr = torch.bmm(A_flat.permute(0, 2, 1), B_flat) / self.tau  # (B, N3, N3)
        attn = F.softmax(C_corr, dim=-1)

        # Aggregate depth features at RGB positions
        F_agg = torch.bmm(B_flat, attn.transpose(1, 2)).reshape(B, C, H3, W3)

        x3 = torch.cat([F_I_L3, F_agg, F_D_L3], dim=1)
        flow_3 = self.flow_pred_L3(x3)   # (B, 2, H3, W3)
        feat_3 = self.feat_ext_L3(x3)    # (B, embed_dim, H3, W3)

        # ============ Scale 2 — Local Correlation + Warp ============
        H2, W2 = f2.shape[2:]
        F_I_L2 = self.proj_rgb_L2(
            F.interpolate(F_I, size=(H2, W2), mode='bilinear', align_corners=False)
        )
        F_D_L2 = self.proj_lid_L2(f2)

        flow_3_up = F.interpolate(
            flow_3, size=(H2, W2), mode='bilinear', align_corners=False
        ) * 2.0  # scale displacement when upsampling 2×
        F_D_L2_w = bilinear_warp(F_D_L2, flow_3_up)

        corr_2 = local_correlation(
            F.normalize(F_I_L2, p=2, dim=1),
            F.normalize(F_D_L2_w, p=2, dim=1),
            self.local_radius_L2,
        )

        x2 = torch.cat([corr_2, F_I_L2, flow_3_up], dim=1)
        flow_2 = flow_3_up + self.flow_pred_L2(x2)  # residual
        feat_2 = self.feat_ext_L2(x2)

        # ============ Scale 1 — Local Correlation + Warp ============
        H1, W1 = f1.shape[2:]
        F_I_L1 = self.proj_rgb_L1(
            F.interpolate(F_I, size=(H1, W1), mode='bilinear', align_corners=False)
        )
        F_D_L1 = self.proj_lid_L1(f1)

        flow_2_up = F.interpolate(
            flow_2, size=(H1, W1), mode='bilinear', align_corners=False
        ) * 2.0
        F_D_L1_w = bilinear_warp(F_D_L1, flow_2_up)

        corr_1 = local_correlation(
            F.normalize(F_I_L1, p=2, dim=1),
            F.normalize(F_D_L1_w, p=2, dim=1),
            self.local_radius_L1,
        )

        x1 = torch.cat([corr_1, F_I_L1, flow_2_up], dim=1)
        flow_1 = flow_2_up + self.flow_pred_L1(x1)
        feat_1 = self.feat_ext_L1(x1)

        # ============ Multi-scale Fusion ============
        feat_2_up = F.interpolate(feat_2, size=(H1, W1), mode='bilinear', align_corners=False)
        feat_3_up = F.interpolate(feat_3, size=(H1, W1), mode='bilinear', align_corners=False)

        flow_embedding = self.fusion(
            torch.cat([feat_1, feat_2_up, feat_3_up], dim=1)
        )  # (B, embed_dim, H1, W1)

        return flow_embedding, [flow_3, flow_2, flow_1]
