"""
Heads for GCMLoc.

    PoseRegressor(in_dim, hidden=512)            pooled feature (B, in_dim) -> (t (B,3), q (B,4))
    HeatmapHead(in_dim, heatmap_dim)             light Stage-2 (offline) attention head
    HeatmapDecoder(dims, fused_dim, heatmap_dim, target_size)
                                                 U-Net style decoder: fused feature + depth
                                                 skip connections -> full-resolution heatmap
All heatmaps are softmax-normalised over the spatial dimensions per channel, so they act
as spatial attention weights for the flow embedding (Sum_{i,j} flow * heatmap).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseRegressor(nn.Module):
    """FC regressor: fc1 -> (fc_transl | fc_rot); returns translation and a unit quaternion.

    w_x / w_q are learnable loss-weight parameters inherited from CMRNet; they are kept in
    the state_dict for checkpoint compatibility but are not used by the GCMLoc losses.
    """

    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.leaky = nn.LeakyReLU(0.1)
        self.fc_transl = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.1), nn.Linear(hidden // 2, 3))
        self.fc_rot = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.1), nn.Linear(hidden // 2, 4))
        self.w_x = nn.Parameter(torch.tensor([0.0]))
        self.w_q = nn.Parameter(torch.tensor([-2.5]))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.leaky(self.fc1(x))
        return self.fc_transl(x), F.normalize(self.fc_rot(x), dim=1)


class HeatmapHead(nn.Module):
    """Two 3x3 conv + BN + ReLU, softmax over space. Used as the offline Stage-2 heatmap."""

    def __init__(self, in_dim: int, heatmap_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, heatmap_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(heatmap_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(heatmap_dim, heatmap_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(heatmap_dim),
            nn.ReLU(inplace=True),
        )
        self.softmax = nn.Softmax(dim=-1)
        _init_conv_bn(self)

    def forward(self, F_M: torch.Tensor) -> torch.Tensor:
        B, _, H, W = F_M.shape
        x = self.conv(F_M)
        return self.softmax(x.reshape(B, x.shape[1], H * W)).reshape(B, -1, H, W)


class HeatmapDecoder(nn.Module):
    """
    U-Net style heatmap decoder.

        input_proj : fused feature (fused_dim, H/8) -> C3, resized to f3 (H/16)
        up3 + dec3 : H/16 -> H/8, skip f2
        up2 + dec2 : H/8  -> H/4, skip f1
        head       : C1 -> heatmap_dim, then bilinear resize to target_size + spatial softmax

    Args:
        dims:        channels of the depth features (C1, C2, C3, C4) used as skips.
        fused_dim:   channels of the input feature.
        heatmap_dim: output channels.
        target_size: (H, W) of the output heatmap (image size for Stage 1, H/4 for online Stage 2).
    """

    def __init__(self, dims=(96, 192, 384, 768), fused_dim: int = 128,
                 heatmap_dim: int = 64, target_size=(384, 1280)):
        super().__init__()
        self.target_size = tuple(target_size)
        C1, C2, C3, _ = dims
        self.input_proj = nn.Sequential(nn.Conv2d(fused_dim, C3, 1, bias=False), nn.BatchNorm2d(C3), nn.GELU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(C3, C2, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(C2), nn.GELU())
        self.dec3 = nn.Sequential(nn.Conv2d(C2 + C2, C2, 3, padding=1, bias=False), nn.BatchNorm2d(C2), nn.GELU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(C2, C1, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(C1), nn.GELU())
        self.dec2 = nn.Sequential(nn.Conv2d(C1 + C1, C1, 3, padding=1, bias=False), nn.BatchNorm2d(C1), nn.GELU())
        self.head = nn.Sequential(
            nn.Conv2d(C1, heatmap_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(heatmap_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(heatmap_dim, heatmap_dim, 1, bias=True),
        )
        self.softmax = nn.Softmax(dim=-1)
        _init_conv_bn(self)

    def forward(self, F_fused: torch.Tensor, depth_feats: list) -> torch.Tensor:
        f1, f2, f3, _ = depth_feats
        x = self.input_proj(F_fused)
        if x.shape[2:] != f3.shape[2:]:
            x = F.interpolate(x, size=f3.shape[2:], mode="bilinear", align_corners=False)
        x = self.up3(x)
        if x.shape[2:] != f2.shape[2:]:
            x = F.interpolate(x, size=f2.shape[2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, f2], dim=1))
        x = self.up2(x)
        if x.shape[2:] != f1.shape[2:]:
            x = F.interpolate(x, size=f1.shape[2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, f1], dim=1))
        x = self.head(x)
        if x.shape[2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        B, C, H, W = x.shape
        return self.softmax(x.reshape(B, C, H * W)).reshape(B, C, H, W)


def _init_conv_bn(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
