"""
GCMLoc — Ablation Model.

Ablation design:
    model_type='original' : exact CMRNet_single_mapping.py (original single-stage pipeline).
    model_type='ablation' : v2 two-stage pipeline; each flag swaps ONE component so that
                            performance differences are attributable to that single change.

RGB backbone (rgb_backbone):
    'cnn'     → OriginalCMRNetRGB   — original CMRNet conv0a~conv4b + 1×1 projection
    'dinov2b' → DINOv2Backbone(variant='b') — ViT-B/14, 768-dim patch token
    'dinov2s' → DINOv2Backbone(variant='s') — ViT-S/14, 384-dim patch token (lighter)
    'dinov2'  → DINOv2Backbone(variant=dinov2_variant)  — use dinov2_variant param
    unfreeze_dinov2_blocks: 0=fully frozen, N=unfreeze last N transformer blocks

Depth backbone (depth_backbone):
    'cnn'    → OriginalCMRNetDepth  — 3-branch CNN (branch_gt / branch_init / branch_lhmap)
    'vmamba' → VMambaBackbone

Other components:
    use_cross_fusion=True  → CrossModalFusion at Stage 1 (depth × RGB feature)
    flow_type='multi_scale'→ MultiScaleFlowMatcher (vs 'correlation': single-scale H/16)

Experiment map (see train_mapping.py for full commands):
    Exp   RGB              unfreeze  Depth   Fusion  Flow         Note
    ────  ───────────────  ────────  ──────  ──────  ───────────  ─────────────────────────────────────
    A1    CNN (orig)       —         CNN     ✗       corr(orig)   Original CMRNet (model_type=original)
    A2    CNN              —         CNN     ✗       correlation  v2 pipeline + CNN-only baseline
    B1    DINOv2-B         0(frozen) CNN     ✗       correlation  +DINOv2-B frozen
    B2    DINOv2-B         4         CNN     ✗       correlation  +DINOv2-B unfreeze 4 blocks
    B3    DINOv2-S         0(frozen) CNN     ✗       correlation  +DINOv2-S frozen (lightweight)
    B4    DINOv2-S         4         CNN     ✗       correlation  +DINOv2-S unfreeze 4 blocks
    C1    CNN              —         VMamba  ✗       correlation  +VMamba depth
    D1    CNN              —         CNN     ✓       correlation  +CrossModalFusion (CNN only)
    D2    DINOv2-B         4         CNN     ✓       correlation  +DINOv2-B + Fusion
    E1    CNN              —         CNN     ✗       multi_scale  +MultiScaleFlow (CNN only)
    E2    DINOv2-B         4         CNN     ✓       multi_scale  +DINOv2-B + Fusion + MultiScale
    F1    DINOv2-B         4         CNN     ✓       multi_scale  Full v2 (≡ train_mapping.py) ★
    F2    DINOv2-B         4         CNN     ✓       multi_scale  Full v2 + flow supervision loss

CNN backbone details (OriginalCMRNetRGB / OriginalCMRNetDepth):
    branch_gt   (_DconvBranch):   dconv0a~dconv3b  → encoder to H/8 only (shallower)
    branch_init (_LconvBranch):   lconv0a~lconv5b  → encoder to H/32, full depth
    branch_lhmap (_LlconvBranch): llconv — first layer stride=2, no H/1 level0
    All use Conv2d(bias=True) + LeakyReLU(0.1), NO BatchNorm — matches original CMRNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Helpers — original CMRNet conv style
# ============================================================

def _cmr_conv(in_ch, out_ch, stride=1):
    """
    Original CMRNet conv block: Conv2d(bias=True) + LeakyReLU(0.1).
    Deliberately NO BatchNorm — matches CMRNet_single_mapping.py exactly.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True),
    )


def _proj1x1(in_ch, out_ch):
    """Light projection: Conv1×1(bias=True) + LeakyReLU. No BN — matches CMRNet style."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=True),
        nn.LeakyReLU(0.1, inplace=True),
    )


# ============================================================
# OriginalCMRNetRGB
# ============================================================

class OriginalCMRNetRGB(nn.Module):
    """
    RGB feature extractor matching original CMRNet conv0a~conv4b.

    Architecture (conv + LeakyReLU, no BN — same as original):
        3  → 8  → 8  → 8   (H/1)
        8  → 16 → 16 → 16  (H/2)
        16 → 32 → 32 → 32  (H/4)
        32 → 64 → 64 → 64  (H/8)
        64 → 96 → 96 → 96  (H/16)  ← primary output

    A lightweight 1×1 projection (Conv + BN + LeakyReLU) maps the 96-ch
    H/16 features to feat_dim so they are compatible with the v2 flow matcher.
    """

    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.level0 = nn.Sequential(
            _cmr_conv(3,  8), _cmr_conv(8,  8), _cmr_conv(8,  8),   # H/1
        )
        self.level1 = nn.Sequential(
            _cmr_conv(8,  16, stride=2), _cmr_conv(16, 16), _cmr_conv(16, 16),  # H/2
        )
        self.level2 = nn.Sequential(
            _cmr_conv(16, 32, stride=2), _cmr_conv(32, 32), _cmr_conv(32, 32),  # H/4
        )
        self.level3 = nn.Sequential(
            _cmr_conv(32, 64, stride=2), _cmr_conv(64, 64), _cmr_conv(64, 64),  # H/8
        )
        self.level4 = nn.Sequential(
            _cmr_conv(64, 96, stride=2), _cmr_conv(96, 96), _cmr_conv(96, 96),  # H/16
        )
        # 1×1 projection to feat_dim (only added layer vs. original CMRNet)
        self.proj = _proj1x1(96, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, feat_dim, H/16, W/16)."""
        x = self.level0(x)
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        return self.proj(x)


# ============================================================
# OriginalCMRNetDepth  (3-branch, independent weights, distinct architectures)
# ============================================================

class _LconvBranch(nn.Module):
    """
    lconv branch — matches original CMRNet lconv0a~lconv5b.
    Used for: branch_init (initial lidar, stage-1 correlation).

    Architecture:
        in → 8→8→8  (H/1)
        8  → 16→16→16  (H/2)
        16 → 32→32→32  (H/4)   ← f1
        32 → 64→64→64  (H/8)   ← f2  [primary when output_stage=2]
        64 → 96→96→96  (H/16)  ← f3
        96 → 128→128→128 (H/32) ← f4
    """

    _STAGE_CH = {1: 32, 2: 64, 3: 96, 4: 128}

    def __init__(self, in_ch: int, vmamba_dims: tuple,
                 feat_dim: int, output_stage: int = 2):
        super().__init__()
        self.output_stage = output_stage
        C1, C2, C3, C4 = vmamba_dims

        self.level0 = nn.Sequential(
            _cmr_conv(in_ch, 8), _cmr_conv(8, 8), _cmr_conv(8, 8),         # H/1
        )
        self.level1 = nn.Sequential(
            _cmr_conv(8,  16, stride=2), _cmr_conv(16, 16), _cmr_conv(16, 16),  # H/2
        )
        self.level2 = nn.Sequential(
            _cmr_conv(16, 32, stride=2), _cmr_conv(32, 32), _cmr_conv(32, 32),  # H/4
        )
        self.level3 = nn.Sequential(
            _cmr_conv(32, 64, stride=2), _cmr_conv(64, 64), _cmr_conv(64, 64),  # H/8
        )
        self.level4 = nn.Sequential(
            _cmr_conv(64, 96, stride=2), _cmr_conv(96, 96), _cmr_conv(96, 96),  # H/16
        )
        self.level5 = nn.Sequential(
            _cmr_conv(96, 128, stride=2), _cmr_conv(128, 128), _cmr_conv(128, 128),  # H/32
        )

        self.proj_f1 = _proj1x1(32,  C1)
        self.proj_f2 = _proj1x1(64,  C2)
        self.proj_f3 = _proj1x1(96,  C3)
        self.proj_f4 = _proj1x1(128, C4)
        self.proj_primary = _proj1x1(self._STAGE_CH[output_stage], feat_dim)

    def forward(self, x: torch.Tensor):
        x      = self.level0(x)       # H/1,  8
        x      = self.level1(x)       # H/2,  16
        raw_f1 = self.level2(x)       # H/4,  32
        raw_f2 = self.level3(raw_f1)  # H/8,  64
        raw_f3 = self.level4(raw_f2)  # H/16, 96
        raw_f4 = self.level5(raw_f3)  # H/32, 128

        f1 = self.proj_f1(raw_f1)
        f2 = self.proj_f2(raw_f2)
        f3 = self.proj_f3(raw_f3)
        f4 = self.proj_f4(raw_f4)

        raw_primary = {1: raw_f1, 2: raw_f2, 3: raw_f3, 4: raw_f4}[self.output_stage]
        primary = self.proj_primary(raw_primary)
        return primary, [f1, f2, f3, f4]


class _DconvBranch(nn.Module):
    """
    dconv branch — matches original CMRNet dconv0a~dconv3b (shallower encoder).
    Used for: branch_gt (GT depth, stage-1 heatmap).

    Architecture (encoder to H/8 only, same as original dconv):
        in → 8→8→8  (H/1)
        8  → 16→16→16  (H/2)
        16 → 32→32→32  (H/4)  ← f1
        32 → 64→64→64  (H/8)  ← f2  [primary]

    f3 (H/16) and f4 (H/32) are derived via single lightweight stride-2 convs
    (not full 3-layer blocks) purely for Stage1HeatmapDecoder size compatibility.
    f4 is assigned but unused by Stage1HeatmapDecoder.
    """

    def __init__(self, in_ch: int, vmamba_dims: tuple,
                 feat_dim: int, output_stage: int = 2):
        super().__init__()
        C1, C2, C3, C4 = vmamba_dims

        # Original dconv0a~dconv3b: encoder to H/8
        self.level0 = nn.Sequential(
            _cmr_conv(in_ch, 8), _cmr_conv(8, 8), _cmr_conv(8, 8),         # H/1
        )
        self.level1 = nn.Sequential(
            _cmr_conv(8,  16, stride=2), _cmr_conv(16, 16), _cmr_conv(16, 16),  # H/2
        )
        self.level2 = nn.Sequential(
            _cmr_conv(16, 32, stride=2), _cmr_conv(32, 32), _cmr_conv(32, 32),  # H/4
        )
        self.level3 = nn.Sequential(
            _cmr_conv(32, 64, stride=2), _cmr_conv(64, 64), _cmr_conv(64, 64),  # H/8
        )

        # Lightweight extension for skip-connection size compatibility only
        self.level4_lite = _cmr_conv(64,  96, stride=2)  # H/16, single conv
        self.level5_lite = _cmr_conv(96, 128, stride=2)  # H/32, single conv

        self.proj_f1 = _proj1x1(32,  C1)
        self.proj_f2 = _proj1x1(64,  C2)
        self.proj_f3 = _proj1x1(96,  C3)
        self.proj_f4 = _proj1x1(128, C4)
        self.proj_primary = _proj1x1(64, feat_dim)   # primary always at H/8 (64ch)

    def forward(self, x: torch.Tensor):
        x      = self.level0(x)            # H/1,  8
        x      = self.level1(x)            # H/2,  16
        raw_f1 = self.level2(x)            # H/4,  32
        raw_f2 = self.level3(raw_f1)       # H/8,  64
        raw_f3 = self.level4_lite(raw_f2)  # H/16, 96  (lightweight)
        raw_f4 = self.level5_lite(raw_f3)  # H/32, 128 (lightweight)

        f1 = self.proj_f1(raw_f1)
        f2 = self.proj_f2(raw_f2)
        f3 = self.proj_f3(raw_f3)
        f4 = self.proj_f4(raw_f4)

        primary = self.proj_primary(raw_f2)  # primary at H/8
        return primary, [f1, f2, f3, f4]


class _LlconvBranch(nn.Module):
    """
    llconv branch — matches original CMRNet llconv architecture.
    Used for: branch_lhmap (filtered lidar, stage-2).

    Key difference: first layer has stride=2 (no H/1 level0),
    starts directly at H/2 — same as original llconv1a.

    Architecture:
        in → 16→16→16  (H/2, stride=2 first layer — NO H/1 level)
        16 → 32→32→32  (H/4)   ← f1
        32 → 64→64→64  (H/8)   ← f2  [primary when output_stage=2]
        64 → 96→96→96  (H/16)  ← f3
        96 → 128→128→128 (H/32) ← f4
    """

    _STAGE_CH = {1: 32, 2: 64, 3: 96, 4: 128}

    def __init__(self, in_ch: int, vmamba_dims: tuple,
                 feat_dim: int, output_stage: int = 2):
        super().__init__()
        self.output_stage = output_stage
        C1, C2, C3, C4 = vmamba_dims

        # llconv: first layer stride=2 (in→16 at H/2), no H/1 level0
        self.level1 = nn.Sequential(
            _cmr_conv(in_ch, 16, stride=2), _cmr_conv(16, 16), _cmr_conv(16, 16),  # H/2
        )
        self.level2 = nn.Sequential(
            _cmr_conv(16, 32, stride=2), _cmr_conv(32, 32), _cmr_conv(32, 32),  # H/4
        )
        self.level3 = nn.Sequential(
            _cmr_conv(32, 64, stride=2), _cmr_conv(64, 64), _cmr_conv(64, 64),  # H/8
        )
        self.level4 = nn.Sequential(
            _cmr_conv(64, 96, stride=2), _cmr_conv(96, 96), _cmr_conv(96, 96),  # H/16
        )
        self.level5 = nn.Sequential(
            _cmr_conv(96, 128, stride=2), _cmr_conv(128, 128), _cmr_conv(128, 128),  # H/32
        )

        self.proj_f1 = _proj1x1(32,  C1)
        self.proj_f2 = _proj1x1(64,  C2)
        self.proj_f3 = _proj1x1(96,  C3)
        self.proj_f4 = _proj1x1(128, C4)
        self.proj_primary = _proj1x1(self._STAGE_CH[output_stage], feat_dim)

    def forward(self, x: torch.Tensor):
        x      = self.level1(x)       # H/2,  16 (stride=2, no H/1)
        raw_f1 = self.level2(x)       # H/4,  32
        raw_f2 = self.level3(raw_f1)  # H/8,  64
        raw_f3 = self.level4(raw_f2)  # H/16, 96
        raw_f4 = self.level5(raw_f3)  # H/32, 128

        f1 = self.proj_f1(raw_f1)
        f2 = self.proj_f2(raw_f2)
        f3 = self.proj_f3(raw_f3)
        f4 = self.proj_f4(raw_f4)

        raw_primary = {1: raw_f1, 2: raw_f2, 3: raw_f3, 4: raw_f4}[self.output_stage]
        primary = self.proj_primary(raw_primary)
        return primary, [f1, f2, f3, f4]


class OriginalCMRNetDepth(nn.Module):
    """
    3-branch depth backbone with distinct architectures matching original CMRNet:
      - branch_gt   (_DconvBranch):   dconv — shallower encoder (to H/8), matches dconv0~dconv3
      - branch_init (_LconvBranch):   lconv — full encoder (to H/32), matches lconv0~lconv5
      - branch_lhmap (_LlconvBranch): llconv — stride=2 first layer, no H/1, matches llconv

    Interface matches VMambaBackbone:
        extract_depth_gt(depth)   → (F_D_gt,  [f1,f2,f3,f4])
        extract_depth_init(lidar) → (F_D_init, [f1,f2,f3,f4])
        extract_gcmloc(gcmloc)      → (F_M,      [f1,f2,f3,f4])
    """

    def __init__(self, in_channels: int = 1, feat_dim: int = 128,
                 vmamba_dims: tuple = (96, 192, 384, 768),
                 output_stage: int = 2,
                 legacy_depth_branches: bool = False):
        super().__init__()
        kw = dict(vmamba_dims=vmamba_dims, feat_dim=feat_dim, output_stage=output_stage)
        self.branch_gt    = _DconvBranch(in_ch=in_channels, **kw)
        self.branch_init  = _LconvBranch(in_ch=in_channels, **kw)
        # legacy_depth_branches=True: old checkpoints where all three branches used
        # the same _LconvBranch architecture (all had level0).
        if legacy_depth_branches:
            self.branch_lhmap = _LconvBranch(in_ch=in_channels, **kw)
        else:
            self.branch_lhmap = _LlconvBranch(in_ch=in_channels, **kw)

    def extract_depth_gt(self, x):
        return self.branch_gt(x)

    def extract_depth_init(self, x):
        return self.branch_init(x)

    def extract_gcmloc(self, x):
        return self.branch_lhmap(x)


# ============================================================
# OriginalCMRNetCorrelation  (flow_type='correlation')
# ============================================================

class OriginalCMRNetCorrelation(nn.Module):
    """
    Single-scale local correlation at H/16 — matches original CMRNet's
    corr(c14, c24) scale (both are H/16 features).

    Input:
        F_rgb:      (B, feat_dim, H/16, W/16)  — from OriginalCMRNetRGB
        depth_feats: list [f1(H/4), f2(H/8), f3(H/16), f4(H/32)]

    Uses depth_feats[2] (H/16, C3) to match the RGB scale.
    """

    def __init__(self, rgb_dim: int, depth_dims: tuple,
                 embed_dim: int, md: int = 4):
        super().__init__()
        self.md = md
        C3 = depth_dims[2]   # H/16 channel dim

        self.rgb_proj   = _proj1x1(rgb_dim, 64)
        self.depth_proj = _proj1x1(C3,      64)

        corr_ch = (2 * md + 1) ** 2   # 81
        self.head = nn.Sequential(
            nn.Conv2d(corr_ch, 128, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, embed_dim, 1),
        )

    def forward(self, F_rgb: torch.Tensor, depth_feats: list):
        """Returns (flow_embed, []) where flow_embed is (B, embed_dim, H/16, W/16)."""
        F_d = depth_feats[2]   # H/16 — same scale as RGB output

        fr = self.rgb_proj(F_rgb)
        fd = self.depth_proj(F_d)

        # Align spatial size (handles any rounding differences)
        if fr.shape[2:] != fd.shape[2:]:
            fr = F.interpolate(fr, fd.shape[2:], mode='bilinear', align_corners=False)

        fr = F.normalize(fr, dim=1)
        fd = F.normalize(fd, dim=1)

        corr = self._local_corr(fr, fd)
        return self.head(corr), []

    def _local_corr(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f1.shape
        md = self.md
        f2_pad = F.pad(f2, [md] * 4)
        corrs = [
            (f1 * f2_pad[:, :, i:i + H, j:j + W]).sum(1, keepdim=True)
            for i in range(2 * md + 1)
            for j in range(2 * md + 1)
        ]
        return torch.cat(corrs, dim=1)


# ============================================================
# GCMLocMapping — main model
# ============================================================

class GCMLocMapping(nn.Module):
    """
    v2 pipeline with ablation switches.

    Default (rgb='cnn', depth='cnn', fusion=False, flow='correlation'):
        Uses original CMRNet CNN backbones inside the v2 2-stage pipeline.
        This is NOT identical to Exp1 (original CMRNet) because the pipeline
        structure differs, but the CNN quality is the same.

    Each flag swaps exactly one component to the v2 version:
        rgb_backbone='dinov2'   → replace OriginalCMRNetRGB with DINOv2
        depth_backbone='vmamba' → replace OriginalCMRNetDepth with VMamba
        use_cross_fusion=True   → add CrossModalFusion between depth and RGB
        flow_type='multi_scale' → replace correlation with MultiScaleFlowMatcher
    """

    def __init__(
        self,
        image_size=(384, 1280),
        feat_dim=128,
        heatmap_dim_s1=64,
        heatmap_dim_s2=128,
        topk_points=5000,
        vmamba_dims=(96, 192, 384, 768),
        vmamba_output_stage=2,
        use_cnn_fallback=False,
        dropout=0.0,
        use_reflectance=False,
        # ===== Ablation flags =====
        rgb_backbone='cnn',           # 'dinov2' | 'dinov2s' | 'dinov2b' | 'cnn'
        dinov2_variant='b',           # 'b' (ViT-B) or 's' (ViT-S), used when rgb_backbone starts with 'dinov2'
        unfreeze_dinov2_blocks=0,
        depth_backbone='cnn',         # 'vmamba' | 'cnn'
        use_cross_fusion=False,
        flow_type='correlation',      # 'multi_scale' | 'correlation'
    ):
        super().__init__()
        self.image_size = image_size
        self.feat_dim = feat_dim
        self.topk_points = topk_points
        self.flow_type = flow_type
        self.use_cross_fusion = use_cross_fusion

        input_lidar = 2 if use_reflectance else 1

        # ===== RGB Backbone =====
        # rgb_backbone='dinov2s' or 'dinov2b' directly specify the variant;
        # rgb_backbone='dinov2' uses the dinov2_variant parameter.
        if rgb_backbone.startswith('dinov2'):
            from models.backbone.dinov2_wrapper import DINOv2Backbone
            if rgb_backbone == 'dinov2s':
                _variant = 's'
            elif rgb_backbone == 'dinov2b':
                _variant = 'b'
            else:
                _variant = dinov2_variant
            self.rgb_backbone = DINOv2Backbone(out_dim=feat_dim, variant=_variant)
            if unfreeze_dinov2_blocks > 0:
                self._unfreeze_dinov2(unfreeze_dinov2_blocks)
        else:
            self.rgb_backbone = OriginalCMRNetRGB(feat_dim=feat_dim)

        # ===== Depth Backbone =====
        if depth_backbone == 'vmamba':
            from models.backbone.vmamba_backbone import VMambaBackbone
            self.depth_backbone = VMambaBackbone(
                out_dim=feat_dim, vmamba_dims=vmamba_dims,
                output_stage=vmamba_output_stage,
                in_channels=input_lidar,
                use_cnn_fallback=use_cnn_fallback,
            )
        else:
            self.depth_backbone = OriginalCMRNetDepth(
                in_channels=input_lidar, feat_dim=feat_dim,
                vmamba_dims=vmamba_dims, output_stage=vmamba_output_stage,
            )

        # ===== Cross-Modal Fusion (optional) =====
        if use_cross_fusion:
            from models.fusion.cross_modal_fusion import CrossModalFusion
            self.fusion = CrossModalFusion(feat_dim=feat_dim)

        # ===== Stage 1 Heatmap Decoder =====
        from models.heads.heads import HeatmapHead, Stage1HeatmapDecoder, PoseRegressor

        self.heatmap_decoder_s1 = Stage1HeatmapDecoder(
            vmamba_dims=vmamba_dims, fused_dim=feat_dim,
            heatmap_dim=heatmap_dim_s1, target_size=image_size,
        )

        # ===== Flow Matching =====
        if flow_type == 'multi_scale':
            from models.fusion.multi_scale_flow import MultiScaleFlowMatcher
            self.flow_s1 = MultiScaleFlowMatcher(
                rgb_dim=feat_dim, depth_dims=vmamba_dims[:3],
                corr_dim=feat_dim, embed_dim=heatmap_dim_s1,
            )
            self.flow_s2 = MultiScaleFlowMatcher(
                rgb_dim=feat_dim, depth_dims=vmamba_dims[:3],
                corr_dim=feat_dim, embed_dim=heatmap_dim_s2,
            )
        else:
            # OriginalCMRNetCorrelation: H/16 scale, same as original CMRNet corr(c14,c24)
            self.flow_s1 = OriginalCMRNetCorrelation(
                rgb_dim=feat_dim, depth_dims=vmamba_dims[:3],
                embed_dim=heatmap_dim_s1, md=4,
            )
            self.flow_s2 = OriginalCMRNetCorrelation(
                rgb_dim=feat_dim, depth_dims=vmamba_dims[:3],
                embed_dim=heatmap_dim_s2, md=4,
            )

        # ===== Pose Regressors =====
        self.pose_reg_s1 = PoseRegressor(
            in_dim=heatmap_dim_s1, hidden=512, dropout=dropout,
        )
        self.heatmap_head_s2 = HeatmapHead(
            in_dim=feat_dim, heatmap_dim=heatmap_dim_s2,
        )
        self.pose_reg_s2 = PoseRegressor(
            in_dim=heatmap_dim_s2, hidden=512, dropout=dropout,
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _unfreeze_dinov2(self, n_blocks: int):
        backbone = self.rgb_backbone.dino
        total = len(backbone.blocks)
        n = min(n_blocks, total)
        for block in backbone.blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        if hasattr(backbone, 'norm'):
            for p in backbone.norm.parameters():
                p.requires_grad = True
        print(f"[Ablation] Unfroze last {n}/{total} DINOv2 blocks")

    def _init_weights(self):
        """Init only trainable weights that haven't been set already."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in')
                    if m.bias is not None:
                        m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in')
                    if m.bias is not None:
                        m.bias.data.zero_()

    # ------------------------------------------------------------------
    def forward(self, rgb, lidar, depth, flagt, u, v):
        """
        Same signature as original CMRNet and GCMLocMapping.
        Returns: (transl, rot, transl0, rot0)
        """
        B = rgb.shape[0]

        # ===== Feature Extraction =====
        F_I = self.rgb_backbone(rgb)                                   # (B, C, H/16, W/16)
        F_D_gt,   feats_gt   = self.depth_backbone.extract_depth_gt(depth)
        F_D_init, feats_init = self.depth_backbone.extract_depth_init(lidar)

        # ===== Stage 1: Heatmap + Coarse Pose =====
        if self.use_cross_fusion:
            F_fused = self.fusion(F_D_gt, F_I)
        else:
            F_fused = F_D_gt

        h_lidar = self.heatmap_decoder_s1(F_fused, feats_gt)          # (B, hm_s1, H, W)

        flow_embed_s1, flows_s1 = self.flow_s1(F_I, feats_init[:3])
        self._stage1_flows = flows_s1 if self.flow_type == 'multi_scale' else None

        if flow_embed_s1.shape[2:] != h_lidar.shape[2:]:
            flow_embed_s1 = F.interpolate(
                flow_embed_s1, h_lidar.shape[2:],
                mode='bilinear', align_corners=False,
            )

        x_s1 = torch.sum(flow_embed_s1 * h_lidar, dim=(2, 3))         # (B, hm_s1)
        transl0, rot0 = self.pose_reg_s1(x_s1)

        # ===== Point Selection =====
        h_collapsed = h_lidar.sum(dim=1, keepdim=True) * flagt
        H, W = h_collapsed.shape[2], h_collapsed.shape[3]
        h_flat = h_collapsed.reshape(B, H * W)
        _, indices = h_flat.topk(self.topk_points, dim=-1, largest=True, sorted=False)

        u_flat = u.long().reshape(B, H * W)
        v_flat = v.long().reshape(B, H * W)
        f = torch.zeros((B, 1, H, W), device=rgb.device)
        for i in range(B):
            f[i, 0, u_flat[i, indices[i]], v_flat[i, indices[i]]] = 1
        lidar_filtered = f * lidar

        # ===== Stage 2: Refined Pose =====
        F_M, feats_m = self.depth_backbone.extract_gcmloc(lidar_filtered)
        ch = self.heatmap_head_s2(F_M)                                 # (B, hm_s2, H', W')

        flow_embed_s2, _ = self.flow_s2(F_I, feats_m[:3])
        if flow_embed_s2.shape[2:] != ch.shape[2:]:
            flow_embed_s2 = F.interpolate(
                flow_embed_s2, ch.shape[2:],
                mode='bilinear', align_corners=False,
            )

        x_s2 = torch.sum(flow_embed_s2 * ch, dim=(2, 3))              # (B, hm_s2)
        transl, rot = self.pose_reg_s2(x_s2)

        return transl, rot, transl0, rot0
