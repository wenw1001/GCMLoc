"""
GCMLoc — Ablation Online Localization Model (Stage 2 only).

Standalone file following the same pattern as CMRNet_single_loc.py.
Module names match GCMLocMapping Stage 2 exactly for checkpoint compatibility:
    rgb_backbone, depth_backbone, heatmap_head_s2, flow_s2, pose_reg_s2

Key improvement over the original GCMLocLocalization:
    heatmap_head_s2 uses Stage1HeatmapDecoder (multi-scale U-Net, H/16→H/8→H/4)
    instead of HeatmapHead (single-scale), matching the richer heatmap decoder in
    CMRNet_single_loc.py (deconvh1/h2 + skip connections from c24/c25).

Checkpoint loading from train_mapping.py mapping checkpoint (strict=False):
    - Stage 1 keys (heatmap_decoder_s1, flow_s1, pose_reg_s1, fusion) → unexpected, ignored
    - heatmap_head_s2.* → shape mismatch (HeatmapHead ≠ Stage1HeatmapDecoder) → random init
    - rgb_backbone, depth_backbone, flow_s2, pose_reg_s2 → loaded correctly
    - depth_backbone keeps all 3 branches; only extract_gcmloc (branch_lhmap) is called online
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCMLocLocalization(nn.Module):
    """
    GCMLoc ablation localization model (Stage 2 only).

    Args:
        image_size:             (H, W) of input images, default (384, 1280).
        feat_dim:               Unified feature dimension.
        heatmap_dim_s2:         Heatmap attention channels for Stage 2.
        vmamba_dims:            VMamba stage channel dims (C1, C2, C3, C4).
        vmamba_output_stage:    Which backbone stage is used as primary (default 2 = H/8).
        use_cnn_fallback:       Use CNN inside VMambaBackbone when vmamba unavailable.
        dropout:                Dropout rate in PoseRegressor.
        use_reflectance:        Whether lidar input has a reflectance channel.
        rgb_backbone:           'cnn' | 'dinov2b' (ViT-B/14) | 'dinov2s' (ViT-S/14) |
                                'dinov2' (uses dinov2_variant param).
        dinov2_variant:         'b' (ViT-B/14, 768-dim) or 's' (ViT-S/14, 384-dim);
                                only used when rgb_backbone='dinov2'.
        unfreeze_dinov2_blocks: How many trailing DINOv2 transformer blocks to unfreeze.
        depth_backbone:         'cnn' (OriginalCMRNetDepth) | 'vmamba' (VMambaBackbone).
        flow_type:              'correlation' (single-scale) | 'multi_scale' (PWC+GMFlow).
        legacy_depth_branches:  True for checkpoints trained before _LlconvBranch refactor
                                (all three depth branches had the same architecture).
    """

    def __init__(
        self,
        image_size=(384, 1280),
        feat_dim=128,
        heatmap_dim_s2=128,
        vmamba_dims=(96, 192, 384, 768),
        vmamba_output_stage=2,
        use_cnn_fallback=False,
        dropout=0.0,
        use_reflectance=False,
        # ===== Ablation flags =====
        rgb_backbone='cnn',           # 'dinov2b' | 'dinov2s' | 'dinov2' | 'cnn'
        dinov2_variant='b',           # 'b'=ViT-B/14, 's'=ViT-S/14; used when rgb_backbone='dinov2'
        unfreeze_dinov2_blocks=0,
        depth_backbone='cnn',
        flow_type='correlation',
        # ===== Compatibility =====
        legacy_depth_branches=False,
    ):
        super().__init__()
        self.flow_type = flow_type
        input_lidar = 2 if use_reflectance else 1

        # ===== RGB Backbone =====
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
            from models.GCMLoc.GCMLoc_mapping import OriginalCMRNetRGB
            self.rgb_backbone = OriginalCMRNetRGB(feat_dim=feat_dim)

        # ===== Depth Backbone =====
        # Full 3-branch object kept for checkpoint key compatibility (branch_gt and
        # branch_init weights load from mapping ckpt but are never called in forward).
        # Only extract_gcmloc (branch_lhmap / _LlconvBranch) is used online.
        if depth_backbone == 'vmamba':
            from models.backbone.vmamba_backbone import VMambaBackbone
            self.depth_backbone = VMambaBackbone(
                out_dim=feat_dim,
                vmamba_dims=vmamba_dims,
                output_stage=vmamba_output_stage,
                in_channels=input_lidar,
                use_cnn_fallback=use_cnn_fallback,
            )
        else:
            from models.GCMLoc.GCMLoc_mapping import OriginalCMRNetDepth
            self.depth_backbone = OriginalCMRNetDepth(
                in_channels=input_lidar,
                feat_dim=feat_dim,
                vmamba_dims=vmamba_dims,
                output_stage=vmamba_output_stage,
                legacy_depth_branches=legacy_depth_branches,
            )

        # ===== Stage 2 Heatmap Decoder (multi-scale U-Net) =====
        # Uses Stage1HeatmapDecoder instead of HeatmapHead:
        #   H/16 (input_proj) → up3 → cat f2(H/8) → up2 → cat f1(H/4) → head → H/4
        # target_size = H/4 so that the final interpolate in Stage1HeatmapDecoder is skipped.
        # NOTE: Mapping checkpoint has HeatmapHead weights here → shape mismatch →
        #       this module starts from random init (all other modules load correctly).
        from models.heads.heads import Stage1HeatmapDecoder
        self.heatmap_head_s2 = Stage1HeatmapDecoder(
            vmamba_dims=vmamba_dims,
            fused_dim=feat_dim,
            heatmap_dim=heatmap_dim_s2,
            target_size=(image_size[0] // 4, image_size[1] // 4),
        )

        # ===== Flow Matching =====
        if flow_type == 'multi_scale':
            from models.fusion.multi_scale_flow import MultiScaleFlowMatcher
            self.flow_s2 = MultiScaleFlowMatcher(
                rgb_dim=feat_dim,
                depth_dims=vmamba_dims[:3],
                corr_dim=feat_dim,
                embed_dim=heatmap_dim_s2,
            )
        else:
            from models.GCMLoc.GCMLoc_mapping import OriginalCMRNetCorrelation
            self.flow_s2 = OriginalCMRNetCorrelation(
                rgb_dim=feat_dim,
                depth_dims=vmamba_dims[:3],
                embed_dim=heatmap_dim_s2,
                md=4,
            )

        # ===== Pose Regressor =====
        # w_x and w_q live inside PoseRegressor so they load from mapping checkpoint.
        from models.heads.heads import PoseRegressor
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
        print(f"[LocAblation] Unfroze last {n}/{total} DINOv2 blocks")

    def _init_weights(self):
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
    def forward(self, rgb: torch.Tensor, lidar: torch.Tensor):
        """
        Online localization forward pass with per-module timing.

        Returns:
            transl: (B, 3)
            rot:    (B, 4)
            w_x, w_q: learnable loss weights
            module_times: dict of module_name -> seconds (GPU-synced wall time)
        """
        import time

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        module_times = {}

        # ===== RGB Backbone =====
        _sync(); t0 = time.perf_counter()
        F_I = self.rgb_backbone(rgb)
        _sync(); module_times['rgb_backbone'] = time.perf_counter() - t0

        # ===== Depth Backbone =====
        _sync(); t0 = time.perf_counter()
        F_M, feats_m = self.depth_backbone.extract_gcmloc(lidar)
        _sync(); module_times['depth_backbone'] = time.perf_counter() - t0

        # ===== Heatmap Decoder =====
        _sync(); t0 = time.perf_counter()
        ch = self.heatmap_head_s2(F_M, feats_m)
        _sync(); module_times['heatmap_head_s2'] = time.perf_counter() - t0

        # ===== Flow Matching =====
        _sync(); t0 = time.perf_counter()
        flow_embed, _ = self.flow_s2(F_I, feats_m[:3])
        if flow_embed.shape[2:] != ch.shape[2:]:
            flow_embed = F.interpolate(
                flow_embed, size=ch.shape[2:],
                mode='bilinear', align_corners=False,
            )
        _sync(); module_times['flow_s2'] = time.perf_counter() - t0

        # ===== Heatmap-Weighted Pooling + Pose Regression =====
        _sync(); t0 = time.perf_counter()
        xf = (flow_embed * ch).sum(dim=(2, 3))
        transl, rot = self.pose_reg_s2(xf)
        _sync(); module_times['pose_reg_s2'] = time.perf_counter() - t0

        return transl, rot, self.pose_reg_s2.w_x, self.pose_reg_s2.w_q, module_times
