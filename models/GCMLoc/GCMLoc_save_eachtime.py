"""
GCMLoc — Save Model (Stage 1 only).

Standalone counterpart of GCMLoc_mapping.py, analogous to the original:
    CMRNet_single_mapping.py  →  GCMLoc_mapping.py  (mapping / training)
    CMRNet_single_save.py     →  GCMLoc_save.py       (point-cloud saving)

Loads Stage-1 weights from a train_mapping.py checkpoint (strict=False;
Stage-2 keys in the checkpoint are simply ignored).

Weight compatibility:
    All Stage-1 module names match GCMLocMapping exactly:
        self.rgb_backbone, self.depth_backbone, (self.fusion),
        self.heatmap_decoder_s1, self.flow_s1, self.pose_reg_s1

Usage (in train_save.py):
    ckpt  = torch.load(weights, map_location='cpu')
    cfg   = ckpt['config']                  # original train_mapping config
    model = GCMLocSave(**build_kwargs(cfg), save_root=save_root)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    transl0, rot0, _, _ = model(rgb, lidar, depth, flagt, u, v, pcl, info)
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCMLocSave(nn.Module):
    """
    v2 Save Model — Stage 1 only.

    Runs the same Stage-1 forward pass as GCMLocMapping (heatmap + coarse
    pose), selects top-K points by heatmap response, and saves the 3-D
    coordinates to disk as .npy files for use by the localization stage.

    Constructor arguments and module names are identical to the Stage-1 portion
    of GCMLocMapping so that train_mapping.py checkpoints load cleanly.
    """

    def __init__(
        self,
        image_size=(384, 1280),
        feat_dim=128,
        heatmap_dim_s1=64,
        topk_points=5000,
        vmamba_dims=(96, 192, 384, 768),
        vmamba_output_stage=2,
        use_cnn_fallback=False,
        dropout=0.0,
        use_reflectance=False,
        # ===== Ablation flags (must match the trained model) =====
        rgb_backbone='cnn',           # 'dinov2b' | 'dinov2s' | 'dinov2' | 'cnn'
        dinov2_variant='b',           # 'b'=ViT-B/14, 's'=ViT-S/14; used when rgb_backbone='dinov2'
        unfreeze_dinov2_blocks=0,
        depth_backbone='cnn',         # 'vmamba' | 'cnn'
        use_cross_fusion=False,
        flow_type='correlation',      # 'multi_scale' | 'correlation'
        # ===== Compatibility =====
        legacy_depth_branches=False,  # True for checkpoints trained before _LlconvBranch refactor
        # ===== Save config =====
        save_root='./KITTI_ODOMETRY/sequences',
        save_name="WenMap_pcl",
    ):
        super().__init__()
        self.image_size    = image_size
        self.feat_dim      = feat_dim
        self.topk_points   = topk_points
        self.flow_type     = flow_type
        self.use_cross_fusion = use_cross_fusion
        self.save_root     = save_root
        self.save_name     = save_name

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
        else:
            from models.GCMLoc.GCMLoc_mapping import OriginalCMRNetRGB
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
            from models.GCMLoc.GCMLoc_mapping import OriginalCMRNetDepth
            self.depth_backbone = OriginalCMRNetDepth(
                in_channels=input_lidar, feat_dim=feat_dim,
                vmamba_dims=vmamba_dims, output_stage=vmamba_output_stage,
                legacy_depth_branches=legacy_depth_branches,
            )

        # ===== Cross-Modal Fusion (optional) =====
        if use_cross_fusion:
            from models.fusion.cross_modal_fusion import CrossModalFusion
            self.fusion = CrossModalFusion(feat_dim=feat_dim)

        # ===== Stage-1 Heatmap Decoder =====
        from models.heads.heads import Stage1HeatmapDecoder, PoseRegressor
        self.heatmap_decoder_s1 = Stage1HeatmapDecoder(
            vmamba_dims=vmamba_dims, fused_dim=feat_dim,
            heatmap_dim=heatmap_dim_s1, target_size=image_size,
        )

        # ===== Stage-1 Flow Matching =====
        if flow_type == 'multi_scale':
            from models.fusion.multi_scale_flow import MultiScaleFlowMatcher
            self.flow_s1 = MultiScaleFlowMatcher(
                rgb_dim=feat_dim, depth_dims=vmamba_dims[:3],
                corr_dim=feat_dim, embed_dim=heatmap_dim_s1,
            )
        else:
            from models.GCMLoc.GCMLoc_mapping import OriginalCMRNetCorrelation
            self.flow_s1 = OriginalCMRNetCorrelation(
                rgb_dim=feat_dim, depth_dims=vmamba_dims[:3],
                embed_dim=heatmap_dim_s1, md=4,
            )

        # ===== Stage-1 Pose Regressor =====
        self.pose_reg_s1 = PoseRegressor(
            in_dim=heatmap_dim_s1, hidden=512, dropout=dropout,
        )

        self._init_weights()

    # ------------------------------------------------------------------
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
    def forward(self, rgb, lidar, depth, flagt, u, v, pcl, info):
        """
        Stage-1 forward + GCMLoc point-cloud saving.

        Signature matches the original CMRNet_single_save.CMRNet so that
        train_save.py can call it identically.

        Args:
            rgb:    (B, 3, H, W)
            lidar:  (B, 1, H, W)  initial projected depth (misaligned)
            depth:  (B, 1, H, W)  GT projected depth
            flagt:  (B, 1, H, W)  valid-point mask
            u, v:   (B, 1, H, W)  pixel coordinates (row / col)
            pcl:    (B, 3, H, W)  3-D point grid (x / y / z in camera space)
            info:   list[Tensor]  length B; each entry = [seq_idx, frame_idx]

        Returns:
            (transl0, rot0, transl0, rot0, module_times)
            module_times: dict of module_name -> seconds (GPU-synced wall time)
        """
        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        B = rgb.shape[0]
        module_times = {}

        # ===== RGB Backbone =====
        _sync(); t0 = time.perf_counter()
        F_I = self.rgb_backbone(rgb)
        _sync(); module_times['rgb_backbone'] = time.perf_counter() - t0

        # ===== Depth Backbone (GT branch) =====
        _sync(); t0 = time.perf_counter()
        F_D_gt, feats_gt = self.depth_backbone.extract_depth_gt(depth)
        _sync(); module_times['depth_backbone_gt'] = time.perf_counter() - t0

        # ===== Depth Backbone (Init branch) =====
        _sync(); t0 = time.perf_counter()
        F_D_init, feats_init = self.depth_backbone.extract_depth_init(lidar)
        _sync(); module_times['depth_backbone_init'] = time.perf_counter() - t0

        # ===== Cross-Modal Fusion (optional) + Heatmap Decoder =====
        _sync(); t0 = time.perf_counter()
        if self.use_cross_fusion:
            F_fused = self.fusion(F_D_gt, F_I)
        else:
            F_fused = F_D_gt
        h_lidar = self.heatmap_decoder_s1(F_fused, feats_gt)
        _sync(); module_times['heatmap_decoder_s1'] = time.perf_counter() - t0

        # ===== Flow Matching =====
        _sync(); t0 = time.perf_counter()
        flow_embed_s1, _ = self.flow_s1(F_I, feats_init[:3])
        if flow_embed_s1.shape[2:] != h_lidar.shape[2:]:
            flow_embed_s1 = F.interpolate(
                flow_embed_s1, h_lidar.shape[2:], mode='bilinear', align_corners=False)
        _sync(); module_times['flow_s1'] = time.perf_counter() - t0

        # ===== Pose Regression =====
        _sync(); t0 = time.perf_counter()
        x_s1 = torch.sum(flow_embed_s1 * h_lidar, dim=(2, 3))
        transl0, rot0 = self.pose_reg_s1(x_s1)
        _sync(); module_times['pose_reg_s1'] = time.perf_counter() - t0

        # ===== Top-K point selection (no save) =====
        _sync(); t0 = time.perf_counter()
        h_collapsed = h_lidar.sum(dim=1, keepdim=True) * flagt
        H, W = h_collapsed.shape[2], h_collapsed.shape[3]
        h_flat = h_collapsed.reshape(B, H * W)
        h_flat.topk(self.topk_points, dim=-1, largest=True, sorted=False)
        _sync(); module_times['topk_select'] = time.perf_counter() - t0

        return transl0, rot0, transl0, rot0, module_times
