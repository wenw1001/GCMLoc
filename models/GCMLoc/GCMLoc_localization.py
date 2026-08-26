"""
GCMLocLocalization — online localization network (Stage 2 only).

    model = GCMLocLocalization(image_size=(384, 1280), dinov2_variant='b')
    transl, rot = model(rgb, lidar)                  # lidar = projected distilled map (train_save.py)
    transl, rot = model(rgb, lidar, timing=times)    # also fills `times` with per-module seconds

    F_I = DINOv2(rgb); F_M, skips = CNN_map(lidar)
    heatmap = HeatmapDecoder(F_M, skips)              (H/4 resolution)
    (t, q)  = PoseRegressor(Sum heatmap * GCMFlow(F_I, F_M))

Initialised from a mapping checkpoint (strict=False): rgb_backbone, depth_backbone,
flow_s2 and pose_reg_s2 are loaded; heatmap_head_s2 is a HeatmapDecoder here (the
offline Stage 2 uses the light HeatmapHead) and therefore starts from scratch.
depth_backbone keeps all three branches so the keys match; only branch_lhmap runs online.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import DINOv2Backbone, DepthCNN
from models.fusion import GCMFlow
from models.heads import HeatmapDecoder, PoseRegressor
from models.GCMLoc.GCMLoc_mapping import DEPTH_DIMS

TIMED_MODULES = ('rgb_backbone', 'depth_backbone', 'heatmap_head_s2', 'flow_s2', 'pose_reg_s2')


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class GCMLocLocalization(nn.Module):
    def __init__(self, image_size=(384, 1280), dinov2_variant='b', feat_dim=128,
                 heatmap_dim_s2=128, unfreeze_dinov2_blocks=4):
        super().__init__()
        self.image_size = tuple(image_size)
        self.rgb_backbone = DINOv2Backbone(out_dim=feat_dim, variant=dinov2_variant,
                                           unfreeze_last_blocks=unfreeze_dinov2_blocks)
        self.depth_backbone = DepthCNN(feat_dim=feat_dim, dims=DEPTH_DIMS)
        self.heatmap_head_s2 = HeatmapDecoder(dims=DEPTH_DIMS, fused_dim=feat_dim, heatmap_dim=heatmap_dim_s2,
                                              target_size=(image_size[0] // 4, image_size[1] // 4))
        self.flow_s2 = GCMFlow(rgb_dim=feat_dim, depth_dims=DEPTH_DIMS[:3],
                               corr_dim=feat_dim, embed_dim=heatmap_dim_s2)
        self.pose_reg_s2 = PoseRegressor(in_dim=heatmap_dim_s2)
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.startswith('rgb_backbone.dino'):
                continue
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, rgb, lidar, timing=None):
        """
        rgb (B, 3, H, W), lidar (B, 1, H, W) -> transl (B, 3), rot (B, 4) unit quaternion.
        timing: optional dict; if given, GPU-synchronised wall time (s) of each module in
                TIMED_MODULES is written into it.
        """
        timed = timing is not None

        if timed: _sync(); t0 = time.perf_counter()
        F_I = self.rgb_backbone(rgb)
        if timed: _sync(); timing['rgb_backbone'] = time.perf_counter() - t0; t0 = time.perf_counter()
        F_M, feats_m = self.depth_backbone.extract_map(lidar)
        if timed: _sync(); timing['depth_backbone'] = time.perf_counter() - t0; t0 = time.perf_counter()
        heatmap = self.heatmap_head_s2(F_M, feats_m)
        if timed: _sync(); timing['heatmap_head_s2'] = time.perf_counter() - t0; t0 = time.perf_counter()
        flow_embed, _ = self.flow_s2(F_I, feats_m[:3])
        if flow_embed.shape[2:] != heatmap.shape[2:]:
            flow_embed = F.interpolate(flow_embed, size=heatmap.shape[2:], mode='bilinear', align_corners=False)
        if timed: _sync(); timing['flow_s2'] = time.perf_counter() - t0; t0 = time.perf_counter()
        transl, rot = self.pose_reg_s2((flow_embed * heatmap).sum(dim=(2, 3)))
        if timed: _sync(); timing['pose_reg_s2'] = time.perf_counter() - t0

        return transl, rot
