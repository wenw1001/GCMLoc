"""
GCMLocMapping — offline mapping network (Stage 1 + Stage 2).

    model = GCMLocMapping(image_size=(384, 1280), dinov2_variant='b')
    transl, rot, transl0, rot0 = model(rgb, lidar, depth, flagt, u, v)   # training
    transl0, rot0, heatmap, F_I = model.stage1(rgb, lidar, depth)          # Stage 1 only
    indices = model.select_points(heatmap, flagt)                          # top-K map points

Inputs (all (B, ., H, W), H x W = image_size):
    rgb    (3)  image
    lidar  (1)  depth image of the local map projected with the *initial* (perturbed) pose
    depth  (1)  depth image projected with the *ground-truth* pose
    flagt  (1)  validity mask of `depth` (visible in both projections)
    u, v   (1)  pixel correspondence GT-projection -> initial-projection (row / col)

Stage 1  F_I = DINOv2(rgb); F_gt = CNN_gt(depth); F_init = CNN_init(lidar)
         heatmap = HeatmapDecoder(CrossModalFusion(F_gt, F_I), skips(F_gt))
         (t0, q0) = PoseRegressor(Sum heatmap * GCMFlow(F_I, F_init))
Select   top-K pixels of Sum_c heatmap * flagt, mapped through (u, v) into the initial
         projection -> sparse map `lidar_filtered` (this is the distilled map that
         train_save.py writes to disk)
Stage 2  F_M = CNN_map(lidar_filtered)
         (t, q) = PoseRegressor(Sum HeatmapHead(F_M) * GCMFlow(F_I, F_M))

Module names (rgb_backbone, depth_backbone, fusion, heatmap_decoder_s1, flow_s1,
pose_reg_s1, heatmap_head_s2, flow_s2, pose_reg_s2) are those of the trained checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import DINOv2Backbone, DepthCNN
from models.fusion import CrossModalFusion, GCMFlow
from models.heads import HeatmapDecoder, HeatmapHead, PoseRegressor

DEPTH_DIMS = (96, 192, 384, 768)     # channels of the depth features f1 (H/4) .. f4 (H/32)


class GCMLocMapping(nn.Module):
    def __init__(self, image_size=(384, 1280), dinov2_variant='b', feat_dim=128,
                 heatmap_dim_s1=64, heatmap_dim_s2=128, topk_points=5000,
                 unfreeze_dinov2_blocks=4):
        super().__init__()
        self.image_size = tuple(image_size)
        self.feat_dim = feat_dim
        self.topk_points = topk_points

        self.rgb_backbone = DINOv2Backbone(out_dim=feat_dim, variant=dinov2_variant,
                                           unfreeze_last_blocks=unfreeze_dinov2_blocks)
        self.depth_backbone = DepthCNN(feat_dim=feat_dim, dims=DEPTH_DIMS)

        # Stage 1
        self.fusion = CrossModalFusion(feat_dim=feat_dim)
        self.heatmap_decoder_s1 = HeatmapDecoder(dims=DEPTH_DIMS, fused_dim=feat_dim,
                                                 heatmap_dim=heatmap_dim_s1, target_size=image_size)
        self.flow_s1 = GCMFlow(rgb_dim=feat_dim, depth_dims=DEPTH_DIMS[:3],
                               corr_dim=feat_dim, embed_dim=heatmap_dim_s1)
        self.pose_reg_s1 = PoseRegressor(in_dim=heatmap_dim_s1)

        # Stage 2
        self.heatmap_head_s2 = HeatmapHead(in_dim=feat_dim, heatmap_dim=heatmap_dim_s2)
        self.flow_s2 = GCMFlow(rgb_dim=feat_dim, depth_dims=DEPTH_DIMS[:3],
                               corr_dim=feat_dim, embed_dim=heatmap_dim_s2)
        self.pose_reg_s2 = PoseRegressor(in_dim=heatmap_dim_s2)

        self._init_weights()

    def _init_weights(self):
        """kaiming_normal (fan_in) for every trainable conv / linear outside the DINOv2 ViT
        (the ViT handles its own unfrozen blocks, see DINOv2Backbone)."""
        for name, m in self.named_modules():
            if name.startswith('rgb_backbone.dino'):
                continue
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    # ------------------------------------------------------------------
    @staticmethod
    def _pool(flow_embed, heatmap):
        if flow_embed.shape[2:] != heatmap.shape[2:]:
            flow_embed = F.interpolate(flow_embed, heatmap.shape[2:], mode='bilinear', align_corners=False)
        return torch.sum(flow_embed * heatmap, dim=(2, 3))

    def stage1(self, rgb, lidar, depth):
        """Returns (transl0, rot0, heatmap (B, hm_s1, H, W), F_I)."""
        F_I = self.rgb_backbone(rgb)
        F_gt, feats_gt = self.depth_backbone.extract_depth_gt(depth)
        _, feats_init = self.depth_backbone.extract_depth_init(lidar)

        heatmap = self.heatmap_decoder_s1(self.fusion(F_gt, F_I), feats_gt)
        flow_embed, _ = self.flow_s1(F_I, feats_init[:3])
        transl0, rot0 = self.pose_reg_s1(self._pool(flow_embed, heatmap))
        return transl0, rot0, heatmap, F_I

    def select_points(self, heatmap, flagt):
        """Top-K pixel indices (B, K) of the channel-summed heatmap restricted to flagt."""
        B = heatmap.shape[0]
        score = (heatmap.sum(dim=1, keepdim=True) * flagt).reshape(B, -1)
        return score.topk(self.topk_points, dim=-1, largest=True, sorted=False)[1]

    def stage2(self, F_I, lidar_filtered):
        F_M, feats_m = self.depth_backbone.extract_map(lidar_filtered)
        flow_embed, _ = self.flow_s2(F_I, feats_m[:3])
        return self.pose_reg_s2(self._pool(flow_embed, self.heatmap_head_s2(F_M)))

    def forward(self, rgb, lidar, depth, flagt, u, v):
        """Returns (transl, rot, transl0, rot0): Stage-2 and Stage-1 pose corrections."""
        B, _, H, W = rgb.shape
        transl0, rot0, heatmap, F_I = self.stage1(rgb, lidar, depth)

        indices = self.select_points(heatmap, flagt)
        u_flat = u.long().reshape(B, H * W)
        v_flat = v.long().reshape(B, H * W)
        mask = torch.zeros((B, 1, H, W), device=rgb.device)
        for i in range(B):
            mask[i, 0, u_flat[i, indices[i]], v_flat[i, indices[i]]] = 1
        lidar_filtered = mask * lidar

        transl, rot = self.stage2(F_I, lidar_filtered)
        return transl, rot, transl0, rot0
