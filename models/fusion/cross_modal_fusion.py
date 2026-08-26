"""
Cross-modal semantic-geometric fusion (Stage 1, before the heatmap decoder).

    CrossModalFusion(feat_dim=128)
        forward(F_D, F_I) -> F_fused (B, feat_dim, H', W')

The RGB feature F_I is bilinearly resized to the depth feature resolution, the two
are concatenated along the channel axis and fused by a two-step convolution
(1x1 reduce -> 3x3 integrate, BN + ReLU). Only the fused feature feeds the heatmap
decoder; the flow matcher keeps working on the separate modalities.

state_dict layout (unchanged from the trained checkpoints): fusion.bottleneck.{0..4}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConcatBottleneck(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, F_D, F_I):
        return self.bottleneck(torch.cat([F_D, F_I], dim=1))


class CrossModalFusion(nn.Module):
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.fusion = _ConcatBottleneck(feat_dim)

    def forward(self, F_D: torch.Tensor, F_I: torch.Tensor) -> torch.Tensor:
        """F_D (B, C, H', W') geometric feature, F_I (B, C, h, w) semantic feature -> (B, C, H', W')"""
        if F_I.shape[2:] != F_D.shape[2:]:
            F_I = F.interpolate(F_I, size=F_D.shape[2:], mode="bilinear", align_corners=False)
        return self.fusion(F_D, F_I)
