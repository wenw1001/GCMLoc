"""
Three-branch CNN depth encoder for GCMLoc (inherited from CMRNet / LHMap-loc).

    DepthCNN(feat_dim=128, dims=(96, 192, 384, 768))
        extract_depth_gt(depth)    -> branch_gt    : GT-pose depth image      (Stage 1, heatmap)
        extract_depth_init(lidar)  -> branch_init  : initial-pose depth image (Stage 1, matching)
        extract_map(lidar_sparse)  -> branch_lhmap : distilled sparse map     (Stage 2, matching)
    Each call returns (primary, [f1, f2, f3, f4]):
        primary : (B, feat_dim, H/8, W/8)   unified-dimension feature
        f1..f4  : (B, dims[i], H/4 .. H/32) multi-scale features (flow matcher + decoder skips)

All convolutions are Conv2d(bias=True) + LeakyReLU(0.1) without BatchNorm: the projected
depth images are very sparse and batch statistics would be dominated by zeros.
The three branches have independent weights and slightly different layouts (as in CMRNet):
    branch_gt    dconv : H/1 .. H/8 full blocks, H/16 & H/32 single stride-2 convs
    branch_init  lconv : H/1 .. H/32 full blocks
    branch_lhmap llconv: first layer stride 2 (no H/1 level), H/2 .. H/32 full blocks
(module names branch_lhmap / level* / proj_* are kept for checkpoint compatibility)
"""

import torch
import torch.nn as nn


def _conv(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True),
    )


def _proj1x1(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=True),
        nn.LeakyReLU(0.1, inplace=True),
    )


class _LconvBranch(nn.Module):
    """branch_init: in -> 8 (H/1) -> 16 (H/2) -> 32 (H/4) -> 64 (H/8) -> 96 (H/16) -> 128 (H/32)"""

    def __init__(self, in_ch, dims, feat_dim):
        super().__init__()
        C1, C2, C3, C4 = dims
        self.level0 = nn.Sequential(_conv(in_ch, 8), _conv(8, 8), _conv(8, 8))
        self.level1 = nn.Sequential(_conv(8, 16, 2), _conv(16, 16), _conv(16, 16))
        self.level2 = nn.Sequential(_conv(16, 32, 2), _conv(32, 32), _conv(32, 32))
        self.level3 = nn.Sequential(_conv(32, 64, 2), _conv(64, 64), _conv(64, 64))
        self.level4 = nn.Sequential(_conv(64, 96, 2), _conv(96, 96), _conv(96, 96))
        self.level5 = nn.Sequential(_conv(96, 128, 2), _conv(128, 128), _conv(128, 128))
        self.proj_f1 = _proj1x1(32, C1)
        self.proj_f2 = _proj1x1(64, C2)
        self.proj_f3 = _proj1x1(96, C3)
        self.proj_f4 = _proj1x1(128, C4)
        self.proj_primary = _proj1x1(64, feat_dim)

    def forward(self, x):
        x = self.level1(self.level0(x))
        r1 = self.level2(x)
        r2 = self.level3(r1)
        r3 = self.level4(r2)
        r4 = self.level5(r3)
        return self.proj_primary(r2), [self.proj_f1(r1), self.proj_f2(r2), self.proj_f3(r3), self.proj_f4(r4)]


class _DconvBranch(nn.Module):
    """branch_gt: in -> 8 (H/1) -> 16 (H/2) -> 32 (H/4) -> 64 (H/8); H/16, H/32 via single convs"""

    def __init__(self, in_ch, dims, feat_dim):
        super().__init__()
        C1, C2, C3, C4 = dims
        self.level0 = nn.Sequential(_conv(in_ch, 8), _conv(8, 8), _conv(8, 8))
        self.level1 = nn.Sequential(_conv(8, 16, 2), _conv(16, 16), _conv(16, 16))
        self.level2 = nn.Sequential(_conv(16, 32, 2), _conv(32, 32), _conv(32, 32))
        self.level3 = nn.Sequential(_conv(32, 64, 2), _conv(64, 64), _conv(64, 64))
        self.level4_lite = _conv(64, 96, 2)
        self.level5_lite = _conv(96, 128, 2)
        self.proj_f1 = _proj1x1(32, C1)
        self.proj_f2 = _proj1x1(64, C2)
        self.proj_f3 = _proj1x1(96, C3)
        self.proj_f4 = _proj1x1(128, C4)
        self.proj_primary = _proj1x1(64, feat_dim)

    def forward(self, x):
        x = self.level1(self.level0(x))
        r1 = self.level2(x)
        r2 = self.level3(r1)
        r3 = self.level4_lite(r2)
        r4 = self.level5_lite(r3)
        return self.proj_primary(r2), [self.proj_f1(r1), self.proj_f2(r2), self.proj_f3(r3), self.proj_f4(r4)]


class _LlconvBranch(nn.Module):
    """branch_lhmap: in -> 16 (H/2, stride-2 first conv) -> 32 (H/4) -> 64 (H/8) -> 96 (H/16) -> 128 (H/32)"""

    def __init__(self, in_ch, dims, feat_dim):
        super().__init__()
        C1, C2, C3, C4 = dims
        self.level1 = nn.Sequential(_conv(in_ch, 16, 2), _conv(16, 16), _conv(16, 16))
        self.level2 = nn.Sequential(_conv(16, 32, 2), _conv(32, 32), _conv(32, 32))
        self.level3 = nn.Sequential(_conv(32, 64, 2), _conv(64, 64), _conv(64, 64))
        self.level4 = nn.Sequential(_conv(64, 96, 2), _conv(96, 96), _conv(96, 96))
        self.level5 = nn.Sequential(_conv(96, 128, 2), _conv(128, 128), _conv(128, 128))
        self.proj_f1 = _proj1x1(32, C1)
        self.proj_f2 = _proj1x1(64, C2)
        self.proj_f3 = _proj1x1(96, C3)
        self.proj_f4 = _proj1x1(128, C4)
        self.proj_primary = _proj1x1(64, feat_dim)

    def forward(self, x):
        x = self.level1(x)
        r1 = self.level2(x)
        r2 = self.level3(r1)
        r3 = self.level4(r2)
        r4 = self.level5(r3)
        return self.proj_primary(r2), [self.proj_f1(r1), self.proj_f2(r2), self.proj_f3(r3), self.proj_f4(r4)]


class DepthCNN(nn.Module):
    def __init__(self, feat_dim: int = 128, dims: tuple = (96, 192, 384, 768), in_channels: int = 1):
        super().__init__()
        self.dims = tuple(dims)
        self.branch_gt = _DconvBranch(in_channels, dims, feat_dim)
        self.branch_init = _LconvBranch(in_channels, dims, feat_dim)
        self.branch_lhmap = _LlconvBranch(in_channels, dims, feat_dim)

    def extract_depth_gt(self, depth):
        return self.branch_gt(depth)

    def extract_depth_init(self, lidar):
        return self.branch_init(lidar)

    def extract_map(self, lidar_sparse):
        return self.branch_lhmap(lidar_sparse)
