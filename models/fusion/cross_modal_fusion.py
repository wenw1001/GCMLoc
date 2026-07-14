"""
Cross-Modal Semantic-Geometric Fusion Module for GCMLoc (M3)

架構規劃書 §4 — 跨模態語意-幾何融合模組。

介面（符合 GCMLoc_mapping.py）：
    CrossModalFusion(feat_dim=128, mode='A')
    forward(F_D, F_I) → F_fused: (B, feat_dim, H', W')

支援三種融合方案：
    'A': Concat + Bottleneck（推薦基線）
    'B': Cross-Attention（$O(N^2)$，用於小解析度）
    'C': Gated Fusion（動態物件抑制場景推薦）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 方案 A：Concat + Bottleneck
# ---------------------------------------------------------------------------

class ConcatBottleneckFusion(nn.Module):
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

    def forward(self, F_D: torch.Tensor, F_I: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(torch.cat([F_D, F_I], dim=1))


# ---------------------------------------------------------------------------
# 方案 B：Cross-Attention
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    def __init__(self, feat_dim: int, num_heads: int = 8):
        super().__init__()
        assert feat_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.W_Q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_K = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_V = nn.Linear(feat_dim, feat_dim, bias=False)
        self.proj_out = nn.Linear(feat_dim, feat_dim, bias=False)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, F_D: torch.Tensor, F_I: torch.Tensor) -> torch.Tensor:
        B, C, H, W = F_D.shape
        N = H * W
        fd = F_D.flatten(2).permute(0, 2, 1)   # (B, N, C)
        fi = F_I.flatten(2).permute(0, 2, 1)   # (B, N, C)

        Q = self.W_Q(fd).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(fi).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(fi).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, N, C)
        out = self.norm(self.proj_out(out) + fd)
        return out.permute(0, 2, 1).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# 方案 C：Gated Fusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, F_D: torch.Tensor, F_I: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([F_D, F_I], dim=1))
        return g * F_I + (1.0 - g) * F_D


# ---------------------------------------------------------------------------
# 統一入口：CrossModalFusion
# ---------------------------------------------------------------------------

class CrossModalFusion(nn.Module):
    """
    跨模態語意-幾何融合模組（M3）。

    Args:
        feat_dim (int): 輸入/輸出通道數 C。
        mode (str): 'A' | 'B' | 'C'
        num_heads (int): 方案 B 的注意力頭數。
    """

    def __init__(self, feat_dim: int = 128, mode: str = "A", num_heads: int = 8):
        super().__init__()
        assert mode in ("A", "B", "C"), f"mode 必須為 'A'/'B'/'C'，收到：{mode}"

        if mode == "A":
            self.fusion = ConcatBottleneckFusion(feat_dim)
        elif mode == "B":
            self.fusion = CrossAttentionFusion(feat_dim, num_heads)
        else:
            self.fusion = GatedFusion(feat_dim)

    def forward(self, F_D: torch.Tensor, F_I: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F_D: (B, C, H', W')  — 幾何特徵（VMamba 輸出）
            F_I: (B, C, H_d, W_d) — 語意特徵（DINOv2 投影，解析度可不同）
        Returns:
            F_fused: (B, C, H', W')
        """
        # 對齊 F_I 到 F_D 的空間解析度
        if F_I.shape[2:] != F_D.shape[2:]:
            F_I = F.interpolate(
                F_I, size=F_D.shape[2:], mode="bilinear", align_corners=False
            )
        return self.fusion(F_D, F_I)
