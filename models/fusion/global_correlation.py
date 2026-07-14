"""
Global Correlation Module for GCMLoc (M4)

架構規劃書 §5 — GMFlow 式全局特徵相關性計算。

介面（符合 GCMLoc_mapping.py）：
    GlobalCorrelationModule(feat_dim=128, embed_dim=128)
    forward(F_A, F_B) → (E_A, E_B): each (B, embed_dim, H', W')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalCorrelationModule(nn.Module):
    """
    GMFlow 式全局相關模組（M4）。

    步驟：
        1. L2 正規化
        2. 全局相關矩陣 C_corr ∈ R^{N×N}，N=H'×W'
        3. 雙向 Softmax（Attn_A→B, Attn_B→A）
        4. 特徵聚合 + MLP → 增強嵌入 E_A, E_B

    Args:
        feat_dim (int): 輸入特徵通道數 C。
        embed_dim (int): 輸出嵌入通道數 C_e。
        temperature (float | None): 相關矩陣縮放溫度 τ；None=可學習標量。
        downsample_factor (int): 在降採樣後計算相關矩陣（節省記憶體）；1=不降採樣。
    """

    def __init__(
        self,
        feat_dim: int = 128,
        embed_dim: int = 128,
        temperature: float = None,
        downsample_factor: int = 1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.downsample_factor = downsample_factor

        # 溫度係數
        if temperature is None:
            self.log_tau = nn.Parameter(torch.tensor(-2.3026))  # ln(0.1)
            self._learnable_tau = True
        else:
            self.register_buffer("tau", torch.tensor(temperature))
            self._learnable_tau = False

        # MLP：concat(原始+聚合) = 2*feat_dim → embed_dim
        # 使用 GroupNorm 取代 BatchNorm：batch_size=2 時 BN 統計量極不穩定
        num_groups = min(8, embed_dim)
        self.mlp_A = nn.Sequential(
            nn.Conv2d(feat_dim * 2, embed_dim, 1, bias=False),
            nn.GroupNorm(num_groups, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.GroupNorm(num_groups, embed_dim),
        )
        self.mlp_B = nn.Sequential(
            nn.Conv2d(feat_dim * 2, embed_dim, 1, bias=False),
            nn.GroupNorm(num_groups, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.GroupNorm(num_groups, embed_dim),
        )
        self.act = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def tau_value(self) -> torch.Tensor:
        if self._learnable_tau:
            return self.log_tau.exp().clamp(min=1e-4)
        return self.tau

    def forward(self, F_A: torch.Tensor, F_B: torch.Tensor):
        """
        Args:
            F_A: (B, C, H', W')
            F_B: (B, C, H', W')
        Returns:
            E_A: (B, embed_dim, H', W')
            E_B: (B, embed_dim, H', W')
        """
        B, C, H_orig, W_orig = F_A.shape

        # 可選降採樣
        if self.downsample_factor > 1:
            H_d, W_d = H_orig // self.downsample_factor, W_orig // self.downsample_factor
            A = F.interpolate(F_A, (H_d, W_d), mode="bilinear", align_corners=False)
            B_ = F.interpolate(F_B, (H_d, W_d), mode="bilinear", align_corners=False)
        else:
            A, B_, H_d, W_d = F_A, F_B, H_orig, W_orig

        N = H_d * W_d

        # L2 正規化
        A_flat = F.normalize(A.flatten(2), p=2, dim=1)   # (B, C, N)
        B_flat = F.normalize(B_.flatten(2), p=2, dim=1)

        # 全局相關矩陣
        C_corr = torch.bmm(A_flat.permute(0, 2, 1), B_flat) / self.tau_value  # (B, N, N)
        Attn_A2B = F.softmax(C_corr, dim=-1)
        Attn_B2A = F.softmax(C_corr, dim=-2)

        # 特徵聚合
        A_agg = torch.bmm(A_flat, Attn_A2B).reshape(B, C, H_d, W_d)
        B_agg = torch.bmm(B_flat, Attn_B2A.transpose(-1, -2)).reshape(B, C, H_d, W_d)

        # 上採樣（若有降採樣）
        if self.downsample_factor > 1:
            A_agg = F.interpolate(A_agg, (H_orig, W_orig), mode="bilinear", align_corners=False)
            B_agg = F.interpolate(B_agg, (H_orig, W_orig), mode="bilinear", align_corners=False)

        E_A = self.act(self.mlp_A(torch.cat([F_A, A_agg], dim=1)))
        E_B = self.act(self.mlp_B(torch.cat([F_B, B_agg], dim=1)))
        return E_A, E_B
