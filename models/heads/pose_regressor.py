"""
Pose Regressor for GCMLoc — Stage 2

架構規劃書 §1.2 流程：
    E_D, E_M  →  PoseRegressor  →  T_1（Stage 2 精煉位姿）

功能：
    1. 接收 Global Correlation Module 輸出的 E_D, E_M 嵌入特徵
    2. 加權融合（使用 Stage 1 heatmap 進行空間加權 pooling）
    3. 通過 FC 分支分別回歸平移向量 t ∈ R³ 與旋轉四元數 q ∈ R⁴

與原版 CMRNet 的對應關係：
    原版的 fc1 → fc1_trasl/fc1_rot → fc2_trasl/fc2_rot 結構保留，
    改為接收 E_D, E_M 融合特徵取代 PWCNet 的 cost volume 輸出。

輸入：
    E_D:           (B, C_e, H', W')  — 深度嵌入
    E_M:           (B, C_e, H', W')  — 地圖嵌入
    heatmap_prob:  (B, 1, H', W')  — Stage 1 機率 heatmap（用於加權 pooling）
                   若為 None，則使用平均 pooling
輸出：
    transl:  (B, 3)   — 平移向量 t
    rot:     (B, 4)   — 正規化旋轉四元數 q
    w_x:     scalar   — 平移損失的可學習權重（log scale）
    w_q:     scalar   — 旋轉損失的可學習權重（log scale）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseRegressor(nn.Module):
    """
    Stage 2 位姿回歸器。

    Args:
        in_channels (int): E_D 與 E_M 的通道數 C_e（兩者相同）。
        hidden_dim (int): FC 隱藏層維度，預設 512。
        dropout (float): Dropout 比率，預設 0.0。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels

        # --- 空間特徵提煉（concat E_D + E_M → 2*C_e → C_e）---
        self.feat_merge = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        # --- FC 回歸頭（對應原版 CMRNet 結構）---
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.1)

        # 平移分支
        self.fc1_transl = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2_transl = nn.Linear(hidden_dim // 2, 3)

        # 旋轉分支
        self.fc1_rot = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2_rot = nn.Linear(hidden_dim // 2, 4)

        # --- 可學習位姿損失權重（對應原版 w_x, w_q）---
        self.w_x = nn.Parameter(torch.tensor([0.0]))    # 平移損失權重
        self.w_q = nn.Parameter(torch.tensor([-2.5]))   # 旋轉損失權重

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _weighted_pool(
        self,
        feat: torch.Tensor,
        heatmap_prob: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Heatmap 加權空間 Pooling（Soft Attention Pooling）。

        Args:
            feat:          (B, C, H', W')
            heatmap_prob:  (B, 1, H', W') or None
        Returns:
            pooled: (B, C)
        """
        if heatmap_prob is None:
            # 退化為全域平均 pooling
            return feat.mean(dim=[2, 3])

        # 確保 heatmap 解析度與 feat 對齊
        if heatmap_prob.shape[2:] != feat.shape[2:]:
            heatmap_prob = F.interpolate(
                heatmap_prob, size=feat.shape[2:], mode="bilinear", align_corners=False
            )
            # 重新正規化確保加總為 1
            B = heatmap_prob.shape[0]
            heatmap_prob = F.softmax(
                heatmap_prob.flatten(1), dim=-1
            ).reshape_as(heatmap_prob)

        # 加權求和
        pooled = (feat * heatmap_prob).sum(dim=[2, 3])  # (B, C)
        return pooled

    def forward(
        self,
        E_D: torch.Tensor,
        E_M: torch.Tensor,
        heatmap_prob: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            E_D:          (B, C_e, H', W')
            E_M:          (B, C_e, H', W')
            heatmap_prob: (B, 1, H, W) or None
        Returns:
            transl: (B, 3)
            rot:    (B, 4)
            w_x:    nn.Parameter scalar
            w_q:    nn.Parameter scalar
        """
        # 合併 E_D, E_M
        feat = self.feat_merge(torch.cat([E_D, E_M], dim=1))  # (B, C_e, H', W')

        # Heatmap 加權 pooling → 1D 特徵向量
        pooled = self._weighted_pool(feat, heatmap_prob)       # (B, C_e)

        # FC 回歸
        x = self.leaky_relu(self.fc1(pooled))                  # (B, hidden_dim)
        x = self.dropout(x)

        transl = self.leaky_relu(self.fc1_transl(x))
        transl = self.fc2_transl(transl)                        # (B, 3)

        rot = self.leaky_relu(self.fc1_rot(x))
        rot = self.fc2_rot(rot)                                 # (B, 4)
        rot = F.normalize(rot, dim=1)                           # 單位四元數

        return transl, rot, self.w_x, self.w_q
