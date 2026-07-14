"""
Entropy Regularization Loss for GCMLoc (M5) — L_entropy

架構規劃書 §6.4：熱力圖熵正則化

$$L_entropy = -(1/HW) Σ_{h,w} H_hat(h,w) · log(H_hat(h,w) + ε)$$

其中 H_hat 為 softmax 正規化後的 heatmap。

目的：鼓勵 Stage 1 heatmap 集中在少數高信心區域（低熵），
避免機率均勻分散（高熵）於整個地圖。

輸入：
    heatmap_logit: (B, 1, H, W)  — HeatmapHead 輸出的未正規化 logit
    或
    heatmap_prob: (B, 1, H, W)  — 已用 softmax 正規化的 heatmap（需加總為 1）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapEntropyLoss(nn.Module):
    """
    Heatmap 負熵損失（鼓勵集中分佈）。

    L_entropy = -(1/B) Σ_b [ Σ_{h,w} H_hat_b(h,w) · log(H_hat_b(h,w) + ε) ]

    給定負熵（即信息熵的負值），最小化此損失等同於最小化熵，
    使 heatmap 更加集中。

    Args:
        eps (float): 數值穩定性項，防止 log(0)，預設 1e-8。
        from_logit (bool):
            True  → 輸入為 logit，內部做 spatial softmax 正規化
            False → 輸入為已正規化機率 heatmap（加總需為 1）
    """

    def __init__(self, eps: float = 1e-8, from_logit: bool = True):
        super().__init__()
        self.eps = eps
        self.from_logit = from_logit

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heatmap: (B, 1, H, W)  — logit 或已正規化機率
        Returns:
            loss: scalar （已取負值，最小化 = 降低熵 = 集中分佈）
        """
        B, _, H, W = heatmap.shape

        if self.from_logit:
            # 空間 Softmax 正規化
            h_flat = heatmap.reshape(B, -1)               # (B, H*W)
            prob = F.softmax(h_flat, dim=-1)              # (B, H*W)，加總=1
        else:
            prob = heatmap.reshape(B, -1).clamp(min=self.eps)
            # 確保正規化
            prob = prob / prob.sum(dim=-1, keepdim=True)

        # 計算熵：H = -Σ p·log(p)
        entropy = -(prob * torch.log(prob + self.eps)).sum(dim=-1)  # (B,)

        # 損失 = 平均熵（最小化熵 → 鼓勵集中）
        loss = entropy.mean()
        return loss


class FocusedHeatmapLoss(nn.Module):
    """
    擴展版熵正則化，加入 GT 峰值位置的集中性額外懲罰。

    若已知 GT 位置（如訓練時可得），可在 GT 峰值附近額外鼓勵高機率值，
    搭配熵損失形成更明確的監督訊號。

    Args:
        entropy_weight (float): 熵損失的權重。
        focal_weight (float): GT 集中性損失的權重。
        eps (float): 數值穩定性。
    """

    def __init__(
        self,
        entropy_weight: float = 1.0,
        focal_weight: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.entropy_loss = HeatmapEntropyLoss(eps=eps, from_logit=True)
        self.entropy_weight = entropy_weight
        self.focal_weight = focal_weight

    def forward(
        self,
        heatmap_logit: torch.Tensor,
        gt_coords: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            heatmap_logit: (B, 1, H, W)  — 未正規化 logit
            gt_coords:     (B, 2)  — GT 峰值座標 (x, y)，像素單位，可為 None
        Returns:
            loss: scalar
        """
        loss = self.entropy_weight * self.entropy_loss(heatmap_logit)

        if gt_coords is not None and self.focal_weight > 0.0:
            B, _, H, W = heatmap_logit.shape
            # 計算 GT 位置的 softmax 機率值（應盡量高）
            h_flat = heatmap_logit.reshape(B, -1)
            prob = F.softmax(h_flat, dim=-1).reshape(B, 1, H, W)

            # 將 gt_coords 轉為 grid_sample 格式 normalized [-1, 1]
            gx = (gt_coords[:, 0] / (W - 1)) * 2.0 - 1.0  # (B,)
            gy = (gt_coords[:, 1] / (H - 1)) * 2.0 - 1.0  # (B,)
            grid = torch.stack([gx, gy], dim=1).reshape(B, 1, 1, 2)  # (B, 1, 1, 2)

            # 取 GT 位置的機率值
            gt_prob = F.grid_sample(
                prob, grid, mode="bilinear", align_corners=True
            ).reshape(B)  # (B,)

            # 最大化 GT 位置機率（等效最小化負 log 機率）
            focal = -torch.log(gt_prob + 1e-8).mean()
            loss = loss + self.focal_weight * focal

        return loss
