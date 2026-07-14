"""
Anchor Loss for GCMLoc (M5) — L_anchor

架構規劃書 §6.3：DINOv2 錨定損失（核心新增）

支援三種方案（可組合使用）：
    A: 特徵穩定度 MSE（Feature Stability MSE）
       L_anchor_A = (1/N) Σ || φ(u_i^pred) - φ(u_i^gt) ||₂²
    B: DINOv2 顯著性加權損失（Saliency Threshold Penalty）
       L_anchor_B = (1/N) Σ max(0, τ_s - s_i)
    C: 特徵一致性正則化（Cosine Dissimilarity，離線/線上 DINOv2 特徵對比）
       L_anchor_C = (1/N) Σ (1 - cosine_similarity(φ_off(u^gt_i), φ_on(u^pred_i)))

所有方案均依賴：
    - 3D 關鍵點 P = {p_1, ..., p_N}
    - 預測位姿 T_pred / 真值位姿 T_gt
    - DINOv2 特徵圖 F_I（由 DINOv2Wrapper 計算，需先對齊至目標解析度）
    - 相機投影函數 π（由外部傳入，以相容不同相機模型）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 輔助函數
# ---------------------------------------------------------------------------

def project_points(
    pts_3d: torch.Tensor,
    T: torch.Tensor,
    K: torch.Tensor,
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    """
    將 3D 點用位姿 T 與相機內參 K 投影至歸一化 2D 座標（grid_sample 格式）。

    Args:
        pts_3d: (B, N, 3)  — 相機座標系下的 3D 點（或世界座標系下；T 負責轉換）
        T:      (B, 4, 4)  — 位姿矩陣（rigid transform，3D → camera）
        K:      (B, 3, 3)  — 相機內參矩陣
        img_h, img_w: 特徵圖高寬（用於正規化至 [-1, 1]）
    Returns:
        coords_norm: (B, N, 2)  — range [-1, 1]，格式為 (x_norm, y_norm)，
                     可直接傳入 F.grid_sample 的 grid 參數
    """
    B, N, _ = pts_3d.shape

    # 齊次座標
    ones = torch.ones(B, N, 1, device=pts_3d.device, dtype=pts_3d.dtype)
    pts_h = torch.cat([pts_3d, ones], dim=-1)  # (B, N, 4)

    # 位姿轉換：3D → camera frame
    pts_cam = torch.bmm(pts_h, T.transpose(1, 2))[:, :, :3]  # (B, N, 3)

    # 透視投影
    z = pts_cam[:, :, 2:3].clamp(min=1e-6)
    pts_2d = torch.bmm(pts_cam / z, K.transpose(1, 2))[:, :, :2]  # (B, N, 2)

    # 歸一化至 [-1, 1]（grid_sample 格式）
    px = pts_2d[:, :, 0]  # pixel x
    py = pts_2d[:, :, 1]  # pixel y
    x_norm = (px / (img_w - 1)) * 2.0 - 1.0
    y_norm = (py / (img_h - 1)) * 2.0 - 1.0

    coords_norm = torch.stack([x_norm, y_norm], dim=-1)  # (B, N, 2)
    return coords_norm


def sample_features(
    feat_map: torch.Tensor,
    coords_norm: torch.Tensor,
) -> torch.Tensor:
    """
    從特徵圖以雙線性插值取出每個點的特徵向量。

    Args:
        feat_map:   (B, C, H, W)
        coords_norm:(B, N, 2)  — (x_norm, y_norm) in [-1, 1]
    Returns:
        feats: (B, N, C)
    """
    # grid_sample 需要 grid: (B, N, 1, 2) 或 (B, 1, N, 2)
    grid = coords_norm.unsqueeze(2)          # (B, N, 1, 2)
    # reshape feat_map grid 格式
    sampled = F.grid_sample(
        feat_map, grid,
        mode="bilinear", padding_mode="border", align_corners=True
    )  # (B, C, N, 1)
    sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, C)
    return sampled


# ---------------------------------------------------------------------------
# 方案 A：特徵穩定度 MSE
# ---------------------------------------------------------------------------

class FeatureStabilityLoss(nn.Module):
    """
    L_anchor_A = (1/N) Σ || φ(u_i^pred) - φ(u_i^gt) ||₂²

    直覺：好的 3D 關鍵點落在穩定結構上，即使投影位置有微小偏移，
    DINOv2 特徵仍應保持一致；動態物件的特徵會因遮擋或移動而劇烈變化。
    """

    def forward(
        self,
        feat_map: torch.Tensor,
        coords_pred: torch.Tensor,
        coords_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat_map:    (B, C, H, W)  — DINOv2 特徵圖（已對齊至目標解析度）
            coords_pred: (B, N, 2)     — 預測位姿投影座標（normalized）
            coords_gt:   (B, N, 2)     — 真值位姿投影座標（normalized）
        Returns:
            loss: scalar
        """
        phi_pred = sample_features(feat_map, coords_pred)   # (B, N, C)
        phi_gt = sample_features(feat_map, coords_gt)       # (B, N, C)
        loss = ((phi_pred - phi_gt) ** 2).sum(dim=-1).mean()
        return loss


# ---------------------------------------------------------------------------
# 方案 B：DINOv2 顯著性加權門檻損失
# ---------------------------------------------------------------------------

class SaliencyThresholdLoss(nn.Module):
    """
    L_anchor_B = (1/N) Σ max(0, τ_s - s_i)

    使用 DINOv2 CLS token 注意力圖作為顯著性分數，
    懲罰選取到低顯著性（可能為動態或無紋理）區域的點。

    Args:
        saliency_threshold (float): 穩定度門檻 τ_s，預設 0.5。
    """

    def __init__(self, saliency_threshold: float = 0.5):
        super().__init__()
        self.tau_s = saliency_threshold

    def forward(
        self,
        saliency_map: torch.Tensor,
        coords_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            saliency_map: (B, 1, H_p, W_p)  — 正規化後的顯著性圖 S_norm ∈ [0,1]
                          （由 DINOv2 最後一層 CLS→patch 注意力計算）
            coords_gt:    (B, N, 2)          — 真值投影座標（normalized）
        Returns:
            loss: scalar
        """
        # 提取每個點的顯著性分數
        s_i = sample_features(saliency_map, coords_gt)  # (B, N, 1)
        s_i = s_i.squeeze(-1)                            # (B, N)

        # 懲罰低顯著性點
        penalty = F.relu(self.tau_s - s_i)               # max(0, τ_s - s_i)
        loss = penalty.mean()
        return loss


# ---------------------------------------------------------------------------
# 方案 C：特徵一致性正則化（離線/線上 Cosine Dissimilarity）
# ---------------------------------------------------------------------------

class FeatureConsistencyLoss(nn.Module):
    """
    L_anchor_C = (1/N) Σ (1 - cosine_similarity(φ_off(u^gt_i), φ_on(u^pred_i)))

    直覺：好的 3D 點在離線與線上影像中對應的語意特徵應高度一致（cosine → 1）；
    動態物件對應的特徵則不一致（cosine → 0 甚至負值）。
    """

    def forward(
        self,
        feat_map_off: torch.Tensor,
        feat_map_on: torch.Tensor,
        coords_gt: torch.Tensor,
        coords_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat_map_off: (B, C, H, W)  — 離線影像 DINOv2 特徵圖
            feat_map_on:  (B, C, H, W)  — 線上影像 DINOv2 特徵圖
            coords_gt:    (B, N, 2)     — 真值投影座標（用於離線特徵）
            coords_pred:  (B, N, 2)     — 預測投影座標（用於線上特徵）
        Returns:
            loss: scalar
        """
        phi_off = sample_features(feat_map_off, coords_gt)    # (B, N, C)
        phi_on = sample_features(feat_map_on, coords_pred)    # (B, N, C)

        # Cosine Similarity per point
        cos_sim = F.cosine_similarity(phi_off, phi_on, dim=-1)  # (B, N)
        loss = (1.0 - cos_sim).mean()
        return loss


# ---------------------------------------------------------------------------
# 聯合錨定損失入口
# ---------------------------------------------------------------------------

class AnchorLoss(nn.Module):
    """
    組合錨定損失（可選啟用方案 A/B/C）。

    Args:
        lambda_a (float): 方案 A 權重（0.0 表示停用）。
        lambda_b (float): 方案 B 權重。
        lambda_c (float): 方案 C 權重。
        saliency_threshold (float): 方案 B 的 τ_s。
    """

    def __init__(
        self,
        lambda_a: float = 0.1,
        lambda_b: float = 0.0,
        lambda_c: float = 0.05,
        saliency_threshold: float = 0.5,
    ):
        super().__init__()
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.lambda_c = lambda_c

        if lambda_a > 0.0:
            self.loss_a = FeatureStabilityLoss()
        if lambda_b > 0.0:
            self.loss_b = SaliencyThresholdLoss(saliency_threshold)
        if lambda_c > 0.0:
            self.loss_c = FeatureConsistencyLoss()

    def forward(
        self,
        feat_map: torch.Tensor = None,
        feat_map_off: torch.Tensor = None,
        feat_map_on: torch.Tensor = None,
        saliency_map: torch.Tensor = None,
        coords_pred: torch.Tensor = None,
        coords_gt: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            feat_map:     (B, C, H, W)  — 共用特徵圖（方案 A）
            feat_map_off: (B, C, H, W)  — 離線特徵圖（方案 C）
            feat_map_on:  (B, C, H, W)  — 線上特徵圖（方案 C）
            saliency_map: (B, 1, H, W)  — 顯著性圖（方案 B）
            coords_pred:  (B, N, 2)     — 預測投影座標（normalized）
            coords_gt:    (B, N, 2)     — 真值投影座標（normalized）
        Returns:
            dict with keys: 'L_anchor_A', 'L_anchor_B', 'L_anchor_C', 'total'
        """
        device = (
            feat_map.device if feat_map is not None
            else feat_map_off.device if feat_map_off is not None
            else coords_pred.device
        )
        total = torch.tensor(0.0, device=device)
        out = {}

        if self.lambda_a > 0.0 and feat_map is not None:
            la = self.loss_a(feat_map, coords_pred, coords_gt)
            out["L_anchor_A"] = la
            total = total + self.lambda_a * la

        if self.lambda_b > 0.0 and saliency_map is not None:
            lb = self.loss_b(saliency_map, coords_gt)
            out["L_anchor_B"] = lb
            total = total + self.lambda_b * lb

        if self.lambda_c > 0.0 and feat_map_off is not None and feat_map_on is not None:
            lc = self.loss_c(feat_map_off, feat_map_on, coords_gt, coords_pred)
            out["L_anchor_C"] = lc
            total = total + self.lambda_c * lc

        out["total"] = total
        return out
