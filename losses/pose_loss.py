"""
Pose Loss for GCMLoc — L_p0 (Stage 1) and L_p1 (Stage 2)

架構規劃書 §6.2：原版位姿損失（保留）

$$L_p0 = ||t_0 - t_gt||_1 + β · ||q_0 - q_gt||_1$$
$$L_p1 = ||t_1 - t_gt||_1 + β · ||q_1 - q_gt||_1$$

其中 t 為平移向量，q 為旋轉四元數，β 為平衡係數。

提供兩種變體：
    1. L1PoseLoss: 純 L1/SmoothL1 損失（對應原版 ProposedLoss / L1Loss）
    2. HomoscedasticPoseLoss: 可學習不確定性加權（對應原版 GeometricLoss）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from quaternion_distances import quaternion_distance


class L1PoseLoss(nn.Module):
    """
    純 L1（SmoothL1）位姿損失（方案 A，對應原版 ProposedLoss/L1Loss）。

    L = rescale_trans * SmoothL1(t_pred, t_gt) + rescale_rot * SmoothL1(q_pred, q_gt)

    Args:
        rescale_trans (float): 平移損失權重（β_t）。
        rescale_rot (float): 旋轉損失權重（β_r）。
        use_quat_distance (bool):
            True  → 旋轉損失使用 quaternion_distance（與原版一致）
            False → 旋轉損失使用 SmoothL1（簡化版）
    """

    def __init__(
        self,
        rescale_trans: float = 1.0,
        rescale_rot: float = 1.0,
        use_quat_distance: bool = True,
    ):
        super().__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.use_quat_distance = use_quat_distance
        self.transl_loss_fn = nn.SmoothL1Loss(reduction="none")

    def forward(
        self,
        transl_pred: torch.Tensor,
        rot_pred: torch.Tensor,
        transl_gt: torch.Tensor,
        rot_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            transl_pred: (B, 3)
            rot_pred:    (B, 4)  — 已正規化四元數
            transl_gt:   (B, 3)
            rot_gt:      (B, 4)
        Returns:
            total_loss: scalar
        """
        loss = torch.tensor(0.0, device=transl_pred.device)

        if self.rescale_trans != 0.0:
            loss_t = self.transl_loss_fn(transl_pred, transl_gt).sum(dim=1).mean()
            loss = loss + self.rescale_trans * loss_t

        if self.rescale_rot != 0.0:
            if self.use_quat_distance:
                loss_r = quaternion_distance(rot_pred, rot_gt, rot_pred.device).mean()
            else:
                loss_r = self.transl_loss_fn(rot_pred, rot_gt).sum(dim=1).mean()
            loss = loss + self.rescale_rot * loss_r

        return loss


class HomoscedasticPoseLoss(nn.Module):
    """
    可學習不確定性加權位姿損失（方案 B，對應原版 GeometricLoss）。

    L = exp(-s_x) * L_transl + s_x + exp(-s_q) * L_rot + s_q
    其中 s_x, s_q 為可學習的 log 不確定性（homoscedastic uncertainty）。

    直覺：網路自動學習平移與旋轉損失的相對重要性，
    避免手動調整 rescale_trans/rescale_rot 的困難。
    """

    def __init__(self):
        super().__init__()
        self.s_x = nn.Parameter(torch.tensor([0.0]))    # log 平移不確定性
        self.s_q = nn.Parameter(torch.tensor([-3.0]))   # log 旋轉不確定性
        self.transl_loss_fn = nn.SmoothL1Loss(reduction="none")

    def forward(
        self,
        transl_pred: torch.Tensor,
        rot_pred: torch.Tensor,
        transl_gt: torch.Tensor,
        rot_gt: torch.Tensor,
    ) -> torch.Tensor:
        loss_t = self.transl_loss_fn(transl_pred, transl_gt).sum(dim=1).mean()
        loss_r = quaternion_distance(rot_pred, rot_gt, rot_pred.device).mean()

        total = (
            torch.exp(-self.s_x) * loss_t + self.s_x
            + torch.exp(-self.s_q) * loss_r + self.s_q
        )
        return total


class CombinedPoseLoss(nn.Module):
    """
    聯合 Stage 1 + Stage 2 位姿損失（L_p0 + L_p1）。

    Args:
        mode (str): 'l1'（L1PoseLoss）或 'homoscedastic'（HomoscedasticPoseLoss）。
        rescale_trans (float): 僅 mode='l1' 時有效。
        rescale_rot (float): 僅 mode='l1' 時有效。
        weight_stage0 (float): L_p0 的整體權重。
        weight_stage1 (float): L_p1 的整體權重。
    """

    def __init__(
        self,
        mode: str = "l1",
        rescale_trans: float = 1.0,
        rescale_rot: float = 1.0,
        weight_stage0: float = 1.0,
        weight_stage1: float = 1.0,
    ):
        super().__init__()
        assert mode in ("l1", "homoscedastic"), f"mode 必須為 'l1' 或 'homoscedastic'，收到：{mode}"
        self.weight_stage0 = weight_stage0
        self.weight_stage1 = weight_stage1

        if mode == "l1":
            self.loss_fn0 = L1PoseLoss(rescale_trans, rescale_rot)
            self.loss_fn1 = L1PoseLoss(rescale_trans, rescale_rot)
        else:
            self.loss_fn0 = HomoscedasticPoseLoss()
            self.loss_fn1 = HomoscedasticPoseLoss()

    def forward(
        self,
        transl0_pred: torch.Tensor,
        rot0_pred: torch.Tensor,
        transl1_pred: torch.Tensor,
        rot1_pred: torch.Tensor,
        transl_gt: torch.Tensor,
        rot_gt: torch.Tensor,
    ) -> dict:
        """
        Args:
            transl0_pred: (B, 3)  — Stage 1 平移預測
            rot0_pred:    (B, 4)  — Stage 1 旋轉預測
            transl1_pred: (B, 3)  — Stage 2 平移預測
            rot1_pred:    (B, 4)  — Stage 2 旋轉預測
            transl_gt:    (B, 3)  — Ground truth 平移
            rot_gt:       (B, 4)  — Ground truth 旋轉
        Returns:
            dict with keys: 'L_p0', 'L_p1', 'total'
        """
        L_p0 = self.loss_fn0(transl0_pred, rot0_pred, transl_gt, rot_gt)
        L_p1 = self.loss_fn1(transl1_pred, rot1_pred, transl_gt, rot_gt)
        total = self.weight_stage0 * L_p0 + self.weight_stage1 * L_p1
        return {"L_p0": L_p0, "L_p1": L_p1, "total": total}
