"""
Flow Supervision Loss for GCMLoc.

Computes ground truth per-pixel flow from known pose perturbation,
then supervises the MultiScaleFlowMatcher's predictions directly.

Principle:
    We know the GT perturbation (R, t) for each training sample.
    For each visible 3D point P:
        - It projects to (u_init, v_init) in the perturbed depth D_init
        - It projects to (u_gt, v_gt) in the GT depth D_gt
        - GT flow at (u_init, v_init) = (u_gt - u_init, v_gt - v_init)
    This gives the Flow Matcher an explicit target at every pixel,
    instead of relying on indirect gradients through pose regression.
"""

import torch
import torch.nn.functional as F


def compute_gt_flow_multiscale(
    uv: torch.Tensor,
    uvt: torch.Tensor,
    indexes: torch.Tensor,
    H: int,
    W: int,
    scales: tuple = (4, 8, 16),
) -> tuple:
    """
    Compute GT flow fields at multiple scales from sparse correspondences.

    Args:
        uv:      (N, 2) int — D_init pixel coords [col, row]
        uvt:     (N, 2) int — D_gt pixel coords [col, row]
        indexes: (N,) bool  — visibility mask
        H, W:    full image resolution
        scales:  downscale factors for each level (finest to coarsest)

    Returns:
        gt_flows: list of (2, H//s, W//s) tensors [scale1, scale2, scale3]
        gt_masks: list of (1, H//s, W//s) tensors (valid pixel indicators)
    """
    device = uv.device
    valid_uv = uv[indexes].float()    # (M, 2) — D_init positions
    valid_uvt = uvt[indexes].float()   # (M, 2) — D_gt positions

    # Full-resolution pixel displacement
    dx = valid_uvt[:, 0] - valid_uv[:, 0]  # column shift
    dy = valid_uvt[:, 1] - valid_uv[:, 1]  # row shift

    gt_flows = []
    gt_masks = []

    for s in scales:
        h, w = H // s, W // s

        # Scale coordinates and flow to this resolution
        x_s = (valid_uv[:, 0] / s).long().clamp(0, w - 1)
        y_s = (valid_uv[:, 1] / s).long().clamp(0, h - 1)
        dx_s = dx / s
        dy_s = dy / s

        # Vectorized scatter (sum then average)
        idx = y_s * w + x_s  # flat index

        flow_flat = torch.zeros(2, h * w, device=device)
        flow_flat[0].scatter_add_(0, idx, dx_s)
        flow_flat[1].scatter_add_(0, idx, dy_s)

        count_flat = torch.zeros(h * w, device=device)
        count_flat.scatter_add_(0, idx, torch.ones_like(dx_s))

        valid = count_flat > 0
        flow_flat[0, valid] /= count_flat[valid]
        flow_flat[1, valid] /= count_flat[valid]

        gt_flows.append(flow_flat.reshape(2, h, w))
        gt_masks.append(valid.reshape(1, h, w).float())

    return gt_flows, gt_masks


def flow_supervision_loss(
    pred_flows: list,
    gt_flows: list,
    gt_masks: list,
) -> torch.Tensor:
    """
    Smooth L1 loss between predicted and GT flows at valid pixels only.

    Args:
        pred_flows: [flow_3(B,2,H/16,W/16), flow_2(B,2,H/8,W/8), flow_1(B,2,H/4,W/4)]
        gt_flows:   list of (B, 2, H//s, W//s) — batched GT flows
        gt_masks:   list of (B, 1, H//s, W//s) — valid pixel masks

    Returns:
        loss: scalar
    """
    total = torch.tensor(0.0, device=pred_flows[0].device)
    n_scales = 0

    # pred_flows order: [flow_3, flow_2, flow_1] (coarse to fine)
    # gt order:         [scale1(fine), scale2(mid), scale3(coarse)]
    # Match them: pred_flows[0]=flow_3 ↔ gt[2]=scale3, etc.
    for i, (pred, gt, mask) in enumerate(zip(
        reversed(pred_flows), gt_flows, gt_masks
    )):
        if mask.sum() < 1:
            continue

        # Align spatial resolution if needed
        if pred.shape[2:] != gt.shape[2:]:
            gt = F.interpolate(
                gt.unsqueeze(0) if gt.dim() == 3 else gt,
                size=pred.shape[2:], mode='bilinear', align_corners=False,
            )
            mask = F.interpolate(
                mask.unsqueeze(0) if mask.dim() == 3 else mask,
                size=pred.shape[2:], mode='nearest',
            )

        diff = F.smooth_l1_loss(pred * mask, gt * mask, reduction='sum')
        total = total + diff / mask.sum().clamp(min=1)
        n_scales += 1

    return total / max(n_scales, 1)
