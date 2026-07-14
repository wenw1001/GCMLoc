# --- 原版損失類別（供 train_mapping.py / train_loc.py 使用）---
from .legacy import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss

# --- v2 新增損失類別 ---
from .pose_loss import CombinedPoseLoss, HomoscedasticPoseLoss, L1PoseLoss
from .anchor_loss import (
    AnchorLoss,
    FeatureConsistencyLoss,
    FeatureStabilityLoss,
    SaliencyThresholdLoss,
    project_points,
    sample_features,
)
from .entropy_loss import FocusedHeatmapLoss, HeatmapEntropyLoss
from .flow_loss import compute_gt_flow_multiscale, flow_supervision_loss

__all__ = [
    # Legacy (原版)
    "DistancePoints3D",
    "GeometricLoss",
    "L1Loss",
    "ProposedLoss",
    # Pose losses (v2)
    "L1PoseLoss",
    "HomoscedasticPoseLoss",
    "CombinedPoseLoss",
    # Anchor losses (v2)
    "AnchorLoss",
    "FeatureStabilityLoss",
    "SaliencyThresholdLoss",
    "FeatureConsistencyLoss",
    "project_points",
    "sample_features",
    # Entropy losses (v2)
    "HeatmapEntropyLoss",
    "FocusedHeatmapLoss",
    # Flow supervision (v2)
    "compute_gt_flow_multiscale",
    "flow_supervision_loss",
]
