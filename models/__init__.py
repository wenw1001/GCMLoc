"""
GCMLoc network package.

    models.backbone   DINOv2Backbone (RGB), DepthCNN (three-branch depth encoder)
    models.fusion     CrossModalFusion, GCMFlow (multi-scale cross-modal flow matcher)
    models.heads      HeatmapDecoder, HeatmapHead, PoseRegressor
    models.GCMLoc     GCMLocMapping (offline, Stage 1 + 2), GCMLocLocalization (online, Stage 2)
                      + checkpoint helpers (arch_from_state_dict, build_*_from_checkpoint)
"""
