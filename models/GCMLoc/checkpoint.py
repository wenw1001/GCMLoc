"""
Checkpoint helpers: rebuild a GCMLoc network from a saved checkpoint.

The embedded `config` is the source of truth for the architecture (as in standard
model repos); it is cross-validated against the state_dict, so a wrong config fails
with a clear error instead of building the wrong network. When a checkpoint carries
no architecture information at all, the architecture is inferred from the weights
(with a warning). Legacy key spellings from the ablation-era training scripts
(rgb_backbone='dinov2'/'dinov2b'/'dinov2s' + dinov2_variant) are understood.
"""

import torch

from models.GCMLoc.GCMLoc_mapping import GCMLocMapping
from models.GCMLoc.GCMLoc_localization import GCMLocLocalization

_DINO_DIM_TO_VARIANT = {768: 'b', 384: 's'}


def arch_from_config(cfg):
    """Canonical architecture dict read from an embedded checkpoint config.

    Returns None when the config carries no architecture information.
    Handles current configs (dinov2_variant) and legacy ones (rgb_backbone
    'dinov2b' / 'dinov2s' / 'dinov2' + dinov2_variant).
    """
    if not cfg:
        return None
    rb = cfg.get('rgb_backbone')
    if rb in ('dinov2b', 'dinov2s'):
        variant = rb[-1]
    elif rb == 'dinov2':
        variant = cfg.get('dinov2_variant') or 'b'
    elif rb is None:
        variant = cfg.get('dinov2_variant')
    else:                       # e.g. 'cnn' — an ablation architecture (or a wrong config)
        variant = f'<{rb}>'
    if variant is None:
        return None
    arch = {'dinov2_variant': variant, 'feat_dim': int(cfg.get('feat_dim') or 128)}
    if cfg.get('heatmap_dim_s1'):
        arch['heatmap_dim_s1'] = int(cfg['heatmap_dim_s1'])
    if cfg.get('heatmap_dim_s2'):
        arch['heatmap_dim_s2'] = int(cfg['heatmap_dim_s2'])
    return arch


def arch_from_state_dict(state_dict):
    """Architecture dict inferred from the weights themselves (shape-based)."""
    w = state_dict['rgb_backbone.projection.0.weight']          # (feat_dim, dino_dim, 1, 1)
    arch = {'dinov2_variant': _DINO_DIM_TO_VARIANT[w.shape[1]], 'feat_dim': int(w.shape[0])}
    if 'pose_reg_s1.fc1.weight' in state_dict:
        arch['heatmap_dim_s1'] = int(state_dict['pose_reg_s1.fc1.weight'].shape[1])
    if 'pose_reg_s2.fc1.weight' in state_dict:
        arch['heatmap_dim_s2'] = int(state_dict['pose_reg_s2.fc1.weight'].shape[1])
    return arch


def resolve_arch(ckpt, path=''):
    """Architecture of a checkpoint: config first, cross-validated against the weights.

    Raises ValueError when the embedded config contradicts the weights; falls back to
    weight inference (with a warning) when the config has no architecture keys.
    """
    arch_sd = arch_from_state_dict(ckpt['state_dict'])
    arch_cfg = arch_from_config(ckpt.get('config'))
    if arch_cfg is None:
        print(f"[checkpoint] {path or 'checkpoint'}: no architecture in the embedded config — "
              f"inferred from the weights: {arch_sd}")
        return arch_sd
    mismatch = {k: (arch_cfg[k], arch_sd[k])
                for k in arch_sd if k in arch_cfg and arch_cfg[k] != arch_sd[k]}
    if mismatch:
        detail = ', '.join(f"{k}: config={c!r} vs weights={s!r}" for k, (c, s) in mismatch.items())
        raise ValueError(
            f"Checkpoint {path or ''} embedded config does not match its weights ({detail}). "
            f"The config is wrong (e.g. it recorded CLI defaults of an older training script) — "
            f"fix the checkpoint's config['...'] entries to match the weights.")
    return arch_sd


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def build_mapping_from_checkpoint(ckpt, image_size, topk_points=None):
    """GCMLocMapping with the checkpoint's architecture and weights (strict)."""
    arch = resolve_arch(ckpt)
    cfg = ckpt.get('config', {}) or {}
    model = GCMLocMapping(image_size=image_size,
                          topk_points=topk_points or cfg.get('topk_points') or 5000, **arch)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    return model, arch


def build_localization_from_checkpoint(ckpt, image_size):
    """GCMLocLocalization initialised from a *mapping* or a *localization* checkpoint.

    Returns (model, arch, missing_keys): from a mapping checkpoint the Stage-2 heatmap
    decoder (heatmap_head_s2) is not available and stays randomly initialised — this is
    the intended start of localization training. From a localization checkpoint all keys load.
    """
    arch = resolve_arch(ckpt)
    sd = ckpt['state_dict']
    model = GCMLocLocalization(image_size=image_size, dinov2_variant=arch['dinov2_variant'],
                               feat_dim=arch['feat_dim'], heatmap_dim_s2=arch['heatmap_dim_s2'])
    model_keys = model.state_dict()
    # drop Stage-1 keys and shape-mismatching keys (mapping ckpt: heatmap_head_s2 is HeatmapHead)
    usable = {k: v for k, v in sd.items() if k in model_keys and model_keys[k].shape == v.shape}
    missing = [k for k in model_keys if k not in usable]
    model.load_state_dict(usable, strict=False)
    return model, arch, missing
