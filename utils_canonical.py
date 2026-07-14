"""
utils_canonical.py — Canonical-camera normalization for GCMLoc.

Removes camera-hardware dependency by warping every dataset's RGB image to a
single virtual pinhole camera (the active preset) and overriding the per-sample
calib to that camera's intrinsics. The point cloud stays metric in the camera
frame, so projecting it with the canonical calib gives a depth image that is
geometrically consistent across cameras. Model / training loop are unchanged —
only the dataset layer (RGB warp + calib override) and img_shape.

Two presets (switch with set_preset / the `canon_preset` config key):
    'A'  768x384  fx=fy=480  cx=384 cy=192   (HFOV 77.3°, VFOV 43.6°, 2:1)
         = the GCMLoc paper canonical (x_n∈[-0.8,0.8], y_n∈[-0.4,0.4]).
         Least black border for KITTI/Argo; crops wide cams (ITRI/Mio) more.
    'B'  960x480  fx=fy=402.77 cx=480 cy=240  (HFOV 100°, VFOV 61.9°, 2:1)
         Keeps wide-camera (ITRI/Mio) side FOV; a bit more black on KITTI/Argo.

The warp is a pure affine remap (pinhole→pinhole, shared optical axis):
    u_s = (fx_s / CANON_FX) * (u_c - CANON_CX) + cx_s
    v_s = (fy_s / CANON_FY) * (v_c - CANON_CY) + cy_s
precomputed once per source intrinsic, applied per frame with cv2.remap.

NOTE: this is pinhole-only (no lens distortion). Strongly distorted cameras
(e.g. Mio dashcam) must be UNDISTORTED first (cv2.undistort with a calibrated
K+D) before this canonical warp.
"""
import cv2
import numpy as np
import torch
from PIL import Image

PRESETS = {
    'A': dict(W=768, H=384, fx=480.0,  fy=480.0,  cx=384.0, cy=192.0),
    'B': dict(W=960, H=480, fx=402.77, fy=402.77, cx=480.0, cy=240.0),
}

# Active canonical parameters (module globals; set by set_preset()).
CANON_PRESET = None
CANON_W = CANON_H = 0
CANON_FX = CANON_FY = CANON_CX = CANON_CY = 0.0
CANON_SHAPE = (0, 0)


def set_preset(name):
    """Select the active canonical camera ('A' or 'B'). Call before building
    datasets / reading CANON_SHAPE."""
    global CANON_PRESET, CANON_W, CANON_H, CANON_FX, CANON_FY, CANON_CX, CANON_CY, CANON_SHAPE
    if name not in PRESETS:
        raise ValueError(f"unknown canon_preset {name!r}; choose from {list(PRESETS)}")
    p = PRESETS[name]
    CANON_PRESET = name
    CANON_W, CANON_H = p['W'], p['H']
    CANON_FX, CANON_FY = p['fx'], p['fy']
    CANON_CX, CANON_CY = p['cx'], p['cy']
    CANON_SHAPE = (CANON_H, CANON_W)   # (H, W) for img_shape


set_preset('A')   # default = GCMLoc paper canonical


def build_remap_maps(src_fx, src_fy, src_cx, src_cy):
    """Precompute cv2.remap lookup tables (float32, [CANON_H, CANON_W]) for the
    active preset. For each canonical pixel, gives the source pixel to sample."""
    u_c = np.arange(CANON_W, dtype=np.float32)
    v_c = np.arange(CANON_H, dtype=np.float32)
    uu, vv = np.meshgrid(u_c, v_c)
    map_x = (src_fx / CANON_FX) * (uu - CANON_CX) + src_cx
    map_y = (src_fy / CANON_FY) * (vv - CANON_CY) + src_cy
    return map_x.astype(np.float32), map_y.astype(np.float32)


def apply_canonical_warp(pil_img, map_x, map_y):
    """Warp a PIL RGB image to the canonical camera (out-of-range → black)."""
    arr = np.asarray(pil_img)
    warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return Image.fromarray(warped)


def get_canonical_calib():
    """Active canonical [fx, fy, cx, cy] tensor (fresh each call)."""
    return torch.tensor([CANON_FX, CANON_FY, CANON_CX, CANON_CY],
                        dtype=torch.float32)
