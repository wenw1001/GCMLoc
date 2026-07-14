"""
itri_common.py — shared paths and helpers for the ITRI-campus preprocessing.

All preprocessing scripts import config + small parsers from here so that the
source layout is defined in exactly one place.

Source dataset (read-only):
    SRC_ROOT/
        pcd_map/vg05_filtered/*.npy          ← global map (5cm voxel centroids)
        submaps_config.json
        itri_campus_images/itri_p4_data_for_ntut/ntut_data/<seq>/
            camera/lucid_cameras_x00.gige_100_f_hdr.h265/<ts_ns>.jpg
            calib/camera/lucid_cameras_x00.gige_100_f_hdr.h265.json
            kitti_format/{poses.txt, times.txt, calib.txt}

Destination (this project):
    DEST_ROOT/
        raw/        ← symlinks to the source data we actually use
        processed/  ← generated outputs (map.h5, undistorted images, poses_torch)
"""
import glob
import json
import os

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
# Override via environment variables, e.g.
#   ITRI_SRC_ROOT=/data/itri_campus ITRI_DEST_ROOT=./iter_campus python preprocess/build_map.py
SRC_ROOT  = os.environ.get('ITRI_SRC_ROOT', './data/itri_campus')
DEST_ROOT = os.environ.get('ITRI_DEST_ROOT', './iter_campus')

NTUT_DIR  = os.path.join(
    SRC_ROOT, 'itri_campus_images', 'itri_p4_data_for_ntut', 'ntut_data'
)

# The front camera is the sync reference (see info.txt: "以 gige_100_f 為主").
FRONT_CAM_REL   = 'camera/lucid_cameras_x00.gige_100_f_hdr.h265'
FRONT_CALIB_REL = 'calib/camera/lucid_cameras_x00.gige_100_f_hdr.h265.json'

# ── Destination sub-paths ────────────────────────────────────────────────
RAW_DIR       = os.path.join(DEST_ROOT, 'raw')
PROC_DIR      = os.path.join(DEST_ROOT, 'processed')
MAP_H5        = os.path.join(PROC_DIR, 'map.h5')


def list_sequences():
    """Return sorted sequence names under ntut_data that have a kitti_format/."""
    seqs = []
    for name in sorted(os.listdir(NTUT_DIR)):
        if os.path.isdir(os.path.join(NTUT_DIR, name, 'kitti_format')):
            seqs.append(name)
    return seqs


def src_seq_dir(seq):
    return os.path.join(NTUT_DIR, seq)


def proc_seq_dir(seq):
    return os.path.join(PROC_DIR, 'sequences', seq)


# ── Parsers ──────────────────────────────────────────────────────────────

def parse_ts_ns(filename):
    """'1718183030195186504.jpg' → 1718183030195186504 (int, nanoseconds)."""
    return int(os.path.splitext(os.path.basename(filename))[0])


def load_front_pinhole(seq):
    """Pinhole intrinsics for the *stored* front-camera images.

    The stored camera/*.jpg are ALREADY rectified to a pinhole model (straight
    edges + a curved gray fill at the bottom from the original fisheye→pinhole
    step). The matching intrinsics are therefore the ROS projection_matrix P,
    NOT the raw fisheye `intrinsic`+`distortion`. Returns (fx, fy, cx, cy, W, H).
    """
    with open(os.path.join(src_seq_dir(seq), FRONT_CALIB_REL)) as f:
        c = json.load(f)
    P = np.array(c['projection_matrix'], dtype=np.float64).reshape(3, 4)
    return (float(P[0, 0]), float(P[1, 1]), float(P[0, 2]), float(P[1, 2]),
            int(c['image_width']), int(c['image_height']))


def front_image_path(seq, ts_ns):
    """Absolute path of the stored front image for a timestamp."""
    return os.path.join(src_seq_dir(seq), FRONT_CAM_REL, f'{ts_ns}.jpg')


def sync_record_path(seq):
    """Path to the official image↔pose sync file (handles the 'sysnc' typo)."""
    for pat in ('sync_record*.yaml', 'sysnc_record*.yaml', 's*sync_record*.yaml'):
        hits = glob.glob(os.path.join(src_seq_dir(seq), pat))
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(f'no sync_record yaml in {src_seq_dir(seq)}')


def load_sync_match(seq):
    """Official front-image → tf pose pairing.

    Returns dict {img_ts_ns(int): tf_relpath(str)} for every front image that
    has a matched tf/map.base_link/*.json. Images with no tf match are omitted.
    The match list also contains the other cameras, so we filter for the tf.
    """
    import yaml
    y = yaml.safe_load(open(sync_record_path(seq)))
    out = {}
    for img_rel, matches in y['match'].items():
        tf = next((m for m in matches
                   if m.startswith('tf/') and m.endswith('.json')), None)
        if tf is None:
            continue
        stem = os.path.splitext(os.path.basename(img_rel))[0]
        if not stem.isdigit():
            continue
        out[int(stem)] = tf
    return out


def load_tf_extrinsic(seq, tf_relpath):
    """Load a tf/map.base_link/*.json → 4x4 extrinsic E = base_link_T_map.

    (Verified: inv(E) == kitti_format/poses.txt pose; E maps map→base_link.)
    """
    with open(os.path.join(src_seq_dir(seq), tf_relpath)) as f:
        E = np.array(json.load(f)['extrinsic'], dtype=np.float64).reshape(4, 4)
    return E


def load_tr_velo_to_cam(seq):
    """Parse Tr_velo_to_cam from kitti_format/calib.txt → 4x4 (cam_T_velo)."""
    calib_path = os.path.join(src_seq_dir(seq), 'kitti_format', 'calib.txt')
    with open(calib_path) as f:
        for line in f:
            if line.startswith('Tr_velo_to_cam'):
                vals = [float(v) for v in line.split(':', 1)[1].split()]
                T = np.eye(4, dtype=np.float64)
                T[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
                return T
    raise RuntimeError(f'Tr_velo_to_cam not found in {calib_path}')


def load_poses(seq):
    """Load kitti_format/poses.txt → (N,4,4) map_T_base_link (base_link→map).

    Verified identical to inv(tf/map.base_link extrinsic); same frame as pcd_map.
    """
    p = os.path.join(src_seq_dir(seq), 'kitti_format', 'poses.txt')
    raw = np.loadtxt(p, dtype=np.float64).reshape(-1, 3, 4)
    out = np.tile(np.eye(4, dtype=np.float64), (raw.shape[0], 1, 1))
    out[:, :3, :4] = raw
    return out


def load_times(seq):
    """Load kitti_format/times.txt → (N,) seconds (float)."""
    p = os.path.join(src_seq_dir(seq), 'kitti_format', 'times.txt')
    return np.loadtxt(p, dtype=np.float64).reshape(-1)


def list_front_images(seq):
    """Return sorted (ts_ns, abspath) for every front-camera jpg in a sequence."""
    d = os.path.join(src_seq_dir(seq), FRONT_CAM_REL)
    out = []
    for fn in os.listdir(d):
        if not fn.endswith('.jpg'):
            continue
        stem = os.path.splitext(fn)[0]
        if not stem.isdigit():
            # a few sequences contain manually-marked files such as
            # '1716262753124782511-------------------500.jpg'; skip them so the
            # stored timestamp always matches '<ts>.jpg' for the loader.
            continue
        out.append((int(stem), os.path.join(d, fn)))
    out.sort(key=lambda x: x[0])
    return out
