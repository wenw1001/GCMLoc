# -------------------------------------------------------------------
# Dataset_argoverse_localization.py  (CMRNext-format maps, GCMLoc)
#
# Argoverse Tracking v1.1 dataset loader for the ONLINE LOCALIZATION stage.
# Drop-in replacement for DatasetVisibilityKittiSingle (KITTI localization).
#
# Data structure expected at dataset_dir:
#   <dataset_dir>/
#     train1/<log_id>/
#       map.h5                               ← global point cloud (city-relative frame)
#       vehicle_calibration_info.json        ← camera intrinsics
#       ring_front_center/
#         ring_front_center_<ts>.jpg
#         poses_torch/
#           ring_front_center_<ts>.npy       ← 4×4 cam_T_city_relative pose
#     train2/... train3/... train4/...
#
# Point cloud loading strategy:
#   Primary (when maps_folder is given and per-frame .npy files exist):
#       <seq_dir>/<maps_folder>/<ts>.npy    ← pre-built GCMLoc from offline model
#       Shape (M,3) or (3,M) or (4,M), vehicle frame.
#   Fallback (when no per-frame .npy available — default for fresh Argoverse data):
#       Crop from global map.h5 using poses_torch/*.npy (same as mapping dataset).
#
# Split assignment:
#   split='train' → train1, train2, train3  (60 logs, ~36k frames)
#   split='test'  → train4                  ( 5 logs, ~2.7k frames)
#
# Sample dict keys (identical to KITTI localization version):
#   {rgb, point_cloud, calib, tr_error, rot_error, idx, rgb_name}
# -------------------------------------------------------------------

import csv
import json
import os
from math import radians

import h5py
import mathutils
import numpy as np
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from camera_model_localization import CameraModel
import utils
import utils_canonical as UC

# ─────────────────────────────────────────────────────────────────
# Split configuration
# ─────────────────────────────────────────────────────────────────

_TRAIN_SPLITS = ['train1', 'train2', 'train3']
_TEST_SPLITS  = ['train4']
_VAL_SPLITS   = ['val']

# Crop window in camera frame (same as mapping dataset)
_X_MIN, _X_MAX = -10.0, 100.0
_Y_MIN, _Y_MAX = -25.0,  25.0


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _load_calibration(seq_dir):
    """Return (fx, fy, cx, cy) for ring_front_center."""
    with open(os.path.join(seq_dir, 'vehicle_calibration_info.json')) as f:
        calib = json.load(f)
    cam = next(
        c['value'] for c in calib['camera_data_']
        if c['key'] == 'image_raw_ring_front_center'
    )
    return (
        float(cam['focal_length_x_px_']),
        float(cam['focal_length_y_px_']),
        float(cam['focal_center_x_px_']),
        float(cam['focal_center_y_px_']),
    )


def _parse_img_ts(filename):
    """'ring_front_center_315967467034868064.jpg' → '315967467034868064'"""
    return filename.rsplit('_', 1)[-1].replace('.jpg', '')


def _load_pc_from_npy(pc_path):
    """Load a per-frame .npy point cloud → (4, N) float32 with homogeneous row=1."""
    pc_np = np.load(pc_path)
    if pc_np.ndim == 2 and pc_np.shape[1] == 3:
        pc_np = pc_np.T.astype(np.float32)
    elif pc_np.ndim == 2 and pc_np.shape[1] == 4:
        pc_np = pc_np.T.astype(np.float32)
    else:
        pc_np = pc_np.astype(np.float32)

    if pc_np.shape[0] == 3:
        ones  = np.ones((1, pc_np.shape[1]), dtype=np.float32)
        pc_np = np.vstack([pc_np, ones])
    elif pc_np.shape[0] == 4:
        pc_np[3, :] = 1.
    else:
        raise ValueError(f'Unexpected npy shape after transpose: {pc_np.shape}')

    return torch.from_numpy(pc_np)   # (4, N)


def _load_pc_from_map(map_h5_path, pose_npy_path):
    """
    Crop per-frame point cloud from global map using the camera pose.
    Returns (4, K) float32 tensor in model coordinate (forward, right, down, 1).
    """
    with h5py.File(map_h5_path, 'r') as hf:
        vox_h = hf['PC'][:]          # (4, M) city-relative frame

    cam_T_city = np.load(pose_npy_path).astype(np.float32)   # (4,4)
    local_cam  = cam_T_city @ vox_h                           # (4, M) camera frame

    # Crop: depth in [_X_MIN, _X_MAX], horizontal in [-_Y_MAX, _Y_MAX]
    mask = (
        (local_cam[2] > _X_MIN) & (local_cam[2] < _X_MAX) &
        (local_cam[0] > -_Y_MAX) & (local_cam[0] < _Y_MAX)
    )
    local_cam = local_cam[:, mask]

    # Axis reorder: [cam_z(depth), cam_x(right), cam_y(down), 1]
    local_cam = local_cam[[2, 0, 1, 3], :]
    pc_in = torch.from_numpy(local_cam.astype(np.float32))
    pc_in[3, :] = 1.
    return pc_in


# ─────────────────────────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────────────────────────

class DatasetVisibilityKittiSingle(Dataset):
    """
    Argoverse localization dataset — identical external interface to the KITTI version.

    Args:
        dataset_dir     : root of argoverse-tracking (e.g.,
                          './data/argoverse-tracking').
        maps_folder     : if given and per-frame .npy files exist at
                          <seq_dir>/<maps_folder>/<ts>.npy, those are used as the
                          point cloud source (pre-built GCMLoc from offline model).
                          If not found, falls back to global map.h5 crop.
                          Default 'v2_pcl' matches train_loc.py default.
        maps_root       : accepted for API compatibility; not used for Argoverse.
        split           : 'train' (train1~3) | 'test' (train4)
        max_t, max_r    : perturbation bounds (metres, degrees)
        use_reflectance : accepted for API compatibility; not used in map-crop mode.
        test_sequence   : accepted for API compatibility; not used for Argoverse.
        half_res        : if True, halves image resolution and scales calib.
                          Use together with img_shape=(640, 960) in training script.
    """

    def __init__(self, dataset_dir, transform=None, augmentation=False,
                 maps_folder='v2_pcl', use_reflectance=False,
                 max_t=2., max_r=10., split='train', device='cpu',
                 test_sequence=None, maps_root=None, half_res=False,
                 use_canonical=False):
        super().__init__()

        self.dataset_dir     = dataset_dir
        self.maps_folder     = maps_folder
        self.use_reflectance = use_reflectance
        self.max_r           = max_r
        self.max_t           = max_t
        self.augmentation    = augmentation
        self.transform       = transform
        self.split           = split
        self.device          = device
        self.half_res        = half_res
        self.use_canonical   = use_canonical

        self.model = CameraModel()

        # all_files entries:
        # (seq_dir, img_ts_str, calib_tuple, map_h5_path, pose_npy_path,
        #  img_path, pc_path_or_None)
        # pc_path_or_None: path to pre-built per-frame .npy, or None → use map.h5
        self.all_files = []
        self.GTs_R     = []
        self.GTs_T     = []

        if split == 'test':
            active_splits = _TEST_SPLITS
        elif split == 'val':
            active_splits = _VAL_SPLITS
        else:
            active_splits = _TRAIN_SPLITS
        n_prebuilt = 0
        n_map_crop = 0

        for split_name in active_splits:
            split_dir = os.path.join(dataset_dir, split_name)
            if not os.path.isdir(split_dir):
                print(f'[Argoverse Loc] WARNING: split dir not found: {split_dir}')
                continue

            for seq_uuid in sorted(os.listdir(split_dir)):
                seq_dir = os.path.join(split_dir, seq_uuid)
                if not os.path.isdir(seq_dir):
                    continue

                img_dir     = os.path.join(seq_dir, 'ring_front_center')
                poses_dir   = os.path.join(img_dir, 'poses_torch')
                map_h5_path = os.path.join(seq_dir, 'map.h5')

                if not os.path.isdir(img_dir):
                    continue
                if not os.path.isfile(map_h5_path):
                    print(f'[Argoverse Loc] WARNING: map.h5 not found: {map_h5_path}')
                    continue
                if not os.path.isdir(poses_dir):
                    print(f'[Argoverse Loc] WARNING: poses_torch not found: {poses_dir}')
                    continue

                # Per-frame GCMLoc folder (optional)
                gcmloc_dir = os.path.join(seq_dir, maps_folder) if maps_folder else None

                try:
                    calib = _load_calibration(seq_dir)
                except Exception as e:
                    print(f'[Argoverse Loc] WARNING: calib load failed for {seq_uuid}: {e}')
                    continue

                img_files = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))

                for img_file in img_files:
                    img_ts_str = _parse_img_ts(img_file)
                    stem       = os.path.splitext(img_file)[0]   # ring_front_center_<ts>

                    # Pose npy (required for both fallback and primary modes)
                    pose_npy_path = os.path.join(poses_dir, stem + '.npy')
                    if not os.path.isfile(pose_npy_path):
                        continue

                    img_path = os.path.join(img_dir, img_file)

                    # Per-frame GCMLoc npy (optional): keyed by timestamp only
                    pc_path = None
                    if gcmloc_dir and os.path.isdir(gcmloc_dir):
                        candidate = os.path.join(gcmloc_dir, img_ts_str + '.npy')
                        if os.path.isfile(candidate):
                            pc_path = candidate
                            n_prebuilt += 1
                        else:
                            n_map_crop += 1
                    else:
                        n_map_crop += 1

                    self.all_files.append(
                        (seq_dir, img_ts_str, calib, map_h5_path,
                         pose_npy_path, img_path, pc_path)
                    )
                    self.GTs_R.append(np.array([1., 0., 0., 0.]))
                    self.GTs_T.append(np.zeros(3))

        print(f'[Argoverse Loc] split={split}  splits={active_splits}  '
              f'total frames={len(self.all_files)}  '
              f'(prebuilt={n_prebuilt}, map_crop={n_map_crop})'
              f'{"  [canonical]" if use_canonical else ""}')

        self._remap = {}
        self._canon_calib = UC.get_canonical_calib() if use_canonical else None
        if use_canonical:
            for entry in self.all_files:
                calib = entry[2]
                if calib not in self._remap:
                    self._remap[calib] = UC.build_remap_maps(*calib)

        # ── Test-time fixed perturbations ──────────────────────────────────
        self.test_RT = []
        if split == 'val':
            rt_file = os.path.join(
                dataset_dir,
                f'test_RT_argo_loc_val_{max_r:.2f}_{max_t:.2f}.csv',
            )
            if os.path.exists(rt_file):
                print(f'[Argoverse Loc] VAL: Using {rt_file}')
                import pandas as pd
                df = pd.read_csv(rt_file, sep=',')
                for _, row in df.iterrows():
                    self.test_RT.append(list(row))
            else:
                print(f'[Argoverse Loc] VAL: Generating {rt_file}')
                with open(rt_file, 'w', newline='') as fcsv:
                    writer = csv.writer(fcsv)
                    writer.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                    for i in range(len(self.all_files)):
                        rotz     = np.random.uniform(-max_r, max_r) * (np.pi / 180.)
                        roty     = np.random.uniform(-max_r, max_r) * (np.pi / 180.)
                        rotx     = np.random.uniform(-max_r, max_r) * (np.pi / 180.)
                        transl_x = np.random.uniform(-max_t, max_t)
                        transl_y = np.random.uniform(-max_t, max_t)
                        transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                        writer.writerow([i, transl_x, transl_y, transl_z, rotx, roty, rotz])
                        self.test_RT.append([i, transl_x, transl_y, transl_z, rotx, roty, rotz])
            assert len(self.test_RT) == len(self.all_files), (
                f'test_RT length {len(self.test_RT)} != all_files {len(self.all_files)}'
            )
        if split == 'test':
            rt_file = os.path.join(
                dataset_dir,
                f'test_RT_argo_loc_{max_r:.2f}_{max_t:.2f}.csv',
            )
            if os.path.exists(rt_file):
                print(f'[Argoverse Loc] TEST: Using {rt_file}')
                import pandas as pd
                df = pd.read_csv(rt_file, sep=',')
                for _, row in df.iterrows():
                    self.test_RT.append(list(row))
            else:
                print(f'[Argoverse Loc] TEST: Generating {rt_file}')
                with open(rt_file, 'w', newline='') as fcsv:
                    writer = csv.writer(fcsv)
                    writer.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                    for i in range(len(self.all_files)):
                        rotz     = np.random.uniform(-max_r, max_r) * (np.pi / 180.)
                        roty     = np.random.uniform(-max_r, max_r) * (np.pi / 180.)
                        rotx     = np.random.uniform(-max_r, max_r) * (np.pi / 180.)
                        transl_x = np.random.uniform(-max_t, max_t)
                        transl_y = np.random.uniform(-max_t, max_t)
                        transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                        writer.writerow([i, transl_x, transl_y, transl_z, rotx, roty, rotz])
                        self.test_RT.append([i, transl_x, transl_y, transl_z, rotx, roty, rotz])

            assert len(self.test_RT) == len(self.all_files), (
                f'test_RT length {len(self.test_RT)} != all_files {len(self.all_files)}'
            )

    # ── Pose accessors ─────────────────────────────────────────────────────

    def get_ground_truth_poses(self, frame_idx):
        return self.GTs_T[frame_idx], self.GTs_R[frame_idx]

    # ── Transforms ─────────────────────────────────────────────────────────

    def custom_transform(self, rgb, img_rotation=0., flip=False, remap=None):
        if remap is not None:
            rgb = UC.apply_canonical_warp(rgb, remap[0], remap[1])
        elif self.half_res:
            w, h = rgb.size
            rgb = rgb.resize((w // 2, h // 2), Image.BILINEAR)

        to_tensor     = transforms.ToTensor()
        normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if self.split == 'train':
            rgb = transforms.ColorJitter(0.1, 0.1, 0.1)(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    # ── Dataset protocol ───────────────────────────────────────────────────

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        seq_dir, img_ts_str, calib_tuple, map_h5_path, \
            pose_npy_path, img_path, pc_path = self.all_files[idx]

        # ── Load point cloud ──────────────────────────────────────────────
        try:
            if pc_path is not None:
                # Primary: pre-built per-frame GCMLoc .npy
                pc_in = _load_pc_from_npy(pc_path)
            else:
                # Fallback: crop from global map using camera pose
                pc_in = _load_pc_from_map(map_h5_path, pose_npy_path)
        except Exception as e:
            print(f'[Argoverse Loc] pc load failed idx={idx}: {e}')
            return self.__getitem__(np.random.randint(0, len(self)))

        # ── Horizontal mirror augmentation ────────────────────────────────
        h_mirror = False
        if np.random.rand() > 0.5 and self.split == 'train':
            h_mirror = True
            pc_in[1, :] *= -1

        # ── Load image ────────────────────────────────────────────────────
        try:
            img = Image.open(img_path)
        except OSError:
            return self.__getitem__(np.random.randint(0, len(self)))

        img_rotation = 0.
        if self.split == 'train':
            img_rotation = np.random.uniform(-5, 5)
        remap = self._remap[calib_tuple] if self.use_canonical else None
        try:
            img = self.custom_transform(img, img_rotation, h_mirror, remap)
        except OSError:
            return self.__getitem__(np.random.randint(0, len(self)))

        # ── Rotate point cloud to match image rotation ────────────────────
        if self.split == 'train':
            R_img = mathutils.Euler((radians(img_rotation), 0, 0))
            T_img = mathutils.Vector((0., 0., 0.))
            pc_in = utils.rotate_forward(pc_in, R_img, T_img)

        # ── Calibration tensor [fx, fy, cx, cy] ───────────────────────────
        if self.use_canonical:
            calib = self._canon_calib.clone()
        else:
            calib = torch.tensor(list(calib_tuple), dtype=torch.float32)
            if self.half_res:
                calib = calib / 2.0
        if h_mirror:
            img_w = img.shape[2]
            calib[2] = img_w - calib[2]

        # ── Perturbation ──────────────────────────────────────────────────
        if self.split == 'train':
            rotz     = np.random.uniform(-self.max_r, self.max_r) * (np.pi / 180.)
            roty     = np.random.uniform(-self.max_r, self.max_r) * (np.pi / 180.)
            rotx     = np.random.uniform(-self.max_r, self.max_r) * (np.pi / 180.)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            rt       = self.test_RT[idx]
            transl_x, transl_y, transl_z = rt[1], rt[2], rt[3]
            rotx, roty, rotz             = rt[4], rt[5], rt[6]

        R_pert, T_pert = utils.invert_pose(
            mathutils.Euler((rotx, roty, rotz)),
            mathutils.Vector((transl_x, transl_y, transl_z)),
        )
        R_pert = torch.tensor(R_pert)
        T_pert = torch.tensor(T_pert)

        # ── Build sample dict ─────────────────────────────────────────────
        sample = {
            'rgb':         img,
            'point_cloud': pc_in,
            'calib':       calib,
            'tr_error':    T_pert,
            'rot_error':   R_pert,
            'idx':         idx,
            'rgb_name':    img_ts_str,
        }

        return sample
