# -------------------------------------------------------------------
# Dataset_argoverse_mapping.py  (CMRNext-format maps, GCMLoc)
#
# Argoverse Tracking v1.1 dataset loader for the OFFLINE MAPPING stage.
# Drop-in replacement for DatasetVisibilityKittiSingle (KITTI mapping).
#
# Data structure expected at dataset_dir:
#   <dataset_dir>/
#     train1/<log_id>/
#       map.h5                               ← global point cloud (city-relative frame)
#                                              keys: 'PC' (4,M) float32, 'intensity' (1,M)
#       vehicle_calibration_info.json        ← camera intrinsics
#       ring_front_center/
#         ring_front_center_<ts>.jpg         ← RGB images
#         poses_torch/
#           ring_front_center_<ts>.npy       ← 4×4 cam_T_city_relative pose (float32)
#     train2/... train3/... train4/...
#
# Split assignment:
#   split='train' → train1, train2, train3  (60 logs, ~36k frames)
#   split='test'  → train4                  ( 5 logs, ~2.7k frames)
#
# The sample dict returned by __getitem__ matches the KITTI version exactly:
#   {rgb, point_cloud, calib, tr_error, rot_error, idx, rgb_name}
#
# NOTE on image size:
#   Argoverse ring_front_center images are 1920×1200 (W×H). With the
#   default img_shape=(640, 1920) in train_mapping.py, preprocess_batch
#   will clip the bottom 560 rows. Consider using img_shape=(640, 960)
#   together with half-resolution images (set half_res=True) for better
#   field-of-view coverage.
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

from camera_model_mapping import CameraModel
from utils import invert_pose, rotate_forward
import utils_canonical as UC

# ─────────────────────────────────────────────────────────────────
# Split configuration
# ─────────────────────────────────────────────────────────────────

_TRAIN_SPLITS = ['train1', 'train2', 'train3']
_TEST_SPLITS  = ['train4']

# Vehicle-frame crop window applied to the global map
# (X = forward, Y = left in vehicle/camera convention after axis reorder)
_X_MIN, _X_MAX = -10.0, 100.0
_Y_MIN, _Y_MAX = -25.0,  25.0


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _load_calibration(seq_dir):
    """Return (fx, fy, cx, cy) for ring_front_center from vehicle_calibration_info.json."""
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


# ─────────────────────────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────────────────────────

class DatasetVisibilityKittiSingle(Dataset):
    """
    Argoverse mapping dataset using CMRNext-format global maps (map.h5 + poses_torch).

    Args:
        dataset_dir     : root of argoverse-tracking, e.g.
                          './data/argoverse-tracking'
        maps_folder     : accepted for API compatibility; not used for Argoverse
                          (map is always loaded from <seq_dir>/map.h5).
        maps_root       : accepted for API compatibility; not used for Argoverse.
        split           : 'train' (train1~3) | 'test' (train4)
        max_t, max_r    : perturbation bounds (metres, degrees)
        use_reflectance : if True, also loads intensity from map.h5
        test_sequence   : accepted for API compatibility; not used for Argoverse.
        half_res        : if True, returns images at 960×600 (scale ×0.5) and
                          adjusts calib accordingly. Recommended when using
                          img_shape=(640, 960) in the training script.
    """

    def __init__(self, dataset_dir, transform=None, augmentation=False,
                 maps_folder='argo_local_maps_0.1', use_reflectance=False,
                 max_t=2., max_r=10., split='train', device='cpu',
                 test_sequence=None, maps_root=None, half_res=False,
                 use_canonical=False):
        super().__init__()

        self.dataset_dir     = dataset_dir
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
        # (seq_dir, img_ts_str, calib_tuple, map_h5_path, pose_npy_path, img_path)
        self.all_files = []
        self.GTs_R     = []   # GT rotation (qw, qx, qy, qz) – reserved for future use
        self.GTs_T     = []   # GT translation – reserved for future use

        active_splits = _TEST_SPLITS if split == 'test' else _TRAIN_SPLITS

        for split_name in active_splits:
            split_dir = os.path.join(dataset_dir, split_name)
            if not os.path.isdir(split_dir):
                print(f'[Argoverse Mapping] WARNING: split dir not found: {split_dir}')
                continue

            for seq_uuid in sorted(os.listdir(split_dir)):
                seq_dir = os.path.join(split_dir, seq_uuid)
                if not os.path.isdir(seq_dir):
                    continue

                img_dir    = os.path.join(seq_dir, 'ring_front_center')
                poses_dir  = os.path.join(img_dir, 'poses_torch')
                map_h5_path = os.path.join(seq_dir, 'map.h5')

                if not os.path.isdir(img_dir):
                    continue
                if not os.path.isfile(map_h5_path):
                    print(f'[Argoverse Mapping] WARNING: map.h5 not found: {map_h5_path}')
                    continue
                if not os.path.isdir(poses_dir):
                    print(f'[Argoverse Mapping] WARNING: poses_torch not found: {poses_dir}')
                    continue

                try:
                    calib = _load_calibration(seq_dir)
                except Exception as e:
                    print(f'[Argoverse Mapping] WARNING: calib load failed for {seq_uuid}: {e}')
                    continue

                img_files = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))

                for img_file in img_files:
                    img_ts_str = _parse_img_ts(img_file)

                    # poses_torch npy filename matches the jpg filename (stem + .npy)
                    stem = os.path.splitext(img_file)[0]   # e.g. 'ring_front_center_315967...'
                    pose_npy_path = os.path.join(poses_dir, stem + '.npy')
                    if not os.path.isfile(pose_npy_path):
                        continue

                    img_path = os.path.join(img_dir, img_file)

                    self.all_files.append(
                        (seq_dir, img_ts_str, calib, map_h5_path, pose_npy_path, img_path)
                    )
                    self.GTs_R.append(np.array([1., 0., 0., 0.]))
                    self.GTs_T.append(np.zeros(3))

        print(f'[Argoverse Mapping] split={split}  '
              f'splits={active_splits}  total frames={len(self.all_files)}'
              f'{"  [canonical]" if use_canonical else ""}')

        # one cv2.remap table per distinct source calib
        self._remap = {}
        self._canon_calib = UC.get_canonical_calib() if use_canonical else None
        if use_canonical:
            for entry in self.all_files:
                calib = entry[2]
                if calib not in self._remap:
                    self._remap[calib] = UC.build_remap_maps(*calib)

        # ── Test-time fixed perturbations ──────────────────────────────────
        self.test_RT = []
        if split == 'test':
            rt_file = os.path.join(
                dataset_dir,
                f'test_RT_argo_mapping_{max_r:.2f}_{max_t:.2f}.csv',
            )
            if os.path.exists(rt_file):
                print(f'[Argoverse Mapping] TEST: Using {rt_file}')
                import pandas as pd
                df = pd.read_csv(rt_file, sep=',')
                for _, row in df.iterrows():
                    self.test_RT.append(list(row))
            else:
                print(f'[Argoverse Mapping] TEST: Generating {rt_file}')
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

    # ── Pose accessors ────────────────────────────────────────────────────

    def get_ground_truth_poses(self, frame_idx):
        return self.GTs_T[frame_idx], self.GTs_R[frame_idx]

    # ── Transforms ────────────────────────────────────────────────────────

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

    # ── Dataset protocol ──────────────────────────────────────────────────

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        seq_dir, img_ts_str, calib_tuple, map_h5_path, pose_npy_path, img_path = \
            self.all_files[idx]

        # ── Load global map ───────────────────────────────────────────────
        try:
            with h5py.File(map_h5_path, 'r') as hf:
                vox_h      = hf['PC'][:]          # (4, M) float32, city-relative frame
                vox_intens = hf['intensity'][:]   # (1, M) float32
        except Exception as e:
            print(f'[Argoverse Mapping] map.h5 broken: {map_h5_path} — {e}')
            return self.__getitem__(np.random.randint(0, len(self)))

        # ── Load pose: cam_T_city_relative (4×4) ─────────────────────────
        try:
            cam_T_city = np.load(pose_npy_path).astype(np.float32)   # (4,4)
        except Exception as e:
            print(f'[Argoverse Mapping] pose npy broken: {pose_npy_path} — {e}')
            return self.__getitem__(np.random.randint(0, len(self)))

        # ── Transform global map → camera frame, then crop ───────────────
        local_cam = cam_T_city @ vox_h         # (4, M) in camera frame

        # Crop: cam_z (depth) in [_X_MIN, _X_MAX], cam_x (horiz) in [-_Y_MAX, _Y_MAX]
        mask = (
            (local_cam[2] > _X_MIN) & (local_cam[2] < _X_MAX) &
            (local_cam[0] > -_Y_MAX) & (local_cam[0] < _Y_MAX)
        )
        local_cam    = local_cam[:, mask]        # (4, K)
        local_intens = vox_intens[:, mask]       # (1, K)

        # Axis reorder: [cam_z(depth), cam_x(right), cam_y(down), 1]
        # → model convention: row0=forward, row1=horizontal, row2=vertical
        local_cam = local_cam[[2, 0, 1, 3], :]  # (4, K)

        pc_in = torch.from_numpy(local_cam)      # (4, K) float32
        pc_in[3, :] = 1.

        if self.use_reflectance:
            reflectance = torch.from_numpy(local_intens)   # (1, K)

        # ── Horizontal mirror augmentation ───────────────────────────────
        h_mirror = False
        if np.random.rand() > 0.5 and self.split == 'train':
            h_mirror = True
            pc_in[1, :] *= -1

        # ── Load image ───────────────────────────────────────────────────
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
            pc_in = rotate_forward(pc_in, R_img, T_img)

        # ── Calibration tensor [fx, fy, cx, cy] ──────────────────────────
        if self.use_canonical:
            calib = self._canon_calib.clone()
        else:
            calib = torch.tensor(list(calib_tuple), dtype=torch.float32)
            if self.half_res:
                calib = calib / 2.0   # scale intrinsics with image
        if h_mirror:
            img_w = img.shape[2]
            calib[2] = img_w - calib[2]

        # ── Perturbation ─────────────────────────────────────────────────
        if self.split != 'test':
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

        R_pert, T_pert = invert_pose(
            mathutils.Euler((rotx, roty, rotz)),
            mathutils.Vector((transl_x, transl_y, transl_z)),
        )
        R_pert = torch.tensor(R_pert)
        T_pert = torch.tensor(T_pert)

        # ── Build sample dict ─────────────────────────────────────────────
        # sub_dir: 'train4/<log_uuid>' — used by train_save.py to build GCMLoc save path
        sub_dir = os.path.relpath(seq_dir, self.dataset_dir)

        sample = {
            'rgb':         img,
            'point_cloud': pc_in,
            'calib':       calib,
            'tr_error':    T_pert,
            'rot_error':   R_pert,
            'idx':         idx,
            'rgb_name':    img_ts_str,
            'sub_dir':     sub_dir,
        }
        if self.use_reflectance:
            sample['reflectance'] = reflectance

        return sample
