# -------------------------------------------------------------------
# Dataset_itri_mapping.py  (ITRI-campus, GCMLoc)
#
# Offline MAPPING-stage loader for the self-captured ITRI-campus dataset.
# Drop-in replacement for DatasetVisibilityKittiSingle (KITTI / Argoverse).
#
# Map source = STRATEGY C: at run time, look up the 5 cm voxel submaps near the
# camera (50 m grid, centres in submaps_config.json), load only those (LRU
# cached), transform to the camera frame and crop. Keeps full resolution, never
# loads the whole 119 M-point map, no per-frame storage duplication.
#
# Expected layout (dataset_dir = .../iter_campus):
#   processed/splits.json
#   processed/sequences/<seq>/poses_torch/<ts>.npy   ← 4x4 cam_T_map
#   processed/sequences/<seq>/pinhole_calib.json     ← {fx,fy,cx,cy,...}
#   raw/submaps_config.json                          ← submap centres
#   raw/pcd_map/vg05_filtered/<file>.npy             ← 5 cm submaps
#   raw/sequences/<seq>/camera_front/<ts>.jpg        ← rectified pinhole images
#
# Sample dict (identical to KITTI/Argoverse):
#   {rgb, point_cloud, calib, tr_error, rot_error, idx, rgb_name, sub_dir}
# -------------------------------------------------------------------

import csv
import functools
import json
import os
from math import radians

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

# Crop window in the camera optical frame (metres):
#   depth (forward, cam z) in [_X_MIN, _X_MAX]; horizontal (cam x) in ±_Y_MAX
_X_MIN, _X_MAX = -10.0, 80.0
_Y_MAX = 25.0

_SUBMAP_CELL = 50.0          # submap grid cell size (m)
_SUBMAP_RADIUS = 90.0        # load submaps whose centre is within this box (m)


# ─────────────────────────────────────────────────────────────────
# Submap store (shared, LRU-cached, full 5 cm resolution)
# ─────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=64)
def _load_submap_xyz(path):
    """Load a vg05_filtered submap → (3, N) float32 xyz in the map frame.

    Cached so neighbouring frames that share submaps don't re-read from disk.
    Intensity is dropped (use_reflectance is not used in the map-crop path).
    """
    a = np.load(path, allow_pickle=False)
    xyz = np.ascontiguousarray(a['centroids']).astype(np.float32).reshape(-1, 3)
    return xyz.T.copy()          # (3, N)


class _SubmapIndex:
    """Spatial index over submaps_config.json: pick submaps near a position."""

    def __init__(self, submaps_config_path, submap_dir):
        cfg = json.load(open(submaps_config_path))
        self.centres = []        # (cx, cy, abspath)
        for s in cfg['submaps']:
            p = os.path.join(submap_dir, s['file_name'])
            if os.path.isfile(p):
                self.centres.append((float(s['center_x']),
                                     float(s['center_y']), p))
        self._cx = np.array([c[0] for c in self.centres])
        self._cy = np.array([c[1] for c in self.centres])

    def query(self, x, y, radius=_SUBMAP_RADIUS):
        m = (np.abs(self._cx - x) <= radius) & (np.abs(self._cy - y) <= radius)
        return [self.centres[i][2] for i in np.nonzero(m)[0]]


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────

class DatasetVisibilityKittiSingle(Dataset):
    """ITRI-campus mapping dataset (strategy C submap lookup)."""

    def __init__(self, dataset_dir, transform=None, augmentation=False,
                 maps_folder=None, use_reflectance=False,
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
        self.model           = CameraModel()

        proc = os.path.join(dataset_dir, 'processed')
        raw  = os.path.join(dataset_dir, 'raw')
        self._proc = proc

        self.submaps = _SubmapIndex(
            os.path.join(raw, 'submaps_config.json'),
            os.path.join(raw, 'pcd_map', 'vg05_filtered'),
        )

        splits = json.load(open(os.path.join(proc, 'splits.json')))['sequences']
        split_key = 'test' if split == 'test' else 'train'

        # all_files: (seq, ts_str, calib_tuple, pose_path, img_path)
        self.all_files = []
        self.GTs_R, self.GTs_T = [], []
        for seq, lists in splits.items():
            calib = self._load_calib(seq)
            pose_dir = os.path.join(proc, 'sequences', seq, 'poses_torch')
            img_dir  = os.path.join(raw, 'sequences', seq, 'camera_front')
            for ts in lists[split_key]:
                pose_path = os.path.join(pose_dir, f'{ts}.npy')
                img_path  = os.path.join(img_dir, f'{ts}.jpg')
                if not (os.path.isfile(pose_path) and os.path.isfile(img_path)):
                    continue
                self.all_files.append((seq, str(ts), calib, pose_path, img_path))
                self.GTs_R.append(np.array([1., 0., 0., 0.]))
                self.GTs_T.append(np.zeros(3))

        print(f'[ITRI Mapping] split={split}  frames={len(self.all_files)}'
              f'{"  [canonical]" if use_canonical else ""}')

        # precompute one cv2.remap table per distinct source calib
        self._remap = {}
        self._canon_calib = UC.get_canonical_calib() if use_canonical else None
        if use_canonical:
            for _, _, calib, _, _ in self.all_files:
                if calib not in self._remap:
                    self._remap[calib] = UC.build_remap_maps(*calib)

        self.test_RT = []
        if split == 'test':
            self._init_test_RT(max_r, max_t)

    # ── helpers ────────────────────────────────────────────────────────────

    def _load_calib(self, seq):
        c = json.load(open(os.path.join(
            self._proc, 'sequences', seq, 'pinhole_calib.json')))
        return (c['fx'], c['fy'], c['cx'], c['cy'])

    def _init_test_RT(self, max_r, max_t):
        rt_file = os.path.join(self.dataset_dir,
                               f'test_RT_itri_mapping_{max_r:.2f}_{max_t:.2f}.csv')
        if os.path.exists(rt_file):
            import pandas as pd
            for _, row in pd.read_csv(rt_file).iterrows():
                self.test_RT.append(list(row))
        else:
            with open(rt_file, 'w', newline='') as f:
                w = csv.writer(f); w.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (np.pi / 180.)
                    roty = np.random.uniform(-max_r, max_r) * (np.pi / 180.)
                    rotx = np.random.uniform(-max_r, max_r) * (np.pi / 180.)
                    tx = np.random.uniform(-max_t, max_t)
                    ty = np.random.uniform(-max_t, max_t)
                    tz = np.random.uniform(-max_t, min(max_t, 1.))
                    w.writerow([i, tx, ty, tz, rotx, roty, rotz])
                    self.test_RT.append([i, tx, ty, tz, rotx, roty, rotz])
        assert len(self.test_RT) == len(self.all_files)

    def _local_map(self, cam_T_map):
        """Strategy C: gather nearby 5 cm submaps, transform to camera, crop.

        Returns (4, K) float32 in model coords [forward, right, down, 1].
        """
        # camera centre in the map frame = -R^T t
        R = cam_T_map[:3, :3]; t = cam_T_map[:3, 3]
        cx, cy = (-R.T @ t)[:2]

        paths = self.submaps.query(cx, cy)
        if not paths:
            return torch.zeros((4, 0), dtype=torch.float32)
        xyz = np.concatenate([_load_submap_xyz(p) for p in paths], axis=1)  # (3,M)

        pc = np.vstack([xyz, np.ones((1, xyz.shape[1]), dtype=np.float32)])  # (4,M)
        local = cam_T_map.astype(np.float32) @ pc                            # camera frame
        m = ((local[2] > _X_MIN) & (local[2] < _X_MAX) &
             (np.abs(local[0]) < _Y_MAX))
        local = local[:, m]
        local = local[[2, 0, 1, 3], :]      # → [forward, right, down, 1]
        pc_in = torch.from_numpy(np.ascontiguousarray(local))
        pc_in[3, :] = 1.
        return pc_in

    def get_ground_truth_poses(self, frame_idx):
        return self.GTs_T[frame_idx], self.GTs_R[frame_idx]

    def custom_transform(self, rgb, img_rotation=0., flip=False, remap=None):
        if remap is not None:
            rgb = UC.apply_canonical_warp(rgb, remap[0], remap[1])
        elif self.half_res:
            w, h = rgb.size
            rgb = rgb.resize((w // 2, h // 2), Image.BILINEAR)
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        if self.split == 'train':
            rgb = transforms.ColorJitter(0.1, 0.1, 0.1)(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        seq, ts_str, calib_tuple, pose_path, img_path = self.all_files[idx]

        cam_T_map = np.load(pose_path).astype(np.float32)   # (4,4)
        try:
            pc_in = self._local_map(cam_T_map)
        except Exception as e:
            print(f'[ITRI Mapping] local map failed idx={idx}: {e}')
            return self.__getitem__(np.random.randint(0, len(self)))
        if pc_in.shape[1] == 0:
            return self.__getitem__(np.random.randint(0, len(self)))

        h_mirror = False
        if np.random.rand() > 0.5 and self.split == 'train':
            h_mirror = True
            pc_in[1, :] *= -1

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

        if self.split == 'train':
            R_img = mathutils.Euler((radians(img_rotation), 0, 0))
            T_img = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R_img, T_img)

        if self.use_canonical:
            calib = self._canon_calib.clone()
        else:
            calib = torch.tensor(list(calib_tuple), dtype=torch.float32)
            if self.half_res:
                calib = calib / 2.0
        if h_mirror:
            calib[2] = img.shape[2] - calib[2]

        if self.split != 'test':
            rotz = np.random.uniform(-self.max_r, self.max_r) * (np.pi / 180.)
            roty = np.random.uniform(-self.max_r, self.max_r) * (np.pi / 180.)
            rotx = np.random.uniform(-self.max_r, self.max_r) * (np.pi / 180.)
            tx = np.random.uniform(-self.max_t, self.max_t)
            ty = np.random.uniform(-self.max_t, self.max_t)
            tz = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            rt = self.test_RT[idx]
            tx, ty, tz, rotx, roty, rotz = rt[1], rt[2], rt[3], rt[4], rt[5], rt[6]

        R_pert, T_pert = invert_pose(mathutils.Euler((rotx, roty, rotz)),
                                     mathutils.Vector((tx, ty, tz)))
        R_pert = torch.tensor(R_pert)
        T_pert = torch.tensor(T_pert)

        sample = {
            'rgb': img, 'point_cloud': pc_in, 'calib': calib,
            'tr_error': T_pert, 'rot_error': R_pert,
            'idx': idx, 'rgb_name': ts_str,
            # train_save writes to {save_root}/{sub_dir}/{save_name}/{ts}.npy;
            # set save_root=<iter_campus>/processed/sequences so GCMLoc lands in
            # processed/sequences/<seq>/<save_name>/ where the loc dataset reads.
            'sub_dir': seq,
        }
        return sample
