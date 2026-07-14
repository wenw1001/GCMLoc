# -------------------------------------------------------------------
# Dataset_itri_localization.py  (ITRI-campus, GCMLoc)
#
# Online LOCALIZATION-stage loader for the self-captured ITRI-campus dataset.
#
# Point cloud source:
#   Primary  : pre-built per-frame GCMLoc at
#              processed/sequences/<seq>/<maps_folder>/<ts>.npy   (from train_save)
#   Fallback : strategy-C submap crop from the 5 cm map (same as the mapping
#              dataset) when no per-frame GCMLoc exists yet.
#
# Everything else (split, calib, poses, images) matches Dataset_itri_mapping.
# Sample dict: {rgb, point_cloud, calib, tr_error, rot_error, idx, rgb_name}
# -------------------------------------------------------------------

import csv
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

from camera_model_localization import CameraModel
import utils
import utils_canonical as UC

# reuse the strategy-C submap machinery from the mapping loader
from Dataset_itri_mapping import _SubmapIndex, _X_MIN, _X_MAX, _Y_MAX


def _load_pc_from_npy(pc_path):
    """Pre-built GCMLoc .npy → (4, N) float32 with homogeneous row = 1."""
    pc = np.load(pc_path)
    if pc.ndim == 2 and pc.shape[1] in (3, 4):
        pc = pc.T
    pc = pc.astype(np.float32)
    if pc.shape[0] == 3:
        pc = np.vstack([pc, np.ones((1, pc.shape[1]), dtype=np.float32)])
    elif pc.shape[0] == 4:
        pc[3, :] = 1.
    else:
        raise ValueError(f'unexpected GCMLoc npy shape {pc.shape}')
    return torch.from_numpy(pc)


class DatasetVisibilityKittiSingle(Dataset):
    """ITRI-campus localization dataset (GCMLoc npy primary, submap-crop fallback)."""

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

        # Point-cloud source policy (NO silent fallback):
        #   maps_folder set (e.g. 'v2_pcl') → REQUIRE pre-built GCMLoc .npy for
        #       every frame; raise if any is missing (run train_save first).
        #   maps_folder=''                  → explicitly use submap-crop.
        # all_files: (seq, ts_str, calib_tuple, pose_path, img_path, pc_path_or_None)
        self.all_files = []
        self.GTs_R, self.GTs_T = [], []
        n_prebuilt = n_crop = 0
        missing = []
        for seq, lists in splits.items():
            calib = self._load_calib(seq)
            pose_dir  = os.path.join(proc, 'sequences', seq, 'poses_torch')
            img_dir   = os.path.join(raw, 'sequences', seq, 'camera_front')
            gcmloc_dir = os.path.join(proc, 'sequences', seq, maps_folder) \
                if maps_folder else None
            for ts in lists[split_key]:
                pose_path = os.path.join(pose_dir, f'{ts}.npy')
                img_path  = os.path.join(img_dir, f'{ts}.jpg')
                if not (os.path.isfile(pose_path) and os.path.isfile(img_path)):
                    continue
                pc_path = None
                if gcmloc_dir:
                    cand = os.path.join(gcmloc_dir, f'{ts}.npy')
                    if os.path.isfile(cand):
                        pc_path = cand; n_prebuilt += 1
                    else:
                        missing.append(cand)        # strict: record, fail later
                        continue
                else:
                    n_crop += 1
                self.all_files.append((seq, str(ts), calib, pose_path,
                                       img_path, pc_path))
                self.GTs_R.append(np.array([1., 0., 0., 0.]))
                self.GTs_T.append(np.zeros(3))

        if missing:
            raise RuntimeError(
                f"[ITRI Loc] {len(missing)} frames have no pre-built GCMLoc under "
                f"maps_folder='{maps_folder}' (e.g. {missing[0]}). "
                f"Run train_save.py (dataset=itri) first, or pass maps_folder='' "
                f"to use on-the-fly submap-crop.")

        print(f'[ITRI Loc] split={split}  frames={len(self.all_files)}  '
              f'(prebuilt={n_prebuilt}, submap_crop={n_crop})'
              f'{"  [canonical]" if use_canonical else ""}')

        self._remap = {}
        self._canon_calib = UC.get_canonical_calib() if use_canonical else None
        if use_canonical:
            for _, _, calib, _, _, _ in self.all_files:
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
                               f'test_RT_itri_loc_{max_r:.2f}_{max_t:.2f}.csv')
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
        """Strategy-C submap crop fallback → (4, K) [forward, right, down, 1]."""
        R = cam_T_map[:3, :3]; t = cam_T_map[:3, 3]
        cx, cy = (-R.T @ t)[:2]
        paths = self.submaps.query(cx, cy)
        if not paths:
            return torch.zeros((4, 0), dtype=torch.float32)
        from Dataset_itri_mapping import _load_submap_xyz
        xyz = np.concatenate([_load_submap_xyz(p) for p in paths], axis=1)
        pc = np.vstack([xyz, np.ones((1, xyz.shape[1]), dtype=np.float32)])
        local = cam_T_map.astype(np.float32) @ pc
        m = ((local[2] > _X_MIN) & (local[2] < _X_MAX) &
             (np.abs(local[0]) < _Y_MAX))
        local = local[:, m][[2, 0, 1, 3], :]
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
        seq, ts_str, calib_tuple, pose_path, img_path, pc_path = self.all_files[idx]

        try:
            if pc_path is not None:
                pc_in = _load_pc_from_npy(pc_path)
            else:
                pc_in = self._local_map(np.load(pose_path).astype(np.float32))
        except Exception as e:
            print(f'[ITRI Loc] pc load failed idx={idx}: {e}')
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
            pc_in = utils.rotate_forward(pc_in, R_img, T_img)

        if self.use_canonical:
            calib = self._canon_calib.clone()
        else:
            calib = torch.tensor(list(calib_tuple), dtype=torch.float32)
            if self.half_res:
                calib = calib / 2.0
        if h_mirror:
            calib[2] = img.shape[2] - calib[2]

        if self.split == 'train':
            rotz = np.random.uniform(-self.max_r, self.max_r) * (np.pi / 180.)
            roty = np.random.uniform(-self.max_r, self.max_r) * (np.pi / 180.)
            rotx = np.random.uniform(-self.max_r, self.max_r) * (np.pi / 180.)
            tx = np.random.uniform(-self.max_t, self.max_t)
            ty = np.random.uniform(-self.max_t, self.max_t)
            tz = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            rt = self.test_RT[idx]
            tx, ty, tz, rotx, roty, rotz = rt[1], rt[2], rt[3], rt[4], rt[5], rt[6]

        R_pert, T_pert = utils.invert_pose(mathutils.Euler((rotx, roty, rotz)),
                                           mathutils.Vector((tx, ty, tz)))
        R_pert = torch.tensor(R_pert)
        T_pert = torch.tensor(T_pert)

        return {
            'rgb': img, 'point_cloud': pc_in, 'calib': calib,
            'tr_error': T_pert, 'rot_error': R_pert,
            'idx': idx, 'rgb_name': ts_str,
        }
