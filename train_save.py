"""
GCMLoc — Distilled-map saving (Stage-1 inference with a trained mapping checkpoint).

For every frame of the selected split, Stage 1 of GCMLocMapping produces the saliency
heatmap; the top-K (5000) map points are selected and their 3-D coordinates are written to

    <save_root>/<sequence>/<save_name>/<frame>.npy        float32 (K, 3), camera frame

These per-frame sparse maps are the "distilled map" that the online localization stage
reads (train_loc.py / evaluate.py with maps_folder=<save_name>). The network is rebuilt
from the checkpoint's embedded config, validated against its weights; no flags are needed.

Datasets / what gets saved
    dataset=kitti  save_split=test  -> sequence `test_sequence` (run once per sequence)
                   save_split=train -> all sequences except `test_sequence`
    dataset=argo   save_split=train -> train1-3 logs, save_split=test -> train4 logs
    use_canonical=True (+ canon_preset) must be set when the checkpoint was trained in the
    canonical camera (dataset=mixed); save_root/save_name are shared by all datasets.

Examples
--------
python train_save.py with dataset=kitti data_folder=./KITTI_ODOMETRY/sequences test_sequence=0 \\
    weights=<mapping_checkpoint>.tar save_root=./KITTI_ODOMETRY/sequences save_name=v2_pcl

python train_save.py with dataset=argo data_folder=./data/argoverse-tracking save_split=train \\
    weights=<mapping_checkpoint>.tar save_root=./data/argoverse-tracking save_name=v2_pcl

# mixed-trained checkpoint (canonical camera A): add  use_canonical=True canon_preset=A
# (see run_all_saves.sh for the full batch over all sequences / splits)
"""

import math
import os
import random
import time
from datetime import datetime

import mathutils
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import visibility

from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds

from camera_model_mapping import CameraModel
from Dataset_kitti_mapping import DatasetVisibilityKittiSingle as DatasetKitti
from Dataset_argoverse_mapping import DatasetVisibilityKittiSingle as DatasetArgo
import utils_canonical as UC
from models.GCMLoc import load_checkpoint, build_mapping_from_checkpoint
from quaternion_distances import quaternion_distance
from utils import merge_inputs, rotate_back

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

SETTINGS.DISCOVER_DEPENDENCIES = "none"
SETTINGS.DISCOVER_SOURCES = "none"
ex = Experiment("GCMLoc-save")
ex.captured_out_filter = apply_backspaces_and_linefeeds

DATASETS = {'kitti': DatasetKitti, 'argo': DatasetArgo}


@ex.config
def config():
    dataset = 'kitti'               # 'kitti' | 'argo'
    data_folder = './KITTI_ODOMETRY/sequences'   # dataset root
    test_sequence = 0               # kitti: sequence to save (save_split=test) / to exclude (save_split=train)
    maps_folder = 'local_maps_0.1'  # kitti: dense local maps '<seq>/<maps_folder>/*.h5'
    save_split = 'test'             # 'test' | 'train' (see module docstring)
    weights = None                  # REQUIRED: mapping checkpoint (train_mapping.py)
    save_root = './KITTI_ODOMETRY/sequences'   # output root: <save_root>/<sequence>/<save_name>/<frame>.npy
    save_name = 'v2_pcl'            # output folder name (= maps_folder of the localization stage)
    batch_size = 8
    num_worker = 3                  # DataLoader workers
    max_t = 2.                      # initial-pose perturbation range, metres (fixed test_RT csv on the test split)
    max_r = 10.                     # initial-pose perturbation range, degrees
    max_depth = 100.                # depth normalisation constant (points beyond it are dropped)
    use_canonical = False           # warp to the canonical camera (checkpoint trained with dataset=mixed)
    canon_preset = 'A'              # must match the checkpoint's training preset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _init_fn(worker_id, seed):
    seed = seed + worker_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# =====================================================================
# Pre-processing (same projection as train_mapping.py, plus the 3-D point grid)
# =====================================================================

def preprocess_batch(sample, img_shape, max_depth):
    """
    Returns dict: 'rgb', 'lidar' (initial-pose depth), 'depth' (GT-pose depth), 'flagt',
                  'ltolu'/'ltolv' (GT -> initial pixel correspondence),
                  'pcl' (B, 3, H, W) 3-D point of every GT-projected pixel (camera frame).
    """
    out = {k: [] for k in ('rgb', 'lidar', 'depth', 'flagt', 'ltolu', 'ltolv', 'pcl')}
    sample['tr_error'] = sample['tr_error'].cuda()
    sample['rot_error'] = sample['rot_error'].cuda()

    for idx in range(len(sample['rgb'])):
        real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]
        sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
        pcl = sample['point_cloud'][idx].clone()

        R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(sample['tr_error'][idx])
        pc_rotated = rotate_back(pcl, T @ R)
        if max_depth < 100.:
            pc_rotated = pc_rotated[:, pc_rotated[0, :] < max_depth].clone()

        cam_params = sample['calib'][idx].cuda()
        cam_model = CameraModel()
        cam_model.focal_length = cam_params[:2]
        cam_model.principal_point = cam_params[2:]
        uv, uvt, depth, dt, py, px, _ = cam_model.project_pytorch(pc_rotated, pcl, real_shape, None)
        uv = uv.t().int()
        uvt = uvt.t().int()

        depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float) + 1000.
        depth_imgt = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float) + 1000.
        depth_imgt = visibility.depth_image(uvt.contiguous(), dt, depth_imgt, uvt.shape[0], real_shape[1], real_shape[0])
        depth_img = visibility.depth_image(uv.contiguous(), depth, depth_img, uv.shape[0], real_shape[1], real_shape[0])
        depth_imgt[depth_imgt == 1000.] = 0.
        depth_img[depth_img == 1000.] = 0.

        uv = uv.long()
        uvt = uvt.long()
        indexes = depth_img[uv[:, 1], uv[:, 0]] == depth          # visible in both projections

        flagt = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
        flagt[uvt[indexes, 1], uvt[indexes, 0]] = 1
        depth_imgt = flagt * depth_imgt

        # 3-D coordinates (camera frame) of every GT-projected pixel: x = depth, y, z
        pcl_grid = torch.zeros((3,) + tuple(real_shape[:2]), device='cuda', dtype=torch.float)
        pcl_grid[0, uvt[indexes, 1], uvt[indexes, 0]] = dt[indexes]
        pcl_grid[1, uvt[indexes, 1], uvt[indexes, 0]] = py[indexes]
        pcl_grid[2, uvt[indexes, 1], uvt[indexes, 0]] = px[indexes]

        ltolu = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
        ltolv = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
        ltolu[uvt[indexes, 1], uvt[indexes, 0]] = uv[indexes, 1].float()
        ltolv[uvt[indexes, 1], uvt[indexes, 0]] = uv[indexes, 0].float()

        rgb = sample['rgb'][idx].cuda()
        pad = [0, img_shape[1] - rgb.shape[2], 0, img_shape[0] - rgb.shape[1]]
        out['rgb'].append(F.pad(rgb, pad))
        out['lidar'].append(F.pad((depth_img / max_depth).unsqueeze(0), pad))
        out['depth'].append(F.pad((depth_imgt / max_depth).unsqueeze(0), pad))
        out['flagt'].append(F.pad(flagt.unsqueeze(0), pad))
        out['ltolu'].append(F.pad(ltolu.unsqueeze(0), pad))
        out['ltolv'].append(F.pad(ltolv.unsqueeze(0), pad))
        out['pcl'].append(F.pad(pcl_grid, pad))

    return {k: torch.stack(v) for k, v in out.items()}


def frame_info(sample, idx, dataset):
    """(sequence folder, frame name) used to build the output path."""
    if dataset == 'kitti':
        return f"{int(sample['idx'][idx]):02d}", f"{int(sample['rgb_name'][idx]):06d}"
    return str(sample['sub_dir'][idx]), str(sample['rgb_name'][idx])


# =====================================================================
# Main
# =====================================================================

@ex.automain
def main(_config, _run, seed):
    cfg = dict(_config)
    if cfg['dataset'] not in DATASETS:
        raise ValueError(f"dataset must be one of {list(DATASETS)}, got {cfg['dataset']!r}")
    if not cfg['weights']:
        raise ValueError("weights=<mapping checkpoint> is required")
    if cfg['save_split'] not in ('test', 'train'):
        raise ValueError("save_split must be 'test' or 'train'")

    use_canonical = bool(cfg['use_canonical'])
    UC.set_preset(cfg['canon_preset'])
    if use_canonical:
        img_shape = UC.CANON_SHAPE
    else:
        img_shape = (384, 1280) if cfg['dataset'] == 'kitti' else (640, 960)

    print("=" * 60)
    print("GCMLoc — Distilled-map saving (Stage-1 inference)")
    print("=" * 60)
    print(f"Config: {cfg}")

    # ---------- data ----------
    extra = dict(use_canonical=use_canonical)
    if cfg['dataset'] == 'kitti':
        extra.update(test_sequence=f"{int(cfg['test_sequence']):02d}", maps_folder=cfg['maps_folder'])
    else:
        extra.update(half_res=(not use_canonical))
    dataset = DATASETS[cfg['dataset']](cfg['data_folder'], split=cfg['save_split'],
                                       max_r=cfg['max_r'], max_t=cfg['max_t'], **extra)
    print(f"\n[Dataset] {cfg['dataset']}{' canonical(' + cfg['canon_preset'] + ')' if use_canonical else ''}"
          f"  split={cfg['save_split']}  img_shape={img_shape}  frames={len(dataset)}")

    np.random.seed(seed)
    torch.random.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=False, batch_size=cfg['batch_size'], num_workers=cfg['num_worker'],
        worker_init_fn=lambda x: _init_fn(x, seed), collate_fn=merge_inputs, drop_last=False, pin_memory=True)

    # ---------- model ----------
    print(f"\nLoading checkpoint: {cfg['weights']}")
    ckpt = load_checkpoint(cfg['weights'])
    model, arch = build_mapping_from_checkpoint(ckpt, img_shape)
    model = model.to(device).eval()
    print(f"Model: GCMLoc-{arch['dinov2_variant'].upper()}  {arch}  top-K={model.topk_points}"
          f"  (checkpoint epoch {ckpt.get('epoch')})")

    # ---------- save loop ----------
    print(f"\nSaving to: {cfg['save_root']}/<sequence>/{cfg['save_name']}/<frame>.npy")
    total_t = total_r = 0.
    n_done = 0
    start = time.time()

    for batch_idx, sample in enumerate(loader):
        batch = preprocess_batch(sample, img_shape, cfg['max_depth'])
        B = batch['rgb'].shape[0]

        with torch.no_grad():
            transl0, rot0, heatmap, _ = model.stage1(batch['rgb'], batch['lidar'], batch['depth'])
            indices = model.select_points(heatmap, batch['flagt'])           # (B, K)

        pcl_flat = batch['pcl'].reshape(B, 3, -1)
        for i in range(B):
            seq, frame = frame_info(sample, i, cfg['dataset'])
            out_dir = os.path.join(cfg['save_root'], seq, cfg['save_name'])
            os.makedirs(out_dir, exist_ok=True)
            pts = pcl_flat[i, :, indices[i]].t().cpu().numpy()               # (K, 3)
            np.save(os.path.join(out_dir, f'{frame}.npy'), pts)

        # Stage-1 coarse pose error (sanity check of the loaded checkpoint)
        total_t += sum(torch.norm(sample['tr_error'][j] - transl0[j]).item() * 100. for j in range(B))
        total_r += (quaternion_distance(sample['rot_error'], rot0, rot0.device) * 180. / math.pi).sum().item()
        n_done += B

        if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == len(loader):
            ts = datetime.now().strftime('%H:%M:%S')
            print(f'[{ts}] [{batch_idx + 1:4d}/{len(loader)}]  stage-1 t_err={total_t / n_done:.2f} cm  '
                  f'r_err={total_r / n_done:.3f}°  {(time.time() - start) / n_done:.3f}s/frame')

    print("\n" + "=" * 60)
    print(f"Saved {n_done} frames to {cfg['save_root']}/<sequence>/{cfg['save_name']}/  "
          f"({(time.time() - start) / 60:.1f} min)")
    print(f"Stage-1 coarse pose error: t={total_t / n_done:.2f} cm  r={total_r / n_done:.3f}°")
    print("=" * 60)
    _run.result = total_t / n_done
    return _run.result
