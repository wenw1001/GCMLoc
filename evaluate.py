"""
GCMLoc — Iterative localization evaluation.

Applies a sequence of localization checkpoints (rounds 1..N from train_loc.py) to every
test frame. Starting from the fixed initial offset of the test_RT csv, each iteration
projects the distilled sparse map with the current pose estimate, predicts a pose
correction with that round's network and composes it into the estimate. Networks are
rebuilt from the checkpoints (config validated against the weights); GCMLoc-B / GCMLoc-S
is detected automatically. Frames are processed one at a time and the GPU-synchronised
per-module latency of the online network is recorded alongside the pose errors.

Outputs in <results_dir> (default: eval_<first_ckpt>_<seq>/ next to the checkpoint):
    per_sample_iterinitial.csv,
    per_sample_iter<1..N>.csv     per-frame error + composed pose after each iteration
    module_times.csv              per-frame per-module network latency (ms)
    summary.txt                   aggregate errors, error thresholds and timing
plus one structured row appended to timing_records.csv (see timing_record.py).

Examples
--------
python evaluate.py with dataset=kitti \\
    data_folder=./KITTI_ODOMETRY/sequences test_sequence=0 maps_folder=v2_pcl \\
    max_r=10 max_t=2 "weight=['<iter1>.tar','<iter2>.tar','<iter3>.tar']"

python evaluate.py with dataset=argo \\
    data_folder=./data/argoverse-tracking maps_folder=v2_pcl \\
    max_r=10 max_t=2 "weight=['<iter1>.tar','<iter2>.tar','<iter3>.tar']"

# checkpoints trained with dataset=mixed (canonical camera):
#   add  use_canonical=True canon_preset=A maps_folder=v2_pcl_mix
"""

import csv
import os
import time

import mathutils
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import visibility
from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from tqdm import tqdm

from camera_model_localization import CameraModel
from Dataset_kitti_localization import DatasetVisibilityKittiSingle as DatasetKitti
from Dataset_argoverse_localization import DatasetVisibilityKittiSingle as DatasetArgo
import utils_canonical as UC
from models.GCMLoc import TIMED_MODULES, load_checkpoint, build_localization_from_checkpoint
from quaternion_distances import quaternion_distance
from timing_record import record_timing
from utils import merge_inputs, quat2mat, quaternion_from_matrix, rotate_back, rotate_forward, tvector2mat

SETTINGS.DISCOVER_DEPENDENCIES = "none"
SETTINGS.DISCOVER_SOURCES = "none"
ex = Experiment("GCMLoc-eval")
ex.captured_out_filter = apply_backspaces_and_linefeeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# timing_record.py csv keys for the online network modules
MODULE_KEYS = {
    'rgb_backbone': 'rgb_backbone',
    'depth_backbone': 'depth_backbone',
    'heatmap_head_s2': 'heatmap_decoder',
    'flow_s2': 'gcm_flow',
    'pose_reg_s2': 'pose_regression',
}


@ex.config
def config():
    dataset = 'kitti'               # 'kitti' | 'argo'
    data_folder = './KITTI_ODOMETRY/sequences'   # dataset root
    test_sequence = 0               # kitti: evaluated sequence; ignored by argo
    maps_folder = 'v2_pcl'          # distilled-map folder written by train_save.py
    split = 'test'                  # evaluated split
    weight = None                   # REQUIRED: list of *_loc.tar checkpoints, one per iteration (coarse to fine)
    max_t = 2.                      # initial-offset range of the test_RT csv (metres)
    max_r = 10.                     # initial-offset range of the test_RT csv (degrees)
    max_depth = 100.                # depth normalisation constant
    num_worker = 3                  # DataLoader workers
    use_canonical = False           # warp to the canonical camera (checkpoints trained with dataset=mixed)
    canon_preset = 'A'              # must match the checkpoints' training preset
    results_dir = None              # output directory (default: eval_<first_ckpt>_<seq>/ next to the checkpoint)
    method = None                   # label of the timing_records.csv row (e.g. GCMLoc-B)
    timing_csv = None               # override path of timing_records.csv


def project_map(pc, cam_model, real_shape, shape_pad, max_depth):
    """Project a (misaligned) sparse map point cloud into a padded depth image."""
    uv, depth, _, _, _ = cam_model.project_pytorch(pc, real_shape, None)
    uv = uv.t().int()
    depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float) + 1000.
    depth_img = visibility.depth_image(
        uv.contiguous(), depth.contiguous(), depth_img.contiguous(),
        uv.shape[0], real_shape[1], real_shape[0])
    depth_img[depth_img == 1000.] = 0.
    depth_img /= max_depth
    return F.pad(depth_img.unsqueeze(0), shape_pad)


def run_evaluation(cfg):
    """Run the full iterative evaluation for one dataset; returns an aggregate-stats dict.

    `cfg` needs the keys of the Sacred config above (weight already as a list).
    Called directly by evaluate_multidataset.py.
    """
    if cfg['dataset'] not in ('kitti', 'argo'):
        raise ValueError(f"dataset must be 'kitti' or 'argo', got {cfg['dataset']!r}")
    weight = cfg['weight']
    if not weight:
        raise ValueError("weight=<list of localization checkpoints> is required, e.g. "
                         "\"weight=['iter1.tar','iter2.tar','iter3.tar']\"")
    num_iters = len(weight)

    use_canonical = bool(cfg['use_canonical'])
    UC.set_preset(cfg['canon_preset'])
    if cfg['dataset'] == 'kitti':
        seq = f"{int(cfg['test_sequence']):02d}"
        img_shape = UC.CANON_SHAPE if use_canonical else (384, 1280)
    else:
        seq = f"argo_{cfg['split']}"
        img_shape = UC.CANON_SHAPE if use_canonical else (640, 960)

    results_dir = cfg['results_dir']
    if results_dir is None:
        ckpt_name = os.path.splitext(os.path.basename(weight[0]))[0]
        results_dir = os.path.join(os.path.dirname(weight[0]), f"eval_{ckpt_name}_{seq}")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print(f"GCMLoc — Iterative Evaluation ({num_iters} iterations)")
    print("=" * 60)
    print(f"Config: {cfg}")
    print(f"Results dir: {results_dir}")

    # ---------- data ----------
    common = dict(max_r=cfg['max_r'], max_t=cfg['max_t'], split=cfg['split'],
                  maps_folder=cfg['maps_folder'], use_canonical=use_canonical)
    if cfg['dataset'] == 'kitti':
        dataset = DatasetKitti(cfg['data_folder'], test_sequence=seq, **common)
    else:
        dataset = DatasetArgo(cfg['data_folder'], half_res=(not use_canonical), **common)
    loader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=False, batch_size=1, num_workers=cfg['num_worker'],
        collate_fn=merge_inputs, drop_last=False, pin_memory=False)
    print(f"\n[Dataset] {cfg['dataset']}{' canonical(' + cfg['canon_preset'] + ')' if use_canonical else ''}"
          f"  seq={seq}  split={cfg['split']}  img_shape={img_shape}  frames={len(dataset)}")

    # ---------- models (one per iteration) ----------
    models = []
    for i, w in enumerate(weight):
        ckpt = load_checkpoint(w)
        model, arch, missing = build_localization_from_checkpoint(ckpt, img_shape)
        if missing:
            raise ValueError(f"{w} is not a localization checkpoint "
                             f"({len(missing)} missing keys, e.g. {missing[0]})")
        models.append(model.to(device).eval())
        print(f"[Model {i + 1}/{num_iters}] GCMLoc-{arch['dinov2_variant'].upper()} "
              f"(epoch {ckpt.get('epoch')})  {w}")

    # ---------- accumulators ----------
    err_t = [[] for _ in range(num_iters + 1)]          # metres
    err_r = [[] for _ in range(num_iters + 1)]          # radians
    poses = [[] for _ in range(num_iters + 1)]          # composed (tx..rz)
    module_ms = [[] for _ in range(num_iters)]          # per-frame {module: ms}
    prep_times, infer_times = [], []

    identity_q = torch.tensor([1., 0., 0., 0.], device=device).unsqueeze(0)

    def record_error(i, RT):
        T = RT[:3, 3]
        q = quaternion_from_matrix(RT)
        err_t[i].append(T.norm().item())
        err_r[i].append(quaternion_distance(q.unsqueeze(0), identity_q, q.device).item())
        poses[i].append((T[0].item(), T[1].item(), T[2].item(),
                         q[0].item(), q[1].item(), q[2].item(), q[3].item()))

    # ---------- evaluation loop ----------
    for sample in tqdm(loader):
        sample['tr_error'] = sample['tr_error'].cuda()
        sample['rot_error'] = sample['rot_error'].cuda()
        real_shape = [sample['rgb'][0].shape[1], sample['rgb'][0].shape[2], sample['rgb'][0].shape[0]]
        shape_pad = [0, img_shape[1] - real_shape[1], 0, img_shape[0] - real_shape[0]]

        # ----- timed pre-processing: project the map with the initial offset -----
        torch.cuda.synchronize()
        t_prep = time.perf_counter()

        sample['point_cloud'][0] = sample['point_cloud'][0].cuda()
        pcl = sample['point_cloud'][0].clone()
        R = mathutils.Quaternion(sample['rot_error'][0]).to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(sample['tr_error'][0])
        pc_rotated = rotate_back(pcl, T @ R)
        if cfg['max_depth'] < 100.:
            pc_rotated = pc_rotated[:, pc_rotated[0, :] < cfg['max_depth']].clone()

        cam_params = sample['calib'][0].cuda()
        cam_model = CameraModel()
        cam_model.focal_length = cam_params[:2]
        cam_model.principal_point = cam_params[2:]

        lidar = project_map(pc_rotated, cam_model, real_shape, shape_pad, cfg['max_depth']).unsqueeze(0)
        rgb = F.pad(sample['rgb'][0].cuda(), shape_pad).unsqueeze(0)

        torch.cuda.synchronize()
        prep_times.append(time.perf_counter() - t_prep)

        # ----- initial error (iteration 0) -----
        R_mat = quat2mat(sample['rot_error'][0])
        T_mat = tvector2mat(sample['tr_error'][0])
        RT1 = torch.mm(T_mat, R_mat).inverse()
        cloud = rotate_forward(sample['point_cloud'][0], RT1)   # misaligned map (GT frame -> initial frame)
        RTs = [RT1]
        record_error(0, RT1)

        # ----- iterative refinement -----
        torch.cuda.synchronize()
        t_infer = time.perf_counter()
        with torch.no_grad():
            for it in range(num_iters):
                times = {}
                pred_transl, pred_rot = models[it](rgb, lidar, timing=times)
                module_ms[it].append({m: times.get(m, 0.0) * 1000.0 for m in TIMED_MODULES})

                RT_pred = torch.mm(tvector2mat(pred_transl[0]), quat2mat(pred_rot[0]))
                RTs.append(torch.mm(RTs[it], RT_pred))
                record_error(it + 1, RTs[it + 1])

                cloud = rotate_forward(cloud, RT_pred)
                lidar = project_map(cloud, cam_model, real_shape, shape_pad, cfg['max_depth']).unsqueeze(0)
        torch.cuda.synchronize()
        infer_times.append(time.perf_counter() - t_infer)

    # ---------- aggregate ----------
    n_samples = len(err_t[0])
    t_cm = [np.array(e) * 100.0 for e in err_t]
    r_deg = [np.array(e) * (180.0 / np.pi) for e in err_r]

    print("\nIterative refinement:")
    for i in range(num_iters + 1):
        label = 'initial' if i == 0 else f'iter {i}'
        print(f"  {label:8s} t_err mean {t_cm[i].mean():8.4f} cm   median {np.median(t_cm[i]):8.4f} cm | "
              f"r_err mean {r_deg[i].mean():7.4f} °   median {np.median(r_deg[i]):7.4f} °")

    mean_prep_ms = float(np.mean(prep_times)) * 1000.0
    mean_infer_ms = float(np.mean(infer_times)) * 1000.0
    iter_module_mean = [{m: float(np.mean([s[m] for s in module_ms[it]])) for m in TIMED_MODULES}
                        for it in range(num_iters)]
    print(f"\nTiming: preprocess {mean_prep_ms:.2f} ms/frame, inference {mean_infer_ms:.2f} ms/frame "
          f"({num_iters} iters, {1000.0 / mean_infer_ms:.1f} FPS), "
          f"total {mean_prep_ms + mean_infer_ms:.2f} ms/frame")
    for it in range(num_iters):
        d = iter_module_mean[it]
        total = sum(d.values())
        parts = '  '.join(f"{m}={d[m]:.2f}" for m in TIMED_MODULES)
        print(f"  iter {it + 1} modules (ms): {parts}  | total {total:.2f}")

    # ---------- timing_records.csv row ----------
    breakdown = {key: sum(d[m] for d in iter_module_mean) for m, key in MODULE_KEYS.items()}
    record_timing(
        method=cfg['method'] or f"GCMLoc_iter{num_iters}",
        dataset=cfg['dataset'], preprocess_ms=mean_prep_ms, inference_ms=mean_infer_ms,
        modules=breakdown, stage='loc', seq=seq, split=cfg['split'],
        n_samples=n_samples, weights=weight, note=f"{num_iters} iters",
        csv_path=cfg['timing_csv'])

    # ---------- per-iteration csvs ----------
    for i in range(num_iters + 1):
        label = 'initial' if i == 0 else str(i)
        path = os.path.join(results_dir, f'per_sample_iter{label}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame_idx', 't_err_cm', 'r_err_deg', 'tx', 'ty', 'tz', 'rw', 'rx', 'ry', 'rz'])
            for frame in range(n_samples):
                w.writerow([frame, round(t_cm[i][frame], 4), round(r_deg[i][frame], 4)]
                           + [round(v, 6) for v in poses[i][frame]])
    print(f"[Saved] per_sample_iter*.csv → {results_dir}")

    # ---------- module_times.csv ----------
    path = os.path.join(results_dir, 'module_times.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['sample_idx']
        for it in range(num_iters):
            header += [f'iter{it + 1}_{m}_ms' for m in TIMED_MODULES] + [f'iter{it + 1}_total_ms']
        w.writerow(header)
        for s_idx in range(n_samples):
            row = [s_idx]
            for it in range(num_iters):
                vals = [module_ms[it][s_idx][m] for m in TIMED_MODULES]
                row += [round(v, 4) for v in vals] + [round(sum(vals), 4)]
            w.writerow(row)
    print(f"[Saved] module_times.csv → {path}")

    # ---------- summary.txt ----------
    path = os.path.join(results_dir, 'summary.txt')
    with open(path, 'w') as f:
        f.write(f"Model      : GCMLoc (iterative, {num_iters} iterations)\n")
        f.write(f"Weights    : {weight}\n")
        f.write(f"Dataset    : {cfg['dataset']}  seq={seq}  split={cfg['split']}\n")
        f.write(f"Initial RT : max_r={cfg['max_r']:g} deg  max_t={cfg['max_t']:g} m (fixed test_RT csv)\n")
        f.write(f"N samples  : {n_samples}\n")
        f.write(f"preprocess : {mean_prep_ms:.2f} ms/frame\n")
        f.write(f"inference  : {mean_infer_ms:.2f} ms/frame ({num_iters} iters; "
                f"{mean_infer_ms / num_iters:.2f} ms/iter)\n")
        f.write(f"total      : {mean_prep_ms + mean_infer_ms:.2f} ms/frame "
                f"({1000.0 / (mean_prep_ms + mean_infer_ms):.2f} FPS)\n\n")
        for it in range(num_iters):
            d = iter_module_mean[it]
            total = sum(d.values())
            f.write(f"=== Per-module network latency : iteration {it + 1} ===\n")
            for m in TIMED_MODULES:
                f.write(f"  {m:<18s}: {d[m]:7.2f} ms ({100.0 * d[m] / total if total else 0.0:5.1f}%)\n")
            f.write(f"  {'total':<18s}: {total:7.2f} ms\n\n")
        for i in range(num_iters + 1):
            label = 'initial' if i == 0 else f'iteration {i}'
            f.write(f"=== {label} ===\n")
            f.write(f"t_err mean {t_cm[i].mean():.4f} cm   median {np.median(t_cm[i]):.4f} cm\n")
            f.write(f"r_err mean {r_deg[i].mean():.4f} °    median {np.median(r_deg[i]):.4f} °\n")
            for thresh in (10, 20, 30, 50, 100):
                f.write(f"t_err < {thresh:3d} cm : {100.0 * np.mean(t_cm[i] < thresh):5.1f}%\n")
            for thresh in (1, 2, 5):
                f.write(f"r_err < {thresh} deg  : {100.0 * np.mean(r_deg[i] < thresh):5.1f}%\n")
            f.write("\n")
    print(f"[Saved] summary.txt → {path}")

    return {
        'dataset': cfg['dataset'], 'seq': seq, 'n_samples': n_samples,
        'init_t_cm': float(t_cm[0].mean()), 'init_r_deg': float(r_deg[0].mean()),
        'mean_t_cm': float(t_cm[-1].mean()), 'med_t_cm': float(np.median(t_cm[-1])),
        'mean_r_deg': float(r_deg[-1].mean()), 'med_r_deg': float(np.median(r_deg[-1])),
        'prep_ms': mean_prep_ms, 'infer_ms': mean_infer_ms,
        'fps': 1000.0 / (mean_prep_ms + mean_infer_ms),
    }


@ex.automain
def main(_config):
    cfg = dict(_config)
    weight = cfg['weight']
    if isinstance(weight, str):
        cfg['weight'] = [weight]
    stats = run_evaluation(cfg)
    return stats['med_t_cm']
