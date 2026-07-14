"""
GCMLoc — Iterative Evaluation with Per-Module Timing.

Multi-checkpoint iterative refinement evaluation.  Additionally measures
and reports the wall time of each sub-module (rgb_backbone, depth_backbone,
heatmap_head_s2, flow_s2, pose_reg_s2) for every iteration.  Per-module
timing is printed after all samples are processed and saved to
summary.txt / module_times.csv.

Output:
    <results_dir>/
        per_sample_iterinitial.csv   — per-frame initial errors
        per_sample_iter1.csv ...     — per-frame errors after each iter
        module_times.csv             — per-sample per-module latency (ms)
        summary.txt                  — aggregate stats + per-module timing

Example:
    python evaluate.py with \\
        test_sequence=0 max_r=10 max_t=2 \\
        data_folder=./KITTI_ODOMETRY/sequences \\
        maps_folder=v2_pcl \\
        "weight=['iter1.tar','iter2.tar','iter3.tar']" \\
        results_dir=./results_eachtime

The network architecture is rebuilt from the config stored inside each
checkpoint; pass rgb_backbone= / depth_backbone= / flow_type= /
unfreeze_dinov2_blocks= only to override it.
"""

import csv
import os
import time

import mathutils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import visibility
from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from tqdm import tqdm

from camera_model_localization import CameraModel
from Dataset_kitti_localization import DatasetVisibilityKittiSingle
from Dataset_argoverse_localization import DatasetVisibilityKittiSingle as DatasetArgo
try:
    from Dataset_itri_localization import DatasetVisibilityKittiSingle as DatasetItri
except ModuleNotFoundError:
    DatasetItri = None  # itri dataset optional; not needed for kitti/argo eval
import utils_canonical as UC
from timing_record import record_timing
from models.GCMLoc.GCMLoc_localization_eachtime import GCMLocLocalization
from quaternion_distances import quaternion_distance
from utils import (mat2xyzrpy, merge_inputs, quat2mat, quaternion_from_matrix,
                   rotate_back, rotate_forward, tvector2mat)

SETTINGS.DISCOVER_DEPENDENCIES = "none"
SETTINGS.DISCOVER_SOURCES = "none"
ex = Experiment("GCMLoc-eachtime-eval")
ex.captured_out_filter = apply_backspaces_and_linefeeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODULE_NAMES = ['rgb_backbone', 'depth_backbone', 'heatmap_head_s2', 'flow_s2', 'pose_reg_s2']

# Default weights (overridden by CLI 'weight=[...]') — one checkpoint per
# refinement iteration, coarse to fine.
weight = [
    './checkpoints/loc_iter1.tar',
    './checkpoints/loc_iter2.tar',
    './checkpoints/loc_iter3.tar',
]


@ex.config
def config():
    dataset         = 'kitti'   # 'kitti' | 'argo' | 'itri'
    weight          = None
    test_sequence   = 0
    data_folder     = './KITTI_ODOMETRY/sequences'
    maps_folder     = 'v2_pcl'
    split           = 'test'
    use_canonical   = False     # warp to canonical camera (mixed-trained model)
    canon_preset    = 'A'       # must match the checkpoint's training preset
    max_r           = 10.
    max_t           = 2.
    batch_size      = 1
    num_worker      = 6
    use_reflectance = False
    max_depth       = 100.
    save_name       = None
    results_dir     = None
    method          = None      # label for the timing table column (e.g. Ours-S)
    timing_csv      = None       # override path for the timing records CSV

    rgb_backbone            = None
    unfreeze_dinov2_blocks  = None
    depth_backbone          = None
    flow_type               = None
    feat_dim                = None
    heatmap_dim_s2          = None


def _build_model(checkpoint, img_shape, overrides=None):
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint has no 'config' key.")
    cfg = dict(checkpoint['config'])

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                print(f"  [OVERRIDE] {k}: {cfg.get(k, '?')} → {v}")
                cfg[k] = v

    state_dict = checkpoint['state_dict']
    legacy = 'depth_backbone.branch_lhmap.level0.0.0.weight' in state_dict
    if legacy:
        print("  [INFO] Legacy depth branches detected.")

    print(f"  rgb={cfg.get('rgb_backbone','cnn')} "
          f"depth={cfg.get('depth_backbone','cnn')} "
          f"flow={cfg.get('flow_type','correlation')} "
          f"dinov2_unfreeze={cfg.get('unfreeze_dinov2_blocks',0)}")

    model = GCMLocLocalization(
        image_size=img_shape,
        feat_dim=cfg.get('feat_dim', 128),
        heatmap_dim_s2=cfg.get('heatmap_dim_s2', 128),
        vmamba_output_stage=cfg.get('vmamba_output_stage', 2),
        use_cnn_fallback=cfg.get('use_cnn_fallback', False),
        dropout=0.0,
        use_reflectance=cfg.get('use_reflectance', False),
        rgb_backbone=cfg.get('rgb_backbone', 'cnn'),
        dinov2_variant=cfg.get('dinov2_variant', 'b'),
        unfreeze_dinov2_blocks=cfg.get('unfreeze_dinov2_blocks', 0),
        depth_backbone=cfg.get('depth_backbone', 'cnn'),
        flow_type=cfg.get('flow_type', 'correlation'),
        legacy_depth_branches=legacy,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        print(f"  [WARNING] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return model


def _project_depth(pc, real_shape, cam_params, max_depth, shape_pad):
    cam_model = CameraModel()
    cam_model.focal_length    = cam_params[:2]
    cam_model.principal_point = cam_params[2:]
    uv, depth, _py, _px, _refl = cam_model.project_pytorch(pc, real_shape, None)
    uv = uv.t().int()
    depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float) + 1000.
    depth_img = visibility.depth_image(
        uv.contiguous(), depth, depth_img, uv.shape[0], real_shape[1], real_shape[0])
    depth_img[depth_img == 1000.] = 0.
    depth_img /= max_depth
    depth_img = F.pad(depth_img.unsqueeze(0), shape_pad)
    return depth_img, cam_model


def _print_module_timing(iter_idx, iter_module_ms_list):
    """Print Per-Module Timing for one iteration (averaged over all samples so far)."""
    n = len(iter_module_ms_list)
    if n == 0:
        return
    print(f"\n--- Per-Module Timing : Iteration {iter_idx} (mean over {n} samples) ---")
    total = sum(np.mean([s[m] for s in iter_module_ms_list]) for m in MODULE_NAMES)
    for m in MODULE_NAMES:
        ms  = float(np.mean([s[m] for s in iter_module_ms_list]))
        pct = 100.0 * ms / total if total > 0 else 0.0
        print(f"  {m:<22s}: {ms:7.2f} ms  ({pct:.1f}%)")
    print(f"  {'total (modules)':<22s}: {total:7.2f} ms")
    print()


@ex.automain
def main(_config):
    global weight
    if _config['weight'] is not None:
        weight = _config['weight']
    if weight is None:
        raise ValueError("weight must be specified (list of checkpoint paths).")
    if isinstance(weight, str):
        weight = [weight]

    datasetType = {'kitti': 0, 'argo': 1, 'itri': 2}.get(
        _config.get('dataset', 'kitti'), 1)
    uc = _config.get('use_canonical', False)
    UC.set_preset(_config.get('canon_preset', 'A'))
    num_iters   = len(weight)

    if datasetType == 0:
        seq       = f"{int(_config['test_sequence']):02d}"
        img_shape = (384, 1280)
    elif datasetType == 2:
        seq       = f"itri_{_config['split']}"
        img_shape = (640, 960)
    else:
        seq       = f"argo_{_config['split']}"
        img_shape = (640, 960)
    if uc:
        img_shape = UC.CANON_SHAPE

    # ===== Results Directory =====
    if _config['results_dir'] is not None:
        results_dir = _config['results_dir']
    else:
        ckpt_dir  = os.path.dirname(weight[0])
        ckpt_name = os.path.splitext(os.path.basename(weight[0]))[0]
        results_dir = os.path.join(ckpt_dir, f"eval_eachtime_{ckpt_name}_{seq}")
    os.makedirs(results_dir, exist_ok=True)

    # ===== Dataset =====
    if datasetType == 0:
        dataset = DatasetVisibilityKittiSingle(
            _config['data_folder'],
            max_r=_config['max_r'], max_t=_config['max_t'],
            split=_config['split'],
            use_reflectance=_config['use_reflectance'],
            maps_folder=_config['maps_folder'],
            test_sequence=seq,
            use_canonical=uc,
        )
    elif datasetType == 2:
        dataset = DatasetItri(
            _config['data_folder'],
            max_r=_config['max_r'], max_t=_config['max_t'],
            split=_config['split'],
            use_reflectance=_config['use_reflectance'],
            maps_folder=_config['maps_folder'],
            use_canonical=uc,
        )
    else:
        dataset = DatasetArgo(
            _config['data_folder'],
            max_r=_config['max_r'], max_t=_config['max_t'],
            split=_config['split'],
            use_reflectance=_config['use_reflectance'],
            maps_folder=_config['maps_folder'],
            half_res=(not uc),
            use_canonical=uc,
        )
    loader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=False,
        batch_size=_config['batch_size'],
        num_workers=_config['num_worker'],
        collate_fn=merge_inputs,
        drop_last=False, pin_memory=False,
    )
    print(len(loader))

    # ===== Load all models =====
    arch_overrides = {k: _config.get(k) for k in
                      ('rgb_backbone', 'unfreeze_dinov2_blocks', 'depth_backbone',
                       'flow_type', 'feat_dim', 'heatmap_dim_s2')}
    models = []
    for i, w in enumerate(weight):
        if not w:
            # Empty path: reuse the previous model (iteration not yet trained)
            if models:
                print(f"\n[Model {i+1}/{num_iters}] (empty — reusing model {i})")
                models.append(models[-1])
            else:
                raise ValueError(f"weight[{i}] is empty and there is no previous model to reuse.")
            continue
        print(f"\n[Model {i+1}/{num_iters}] {w}")
        ckpt = torch.load(w, map_location=device)
        m = _build_model(ckpt, img_shape, overrides=arch_overrides)
        m = m.to(device)
        m.eval()
        models.append(m)

    # ===== Error Accumulators =====
    errors_r   = [[] for _ in range(num_iters + 1)]
    errors_t   = [[] for _ in range(num_iters + 1)]
    errors_t2  = [[] for _ in range(num_iters + 1)]
    errors_rpy = [[] for _ in range(num_iters + 1)]
    per_sample_t  = [[] for _ in range(num_iters + 1)]
    per_sample_r  = [[] for _ in range(num_iters + 1)]
    per_sample_rt = [[] for _ in range(num_iters + 1)]
    infer_times = []
    prep_times  = []

    # per_iter_module_ms[i] = list of {module: ms} dicts, one per sample, for iteration i
    per_iter_module_ms = [[] for _ in range(num_iters)]

    # ===== Inference Loop =====
    for batch_idx, sample in enumerate(tqdm(loader)):

        sample['tr_error']  = sample['tr_error'].cuda()
        sample['rot_error'] = sample['rot_error'].cuda()

        lidar_input = []
        rgb_input   = []
        shape_pad   = [0, 0, 0, 0]

        # ===== Timed Pre-process (depth projection + tensor prep) =====
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_prep = time.perf_counter()

        for idx in range(len(sample['rgb'])):
            real_shape = [sample['rgb'][idx].shape[1],
                          sample['rgb'][idx].shape[2],
                          sample['rgb'][idx].shape[0]]

            sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
            pcl = sample['point_cloud'][idx].clone()

            R  = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
            R.resize_4x4()
            T  = mathutils.Matrix.Translation(sample['tr_error'][idx])
            RT = T @ R
            pc_rotated = rotate_back(pcl, RT)

            if _config['max_depth'] < 100.:
                pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()

            cam_params = sample['calib'][idx].cuda()
            shape_pad[3] = img_shape[0] - sample['rgb'][idx].shape[1]
            shape_pad[1] = img_shape[1] - sample['rgb'][idx].shape[2]

            depth_img, cam_model = _project_depth(
                pc_rotated, real_shape, cam_params, _config['max_depth'], shape_pad)

            rgb = F.pad(sample['rgb'][idx].cuda(), shape_pad)
            rgb_input.append(rgb)
            lidar_input.append(depth_img)

        lidar_input = torch.stack(lidar_input)
        rgb_input   = torch.stack(rgb_input)

        rgb           = rgb_input.to(device)
        lidar         = lidar_input.to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prep_times.append(time.perf_counter() - t_prep)

        target_transl = sample['tr_error'].to(device)
        target_rot    = sample['rot_error'].to(device)

        # ===== Initial Error (iteration 0) =====
        point_cloud = sample['point_cloud'][0].to(device)

        R_mat = quat2mat(target_rot[0])
        T_mat = tvector2mat(target_transl[0])
        RT1_inv = torch.mm(T_mat, R_mat)
        RT1 = RT1_inv.clone().inverse()
        rotated_point_cloud = rotate_forward(point_cloud, RT1)
        RTs = [RT1]

        T_composed = RT1[:3, 3]
        R_composed = quaternion_from_matrix(RT1)
        t_err_raw  = T_composed.norm().item()
        r_err_raw  = quaternion_distance(
            R_composed.unsqueeze(0),
            torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0),
            R_composed.device)
        errors_t[0].append(t_err_raw)
        errors_t2[0].append(T_composed)
        errors_r[0].append(r_err_raw)
        per_sample_t[0].append(t_err_raw)
        per_sample_r[0].append(r_err_raw.item() if torch.is_tensor(r_err_raw) else r_err_raw)
        per_sample_rt[0].append((
            T_composed[0].item(), T_composed[1].item(), T_composed[2].item(),
            R_composed[0].item(), R_composed[1].item(), R_composed[2].item(), R_composed[3].item(),
        ))
        rpy_error = mat2xyzrpy(RT1)[3:] * (180.0 / 3.141592)
        errors_rpy[0].append(rpy_error)

        if batch_idx == 0:
            print(f'Initial T_erorr: {errors_t[0]}')
            print(f'Initial R_erorr: {errors_r[0]}')

        # ===== Timed Iterative Forward Pass =====
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            for iteration in range(num_iters):
                pred_transl, pred_rot, _wx, _wq, mod_times = models[iteration](rgb, lidar)

                # Record per-module ms for this sample / iteration
                per_iter_module_ms[iteration].append(
                    {m: mod_times.get(m, 0.0) * 1000.0 for m in MODULE_NAMES}
                )

                R_predicted  = quat2mat(pred_rot[0])
                T_predicted  = tvector2mat(pred_transl[0])
                RT_predicted = torch.mm(T_predicted, R_predicted)
                RTs.append(torch.mm(RTs[iteration], RT_predicted))

                rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)

                uv2, depth2, _py2, _px2, _refl2 = cam_model.project_pytorch(
                    rotated_point_cloud, real_shape, None)
                uv2 = uv2.t().int()
                depth_img2 = torch.zeros(real_shape[:2], device=device) + 1000.
                depth_img2 = visibility.depth_image(
                    uv2.contiguous(), depth2.contiguous(),
                    depth_img2.contiguous(),
                    uv2.shape[0], real_shape[1], real_shape[0])
                depth_img2[depth_img2 == 1000.] = 0.
                depth_img2 /= _config['max_depth']
                depth_img2 = F.pad(depth_img2, shape_pad)
                lidar = depth_img2.unsqueeze(0).unsqueeze(0)

                T_composed = RTs[iteration + 1][:3, 3]
                R_composed = quaternion_from_matrix(RTs[iteration + 1])
                t_err_raw  = T_composed.norm().item()
                r_err_raw  = quaternion_distance(
                    R_composed.unsqueeze(0),
                    torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0),
                    R_composed.device)
                errors_t[iteration + 1].append(t_err_raw)
                errors_t2[iteration + 1].append(T_composed)
                errors_r[iteration + 1].append(r_err_raw)
                per_sample_t[iteration + 1].append(t_err_raw)
                per_sample_r[iteration + 1].append(
                    r_err_raw.item() if torch.is_tensor(r_err_raw) else r_err_raw)
                per_sample_rt[iteration + 1].append((
                    T_composed[0].item(), T_composed[1].item(), T_composed[2].item(),
                    R_composed[0].item(), R_composed[1].item(), R_composed[2].item(), R_composed[3].item(),
                ))
                rpy_error = mat2xyzrpy(RTs[iteration + 1])[3:] * (180.0 / 3.141592)
                errors_rpy[iteration + 1].append(rpy_error)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_times.append(time.perf_counter() - t0)

    # ===== Print Summary (same style as CMRNet evaluate script) =====
    print("Iterative refinement: ")
    for i in range(num_iters + 1):
        errors_r[i] = torch.tensor(errors_r[i]) * (180.0 / 3.141592)
        errors_t[i] = torch.tensor(errors_t[i]) * 100
        print(f"Iteration {i}: \tMean Translation Error: {errors_t[i].mean():.4f} cm "
              f"     Mean Rotation Error: {errors_r[i].mean():.4f} °")
        print(f"Iteration {i}: \tMedian Translation Error: {errors_t[i].median():.4f} cm "
              f"     Median Rotation Error: {errors_r[i].median():.4f} °\n")

    print("-------------------------------------------------------")
    for i in range(num_iters + 1):
        errors_t2[i]  = torch.stack(errors_t2[i])
        errors_rpy[i] = torch.stack(errors_rpy[i])

    # Per-Module Timing per iteration
    for iteration in range(num_iters):
        _print_module_timing(iteration + 1, per_iter_module_ms[iteration])

    plt.plot(errors_t2[-1][:, 0].cpu().numpy())
    plt.show()
    plt.plot(errors_t2[-1][:, 1].cpu().numpy())
    plt.show()
    plt.plot(errors_t2[-1][:, 2].cpu().numpy())
    plt.show()

    if _config['save_name'] is not None:
        torch.save(torch.stack(errors_t).cpu().numpy(),
                   os.path.join(results_dir, f'{_config["save_name"]}_errors_t'))
        torch.save(torch.stack(errors_r).cpu().numpy(),
                   os.path.join(results_dir, f'{_config["save_name"]}_errors_r'))
        torch.save(torch.stack(errors_t2).cpu().numpy(),
                   os.path.join(results_dir, f'{_config["save_name"]}_errors_t2'))
        torch.save(torch.stack(errors_rpy).cpu().numpy(),
                   os.path.join(results_dir, f'{_config["save_name"]}_errors_rpy'))

    # ===== Compute Stats =====
    all_t_cm  = [np.array(per_sample_t[i]) * 100.0           for i in range(num_iters + 1)]
    all_r_deg = [np.array(per_sample_r[i]) * (180.0 / np.pi) for i in range(num_iters + 1)]

    def pct_within(arr, thresh):
        return 100.0 * np.mean(arr < thresh)

    n_samples     = len(per_sample_t[0])
    mean_infer_ms = float(np.mean(infer_times)) * 1000.0
    mean_prep_ms  = float(np.mean(prep_times)) * 1000.0 if prep_times else 0.0
    mean_total_ms = mean_prep_ms + mean_infer_ms
    fps           = 1.0 / float(np.mean(infer_times))

    # Per-iteration per-module mean ms
    iter_module_mean_ms = []
    for iteration in range(num_iters):
        d = {}
        for m in MODULE_NAMES:
            d[m] = float(np.mean([s[m] for s in per_iter_module_ms[iteration]]))
        iter_module_mean_ms.append(d)

    # ===== Per-module breakdown summed across all iterations (ms/sample) =====
    # Maps internal module names -> canonical keys used by the timing table.
    _MODULE_MAP = {
        'rgb_backbone':    'rgb_backbone',
        'depth_backbone':  'depth_backbone',
        'heatmap_head_s2': 'cross_modal_fusion',
        'flow_s2':         'gcm_flow',
        'pose_reg_s2':     'pose_regression',
    }
    module_breakdown = {v: 0.0 for v in _MODULE_MAP.values()}
    for d in iter_module_mean_ms:
        for m, key in _MODULE_MAP.items():
            module_breakdown[key] += d.get(m, 0.0)

    # ===== Record one structured timing row (for the LaTeX table) =====
    method_label = _config.get('method') or f"iter{num_iters}_{_config.get('rgb_backbone') or 'model'}"
    record_timing(
        method=method_label,
        dataset=_config['dataset'],
        preprocess_ms=mean_prep_ms,
        inference_ms=mean_infer_ms,
        modules=module_breakdown,
        stage='loc',
        seq=seq,
        split=_config['split'],
        n_samples=n_samples,
        weights=weight,
        note=f"{num_iters} iters",
        csv_path=_config.get('timing_csv'),
    )

    if datasetType == 0:
        test_RT_file = os.path.join(
            _config['data_folder'],
            f'test_RT_seq{seq}_{_config["max_r"]:.2f}_{_config["max_t"]:.2f}.csv'
        )
    elif datasetType == 2:
        test_RT_file = os.path.join(
            _config['data_folder'],
            f'test_RT_itri_loc_{_config["max_r"]:.2f}_{_config["max_t"]:.2f}.csv'
        )
    else:
        test_RT_file = os.path.join(
            _config['data_folder'],
            f'test_RT_argo_loc_{_config["max_r"]:.2f}_{_config["max_t"]:.2f}.csv'
        )

    # ===== Save per-iteration CSVs =====
    for i in range(num_iters + 1):
        label = 'initial' if i == 0 else str(i)
        csv_path = os.path.join(results_dir, f'per_sample_iter{label}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'frame_idx', 't_err_cm', 'r_err_deg',
                'tx', 'ty', 'tz', 'rw', 'rx', 'ry', 'rz',
            ])
            writer.writeheader()
            for frame in range(n_samples):
                tx, ty, tz, rw, rx, ry, rz = per_sample_rt[i][frame]
                writer.writerow({
                    'frame_idx': frame,
                    't_err_cm':  round(all_t_cm[i][frame], 4),
                    'r_err_deg': round(all_r_deg[i][frame], 4),
                    'tx': round(tx, 6), 'ty': round(ty, 6), 'tz': round(tz, 6),
                    'rw': round(rw, 6), 'rx': round(rx, 6),
                    'ry': round(ry, 6), 'rz': round(rz, 6),
                })
        print(f"[Saved] per_sample_iter{label}.csv → {csv_path}")

    # ===== Save module_times.csv =====
    mod_csv_path = os.path.join(results_dir, 'module_times.csv')
    mod_fieldnames = ['sample_idx']
    for iteration in range(num_iters):
        for m in MODULE_NAMES:
            mod_fieldnames.append(f'iter{iteration+1}_{m}_ms')
        mod_fieldnames.append(f'iter{iteration+1}_total_ms')
    with open(mod_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=mod_fieldnames)
        writer.writeheader()
        for s in range(n_samples):
            row = {'sample_idx': s}
            for iteration in range(num_iters):
                tot = 0.0
                for m in MODULE_NAMES:
                    ms = per_iter_module_ms[iteration][s][m]
                    row[f'iter{iteration+1}_{m}_ms'] = round(ms, 4)
                    tot += ms
                row[f'iter{iteration+1}_total_ms'] = round(tot, 4)
            writer.writerow(row)
    print(f"[Saved] module_times.csv → {mod_csv_path}")

    # ===== Save summary.txt =====
    summary_path = os.path.join(results_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Model      : GCMLoc (iterative, {num_iters} iterations)\n")
        f.write(f"Weights    : {weight}\n")
        f.write(f"Sequence   : {seq}\n")
        f.write(f"Split      : {_config['split']}\n")
        f.write(f"Test RT    : {test_RT_file}\n")
        f.write(f"N samples  : {n_samples}\n")
        f.write(f"preprocess : {mean_prep_ms:.2f} ms/sample\n")
        f.write(f"infer time : {mean_infer_ms:.2f} ms/sample ({num_iters} iters total)\n")
        f.write(f"total      : {mean_total_ms:.2f} ms/sample (preprocess + inference)\n")
        f.write(f"infer time : {mean_infer_ms/num_iters:.2f} ms/sample (per iter)\n")
        f.write(f"fps        : {fps:.2f} ({num_iters} iters)\n")
        f.write(f"fps        : {fps*num_iters:.2f} (per iter)\n\n")

        for iteration in range(num_iters):
            d     = iter_module_mean_ms[iteration]
            total = sum(d.values())
            f.write(f"=== Per-Module Timing : Iteration {iteration+1} ===\n")
            for m in MODULE_NAMES:
                ms  = d[m]
                pct = 100.0 * ms / total if total > 0 else 0.0
                f.write(f"  {m:<22s}: {ms:7.2f} ms  ({pct:.1f}%)\n")
            f.write(f"  {'total (modules)':<22s}: {total:7.2f} ms\n\n")

        for i in range(num_iters + 1):
            t_cm  = all_t_cm[i]
            r_deg = all_r_deg[i]
            mean_t = float(np.mean(t_cm))
            med_t  = float(np.median(t_cm))
            mean_r = float(np.mean(r_deg))
            med_r  = float(np.median(r_deg))
            f.write(f"=== Iteration {i} ===\n")
            f.write(f"t_err mean : {mean_t:.4f} cm\n")
            f.write(f"t_err med  : {med_t:.4f} cm\n")
            f.write(f"r_err mean : {mean_r:.4f} °\n")
            f.write(f"r_err med  : {med_r:.4f} °\n\n")
            for thresh in [10, 20, 30, 50, 100]:
                f.write(f"t_err < {thresh:3d}cm  : {pct_within(t_cm, thresh):.1f}%\n")
            f.write("\n")
            for thresh in [1, 2, 5]:
                f.write(f"r_err < {thresh}°      : {pct_within(r_deg, thresh):.1f}%\n")
            f.write("\n")

    print(f"[Saved] summary.txt → {summary_path}")
    print("End!")
