"""
GCMLoc — Single-Checkpoint Localization Inference Script.

Loads a trained GCMLocLocalization checkpoint and evaluates it on the
test split (fixed perturbations from test_RT CSV).

Output:
    <results_dir>/
        per_sample.csv     — per-frame t_err_cm, r_err_deg, predictions & GT
        summary.txt        — aggregate mean/median/percentile stats

Example:
    python infer_loc.py with \
        test_sequence=0 max_r=10 max_t=2 \
        data_folder=./KITTI_ODOMETRY/sequences \
        maps_folder=v2_pcl \
        weights='./checkpoints/loc_iter1.tar'

The architecture is rebuilt from the config stored in the checkpoint;
the override flags below are only needed for legacy checkpoints.
"""

import csv
import math
import os
import time

import mathutils
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds

from camera_model_localization import CameraModel
from Dataset_kitti_localization import DatasetVisibilityKittiSingle
from losses import ProposedLoss
from models.GCMLoc.GCMLoc_localization import GCMLocLocalization
from quaternion_distances import quaternion_distance
from utils import merge_inputs, rotate_back

SETTINGS.DISCOVER_DEPENDENCIES = "none"
SETTINGS.DISCOVER_SOURCES = "none"
ex = Experiment("GCMLoc-infer")
ex.captured_out_filter = apply_backspaces_and_linefeeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasetType = 0


@ex.config
def config():
    weights         = None          # REQUIRED: path to .tar checkpoint
    test_sequence   = 0             # KITTI sequence index
    data_folder     = './KITTI_ODOMETRY/sequences'
    maps_folder     = 'v2_pcl'      # must match the save_name used in train_save
    split           = 'test'        # 'test' (fixed RT) or 'train' (random RT)
    max_r           = 10.
    max_t           = 2.
    batch_size      = 1
    num_worker      = 3
    use_reflectance = False
    max_depth       = 100.
    results_dir     = None          # if None, saves next to the checkpoint file

    # ===== Architecture overrides (use when checkpoint config is wrong) =====
    # Leave as None to use the architecture stored in the checkpoint config.
    # Set explicitly if the checkpoint config was saved incorrectly.
    rgb_backbone            = None   # e.g. 'cnn' or 'dinov2'
    unfreeze_dinov2_blocks  = None   # e.g. 0 or 4
    depth_backbone          = None   # e.g. 'cnn' or 'vmamba'
    flow_type               = None   # e.g. 'correlation' or 'multi_scale'
    feat_dim                = None   # e.g. 128
    heatmap_dim_s2          = None   # e.g. 128


def _build_model(checkpoint, img_shape, overrides=None):
    """
    Build model from checkpoint config, with optional CLI overrides.
    overrides: dict of keys that take precedence over checkpoint['config'].
    """
    if 'config' in checkpoint:
        cfg = dict(checkpoint['config'])
        print("[INFO] Architecture loaded from checkpoint config.")
    else:
        raise ValueError("Checkpoint has no 'config' key — cannot determine architecture.")

    # Apply overrides (CLI values that are not None override checkpoint config)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                print(f"[OVERRIDE] {k}: {cfg.get(k, '?')} → {v}")
                cfg[k] = v

    state_dict = checkpoint['state_dict']
    legacy = 'depth_backbone.branch_lhmap.level0.0.0.weight' in state_dict
    if legacy:
        print("[INFO] Legacy depth branches detected — using legacy_depth_branches=True.")

    print(f"[INFO] Building model: rgb={cfg.get('rgb_backbone','cnn')} "
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


@ex.automain
def main(_config):
    if _config['weights'] is None:
        raise ValueError("weights must be specified.")

    seq = f"{int(_config['test_sequence']):02d}"
    img_shape = (384, 1280) if datasetType == 0 else (640, 960)

    # ===== Results Directory =====
    if _config['results_dir'] is not None:
        results_dir = _config['results_dir']
    else:
        ckpt_dir = os.path.dirname(_config['weights'])
        ckpt_name = os.path.splitext(os.path.basename(_config['weights']))[0]
        results_dir = os.path.join(ckpt_dir, f"infer_{ckpt_name}_seq{seq}")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("GCMLoc — Ablation Inference")
    print("=" * 60)
    print(f"Checkpoint  : {_config['weights']}")
    print(f"Sequence    : {seq}")
    print(f"Split       : {_config['split']}")
    print(f"Results dir : {results_dir}")

    # ===== Dataset =====
    dataset = DatasetVisibilityKittiSingle(
        _config['data_folder'],
        max_r=_config['max_r'],
        max_t=_config['max_t'],
        split=_config['split'],
        use_reflectance=_config['use_reflectance'],
        maps_folder=_config['maps_folder'],
        test_sequence=seq,
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=_config['batch_size'],
        num_workers=_config['num_worker'],
        collate_fn=merge_inputs,
        drop_last=False,
        pin_memory=True,
    )
    print(f"Dataset size: {len(dataset)} samples")

    # ===== Model =====
    print(f"\nLoading checkpoint: {_config['weights']}")
    checkpoint = torch.load(_config['weights'], map_location=device)
    arch_overrides = {
        'rgb_backbone':           _config.get('rgb_backbone'),
        'unfreeze_dinov2_blocks': _config.get('unfreeze_dinov2_blocks'),
        'depth_backbone':         _config.get('depth_backbone'),
        'flow_type':              _config.get('flow_type'),
        'feat_dim':               _config.get('feat_dim'),
        'heatmap_dim_s2':         _config.get('heatmap_dim_s2'),
    }
    model = _build_model(checkpoint, img_shape, overrides=arch_overrides)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    loss_fn = ProposedLoss(rescale_trans=1, rescale_rot=1)

    # ===== Inference =====
    per_sample_rows = []
    total_t_err = 0.
    total_r_err = 0.
    total_loss  = 0.
    n_samples   = 0
    infer_times = []   # model-only forward time per batch (seconds)

    print(f"\nRunning inference on {len(dataset)} samples...")

    for batch_idx, sample in enumerate(loader):
        lidar_input = []
        rgb_input   = []

        sample['tr_error']  = sample['tr_error'].cuda()
        sample['rot_error'] = sample['rot_error'].cuda()

        for idx in range(len(sample['rgb'])):
            real_shape = [
                sample['rgb'][idx].shape[1],
                sample['rgb'][idx].shape[2],
                sample['rgb'][idx].shape[0],
            ]

            sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
            pcl = sample['point_cloud'][idx].clone()

            reflectance = None
            if _config['use_reflectance']:
                reflectance = sample['reflectance'][idx].cuda()

            R  = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
            R.resize_4x4()
            T  = mathutils.Matrix.Translation(sample['tr_error'][idx])
            RT = T @ R
            pc_rotated = rotate_back(pcl, RT)

            if _config['max_depth'] < 100.:
                pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()

            cam_params = sample['calib'][idx].cuda()
            cam_model  = CameraModel()
            cam_model.focal_length    = cam_params[:2]
            cam_model.principal_point = cam_params[2:]

            uv, depth, py, px, refl = cam_model.project_pytorch(
                pc_rotated, real_shape, reflectance,
            )
            uv        = uv.t().int()
            depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float) + 1000.
            import visibility
            depth_img = visibility.depth_image(
                uv.contiguous(), depth, depth_img, uv.shape[0], real_shape[1], real_shape[0],
            )
            depth_img[depth_img == 1000.] = 0.
            depth_img /= _config['max_depth']
            depth_img  = depth_img.unsqueeze(0)

            rgb = sample['rgb'][idx].cuda()
            shape_pad    = [0, 0, 0, 0]
            shape_pad[3] = img_shape[0] - rgb.shape[1]
            shape_pad[1] = img_shape[1] - rgb.shape[2]
            rgb       = F.pad(rgb,       shape_pad)
            depth_img = F.pad(depth_img, shape_pad)

            rgb_input.append(rgb)
            lidar_input.append(depth_img)

        lidar_input = torch.stack(lidar_input)
        rgb_input   = torch.stack(rgb_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_transl, pred_rot, w_x, w_q = model(rgb_input, lidar_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_times.append((time.perf_counter() - t0) / rgb_input.shape[0])

        batch_loss = loss_fn(
            sample['tr_error'], sample['rot_error'],
            pred_transl, pred_rot,
        )

        r_err_batch = quaternion_distance(
            sample['rot_error'], pred_rot, pred_rot.device
        ) * 180. / math.pi  # (B,)

        B = rgb_input.shape[0]
        for i in range(B):
            t_err_cm = torch.norm(sample['tr_error'][i] - pred_transl[i]).item() * 100.
            r_err_deg = r_err_batch[i].item()

            gt_t = sample['tr_error'][i].cpu().numpy()
            gt_r = sample['rot_error'][i].cpu().numpy()
            p_t  = pred_transl[i].cpu().numpy()
            p_r  = pred_rot[i].cpu().numpy()

            per_sample_rows.append({
                'frame_idx':   batch_idx * _config['batch_size'] + i,
                't_err_cm':    round(t_err_cm, 4),
                'r_err_deg':   round(r_err_deg, 4),
                'gt_tx': round(float(gt_t[0]), 6), 'gt_ty': round(float(gt_t[1]), 6),
                'gt_tz': round(float(gt_t[2]), 6),
                'gt_rw': round(float(gt_r[0]), 6), 'gt_rx': round(float(gt_r[1]), 6),
                'gt_ry': round(float(gt_r[2]), 6), 'gt_rz': round(float(gt_r[3]), 6),
                'pred_tx': round(float(p_t[0]), 6), 'pred_ty': round(float(p_t[1]), 6),
                'pred_tz': round(float(p_t[2]), 6),
                'pred_rw': round(float(p_r[0]), 6), 'pred_rx': round(float(p_r[1]), 6),
                'pred_ry': round(float(p_r[2]), 6), 'pred_rz': round(float(p_r[3]), 6),
            })

            total_t_err += t_err_cm
            total_r_err += r_err_deg
            n_samples   += 1

        total_loss += batch_loss.item() * B

        if (batch_idx + 1) % 50 == 0:
            print(
                f"  [{batch_idx+1:4d}/{len(loader)}] "
                f"running mean: t={total_t_err/n_samples:.2f}cm  r={total_r_err/n_samples:.4f}°"
            )

    # ===== Aggregate Stats =====
    t_errs = np.array([r['t_err_cm']  for r in per_sample_rows])
    r_errs = np.array([r['r_err_deg'] for r in per_sample_rows])

    mean_t  = float(np.mean(t_errs))
    med_t   = float(np.median(t_errs))
    mean_r  = float(np.mean(r_errs))
    med_r   = float(np.median(r_errs))
    mean_loss = total_loss / n_samples
    mean_infer_ms = float(np.mean(infer_times)) * 1000.
    fps = 1. / float(np.mean(infer_times))

    def pct_within(arr, thresh):
        return 100. * np.mean(arr < thresh)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"N samples   : {n_samples}")
    print(f"Val loss    : {mean_loss:.6f}")
    print(f"infer time  : {mean_infer_ms:.2f} ms/sample")
    print(f"fps         : {fps:.2f}")
    print(f"t_err  mean : {mean_t:.4f} cm")
    print(f"t_err  med  : {med_t:.4f} cm")
    print(f"r_err  mean : {mean_r:.4f} °")
    print(f"r_err  med  : {med_r:.4f} °")
    print()
    for thresh in [10, 20, 30, 50, 100]:
        print(f"  t_err < {thresh:3d}cm  : {pct_within(t_errs, thresh):.1f}%")
    print()
    for thresh in [1, 2, 5]:
        print(f"  r_err < {thresh}°      : {pct_within(r_errs, thresh):.1f}%")
    print("=" * 60)

    # ===== Save per-sample CSV =====
    csv_path = os.path.join(results_dir, 'per_sample.csv')
    fieldnames = list(per_sample_rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample_rows)
    print(f"\n[Saved] per_sample.csv → {csv_path}")

    # ===== Save summary =====
    summary_path = os.path.join(results_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Checkpoint : {_config['weights']}\n")
        f.write(f"Sequence   : {seq}\n")
        f.write(f"Split      : {_config['split']}\n")
        f.write(f"N samples  : {n_samples}\n")
        f.write(f"infer time : {mean_infer_ms:.2f} ms/sample\n")
        f.write(f"fps        : {fps:.2f}\n\n")
        f.write(f"val_loss   : {mean_loss:.6f}\n")
        f.write(f"t_err mean : {mean_t:.4f} cm\n")
        f.write(f"t_err med  : {med_t:.4f} cm\n")
        f.write(f"r_err mean : {mean_r:.4f} °\n")
        f.write(f"r_err med  : {med_r:.4f} °\n\n")
        for thresh in [10, 20, 30, 50, 100]:
            f.write(f"t_err < {thresh:3d}cm  : {pct_within(t_errs, thresh):.1f}%\n")
        f.write("\n")
        for thresh in [1, 2, 5]:
            f.write(f"r_err < {thresh}°      : {pct_within(r_errs, thresh):.1f}%\n")
    print(f"[Saved] summary.txt   → {summary_path}")
