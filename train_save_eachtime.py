"""
GCMLoc — Stage-1 Per-Module Timing Script.

Same as train_save.py but does NOT save any .npy files.
Runs Stage-1 inference with per-module GPU timing and prints the results.

Example:
    python train_save_eachtime.py with \
        data_folder='./KITTI_ODOMETRY/sequences' \
        test_sequence=0 \
        weights='./checkpoints/mapping.tar'
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
from Dataset_kitti_mapping import DatasetVisibilityKittiSingle
from losses import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss
from models.GCMLoc.GCMLoc_save_eachtime import GCMLocSave
from quaternion_distances import quaternion_distance
from timing_record import record_timing
from utils import merge_inputs, rotate_back

datasetType = 0  # 0 --- kitti  1 --- argo

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

SETTINGS.DISCOVER_DEPENDENCIES = "none"
SETTINGS.DISCOVER_SOURCES = "none"
ex = Experiment("GCMLoc-save-eachtime")
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    savemodel       = './checkpoints/'
    dataset         = 'kitti'
    data_folder     = './KITTI_ODOMETRY/sequences'
    use_reflectance = False
    test_sequence   = 0
    occlusion_kernel    = 5
    occlusion_threshold = 3
    BASE_LEARNING_RATE  = 1e-4
    loss            = 'simple'
    max_t           = 2.
    max_r           = 10.
    batch_size      = 8
    num_worker      = 3
    resume          = None
    weights         = None          # REQUIRED: path to train_mapping.py checkpoint
    rescale_rot     = 10
    rescale_transl  = 1
    dropout         = 0.0
    max_depth       = 100.
    maps_folder     = 'local_maps_0.1'
    method          = None      # label for the timing table column (e.g. Ours-S)
    timing_csv      = None       # override path for the timing records CSV

    # ===== GCMLoc architecture (overridden by checkpoint config when weights provided) =====
    feat_dim             = 128
    heatmap_dim_s1       = 64
    topk_points          = 5000
    vmamba_output_stage  = 2
    use_cnn_fallback     = False

    # ===== Architecture flags (overridden by checkpoint config when weights provided) =====
    # rgb_backbone: 'cnn', 'dinov2b' (ViT-B/14), 'dinov2s' (ViT-S/14), 'dinov2' (uses dinov2_variant)
    rgb_backbone         = 'dinov2b'
    dinov2_variant       = 'b'           # 'b'=ViT-B/14, 's'=ViT-S/14; used when rgb_backbone='dinov2'
    unfreeze_dinov2_blocks = 4
    depth_backbone       = 'cnn'
    use_cross_fusion     = True
    flow_type            = 'multi_scale'

    # ===== Save config =====
    save_root = './KITTI_ODOMETRY/sequences'
    save_name = "v2_pcl"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 1


def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_model_from_cfg(cfg, image_size, save_root, save_name, legacy_depth_branches=False):
    """Build GCMLocSave from a config dict (typically checkpoint['config'])."""
    if cfg.get('model_type', 'ablation') == 'original':
        raise ValueError(
            "model_type='original' detected in checkpoint. "
            "For saving GCMLoc from the original CMRNet, use main_single_save.py instead."
        )
    return GCMLocSave(
        image_size=image_size,
        feat_dim=cfg.get('feat_dim', 128),
        heatmap_dim_s1=cfg.get('heatmap_dim_s1', 64),
        topk_points=cfg.get('topk_points', 5000),
        vmamba_output_stage=cfg.get('vmamba_output_stage', 2),
        use_cnn_fallback=cfg.get('use_cnn_fallback', False),
        dropout=cfg.get('dropout', 0.0),
        use_reflectance=cfg.get('use_reflectance', False),
        rgb_backbone=cfg.get('rgb_backbone', 'cnn'),
        dinov2_variant=cfg.get('dinov2_variant', 'b'),
        unfreeze_dinov2_blocks=cfg.get('unfreeze_dinov2_blocks', 0),
        depth_backbone=cfg.get('depth_backbone', 'cnn'),
        use_cross_fusion=cfg.get('use_cross_fusion', False),
        flow_type=cfg.get('flow_type', 'correlation'),
        legacy_depth_branches=legacy_depth_branches,
        save_root=save_root,
        save_name=save_name
    )


MODULE_NAMES = ['rgb_backbone', 'depth_backbone_gt', 'depth_backbone_init',
                'heatmap_decoder_s1', 'flow_s1', 'pose_reg_s1', 'topk_select']


@ex.capture
def run_save(model, pcl, info, u, v, rgb_img, refl_img, flagt, depth,
             target_transl, target_rot, loss_fn, camera_model, point_clouds, loss):
    """Run Stage-1 forward with per-module timing; no point cloud saving."""
    model.eval()

    with torch.no_grad():
        transl_err, rot_err, t0, r0, module_times = model(
            rgb_img, refl_img, depth, flagt, u, v, pcl, info)

    if loss != 'points_distance':
        total_loss = loss_fn(target_transl, target_rot, transl_err, rot_err)
    else:
        total_loss = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)

    total_trasl_error = torch.tensor(0.0, device=target_rot.device)
    total_rot_error   = quaternion_distance(target_rot, rot_err, target_rot.device)
    total_rot_error   = total_rot_error * 180. / math.pi
    for j in range(rgb_img.shape[0]):
        total_trasl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.

    return total_loss.item(), total_trasl_error.item(), total_rot_error.sum().item(), module_times


@ex.automain
def main(_config, _run, seed):
    global EPOCH
    print("=" * 60)
    print("GCMLoc — Save GCMLoc Point Clouds")
    print("=" * 60)

    my_config = dict(_config)

    if my_config['test_sequence'] is None:
        raise TypeError('test_sequence cannot be None')
    if datasetType == 0:
        my_config['test_sequence'] = f"{my_config['test_sequence']:02d}"
    print(f"Test Sequence: {my_config['test_sequence']}")
    dataset_class = DatasetVisibilityKittiSingle

    img_shape  = (384, 1280) if datasetType == 0 else (640, 960)
    maps_folder = my_config.get('maps_folder', 'local_maps')

    # ==================== Dataset ====================
    # Run on the test split (same as validation in train_mapping.py)
    dataset_val = dataset_class(
        my_config['data_folder'], max_r=my_config['max_r'], max_t=my_config['max_t'],
        split='test', use_reflectance=my_config['use_reflectance'],
        maps_folder=maps_folder, test_sequence=my_config['test_sequence'],
    )

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x): return _init_fn(x, seed)

    SaveImgLoader = torch.utils.data.DataLoader(
        dataset=dataset_val, shuffle=False,
        batch_size=_config['batch_size'],
        num_workers=_config['num_worker'],
        worker_init_fn=init_fn,
        collate_fn=merge_inputs,
        drop_last=False, pin_memory=True,
    )

    # ==================== Build Model ====================
    print("\nBuilding model...")

    # Priority: use config embedded in checkpoint so architecture always matches.
    if _config['weights'] is not None:
        print(f"Loading checkpoint: {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location='cpu')

        if 'config' in checkpoint:
            ckpt_cfg = checkpoint['config']
            print("[INFO] Using architecture config from checkpoint.")
        else:
            # Older checkpoint without embedded config — fall back to CLI config
            ckpt_cfg = my_config
            print("[WARNING] Checkpoint has no 'config' key; using CLI config for architecture.")

        # Auto-detect old checkpoints where branch_lhmap used the same architecture
        # as branch_init (_LconvBranch, which includes level0).
        # New _LlconvBranch has no level0, so level1 takes in_ch directly.
        state_dict = checkpoint['state_dict']
        legacy = 'depth_backbone.branch_lhmap.level0.0.0.weight' in state_dict
        if legacy:
            print("[INFO] Legacy checkpoint detected (branch_lhmap has level0) — using legacy_depth_branches=True.")

        model = build_model_from_cfg(ckpt_cfg, img_shape, _config['save_root'], _config['save_name'],
                                     legacy_depth_branches=legacy)

        # strict=False: Stage-2 keys in the checkpoint are silently ignored
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"  Loaded state_dict — missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if missing:
            print(f"  [WARNING] Missing keys (not loaded): {missing}")
    else:
        print("[WARNING] No weights specified — results will be random.")
        model = build_model_from_cfg(my_config, img_shape, _config['save_root'], _config['save_name'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    model = model.to(device)
    model.eval()

    # ==================== Loss Function ====================
    if _config['loss'] == 'simple':
        loss_fn = ProposedLoss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'geometric':
        loss_fn = GeometricLoss().to(device)
    elif _config['loss'] == 'points_distance':
        loss_fn = DistancePoints3D()
    elif _config['loss'] == 'L1':
        loss_fn = L1Loss(_config['rescale_transl'], _config['rescale_rot'])
    else:
        raise ValueError(f"Unknown loss: {_config['loss']}")

    # ==================== Save Loop ====================
    print(f"\nSaving GCMLoc point clouds to: {_config['save_root']}")
    print(f"Top-K points: {model.topk_points}")
    print(f"Processing {len(dataset_val)} samples...\n")

    total_test_loss = 0.
    total_test_t    = 0.
    total_test_r    = 0.
    local_loss      = 0.
    start_time      = time.time()
    # Per-sample module times accumulator: list of {module: seconds}
    all_module_times = []
    prep_times  = []   # per-sample pre-process seconds
    infer_times = []   # per-sample inference seconds

    for batch_idx, sample in enumerate(SaveImgLoader):
        batch_start = time.time()

        # ===== Timed Pre-process (depth projection + tensor prep) =====
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_prep = time.perf_counter()

        lidar_input = []
        rgb_input   = []
        flagt_input = []
        depth_input = []
        ltolu_input = []
        ltolv_input = []
        pcl_input   = []
        info_input  = []

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
            uv, uvt, depth, dt, py, px, _ = cam_model.project_pytorch(
                pc_rotated, pcl, real_shape, reflectance)

            uv  = uv.t().int()
            uvt = uvt.t().int()

            depth_img  = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float) + 1000.
            depth_imgt = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float) + 1000.
            depth_imgt = visibility.depth_image(
                uvt.contiguous(), dt, depth_imgt, uvt.shape[0], real_shape[1], real_shape[0])
            depth_img = visibility.depth_image(
                uv.contiguous(), depth, depth_img, uv.shape[0], real_shape[1], real_shape[0])
            depth_imgt[depth_imgt == 1000.] = 0.
            depth_img[depth_img == 1000.]   = 0.

            uv  = uv.long()
            uvt = uvt.long()
            indexes = depth_img[uv[:, 1], uv[:, 0]] == depth
            flagt   = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
            flagt[uvt[indexes, 1], uvt[indexes, 0]] = 1
            depth_imgt = flagt * depth_imgt

            # 3-D point cloud in image grid (x/y/z per pixel)
            lidar_x = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
            lidar_y = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
            lidar_z = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
            lidar_x[uvt[indexes, 1], uvt[indexes, 0]] = dt[indexes]
            lidar_y[uvt[indexes, 1], uvt[indexes, 0]] = py[indexes]
            lidar_z[uvt[indexes, 1], uvt[indexes, 0]] = px[indexes]

            lidarToLidaru = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
            lidarToLidarv = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
            lidarToLidaru[uvt[indexes, 1], uvt[indexes, 0]] = uv[indexes, 1].float()
            lidarToLidarv[uvt[indexes, 1], uvt[indexes, 0]] = uv[indexes, 0].float()

            depth_img  /= _config['max_depth']
            depth_imgt /= _config['max_depth']

            depth_img      = depth_img.unsqueeze(0)
            depth_imgt     = depth_imgt.unsqueeze(0)
            flagt          = flagt.unsqueeze(0)
            lidarToLidaru  = lidarToLidaru.unsqueeze(0)
            lidarToLidarv  = lidarToLidarv.unsqueeze(0)
            pcl_grid       = torch.stack((lidar_x, lidar_y, lidar_z))

            rgb = sample['rgb'][idx].cuda()
            shape_pad    = [0, 0, 0, 0]
            shape_pad[3] = img_shape[0] - rgb.shape[1]
            shape_pad[1] = img_shape[1] - rgb.shape[2]

            rgb           = F.pad(rgb,           shape_pad)
            depth_img     = F.pad(depth_img,     shape_pad)
            depth_imgt    = F.pad(depth_imgt,    shape_pad)
            flagt         = F.pad(flagt,         shape_pad)
            pcl_grid      = F.pad(pcl_grid,      shape_pad)
            lidarToLidaru = F.pad(lidarToLidaru, shape_pad)
            lidarToLidarv = F.pad(lidarToLidarv, shape_pad)

            if datasetType == 0:
                info = torch.tensor([int(sample['idx'][idx]), int(sample['rgb_name'][idx])])
            else:
                info = [sample['idx'][idx], sample['sub_dir'][idx], sample['rgb_name'][idx]]

            rgb_input.append(rgb)
            lidar_input.append(depth_img)
            flagt_input.append(flagt)
            depth_input.append(depth_imgt)
            ltolu_input.append(lidarToLidaru)
            ltolv_input.append(lidarToLidarv)
            pcl_input.append(pcl_grid)
            info_input.append(info)

        lidar_input = torch.stack(lidar_input)
        rgb_input   = torch.stack(rgb_input)
        flagt_input = torch.stack(flagt_input)
        depth_input = torch.stack(depth_input)
        ltolu_input = torch.stack(ltolu_input)
        ltolv_input = torch.stack(ltolv_input)
        pcl_input   = torch.stack(pcl_input)

        batch_sz = rgb_input.shape[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prep_times.append((time.perf_counter() - t_prep) / batch_sz)

        # ===== Timed Inference =====
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_infer = time.perf_counter()

        loss, trasl_e, rot_e, mod_times = run_save(
            model, pcl_input, info_input,
            ltolu_input, ltolv_input,
            rgb_input, lidar_input,
            flagt_input, depth_input,
            sample['tr_error'], sample['rot_error'],
            loss_fn, dataset_val.model, sample['point_cloud'],
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_times.append((time.perf_counter() - t_infer) / batch_sz)

        # mod_times is per batch; divide by batch size for per-sample ms
        all_module_times.append({m: mod_times.get(m, 0.0) * 1000.0 / batch_sz
                                  for m in MODULE_NAMES})

        if loss != loss:
            raise ValueError("Loss is NaN")

        total_test_t    += trasl_e
        total_test_r    += rot_e
        local_loss      += loss
        total_test_loss += loss * len(sample['rgb'])

        if (batch_idx + 1) % 10 == 0:
            ts = datetime.now().strftime('%H:%M:%S')
            n  = batch_idx + 1
            print(
                f'[{ts}] [{n:4d}/{len(SaveImgLoader)}]  '
                f'loss={local_loss / min(n, 10):.6f}  '
                f't_err={total_test_t / (n * _config["batch_size"]):.4f} cm  '
                f'r_err={total_test_r / (n * _config["batch_size"]):.4f}°  '
                f'{time.time() - batch_start:.1f}s/batch'
            )

    ts = datetime.now().strftime('%H:%M:%S')
    n  = len(dataset_val)
    print("\n" + "=" * 60)
    print(f"[{ts}] Inference complete (no files saved)!")
    print(f"  loss={total_test_loss / n:.6f}  "
          f"t_err={total_test_t / n:.4f} cm  "
          f"r_err={total_test_r / n:.4f}°")
    print(f"  time={( time.time() - start_time) / 60:.1f} min")
    print("=" * 60)

    # ===== Per-Module Timing =====
    print("\n===== Per-Module Timing (mean over all samples) =====")
    module_mean_ms = {m: float(np.mean([s[m] for s in all_module_times]))
                      for m in MODULE_NAMES}
    total_ms = sum(module_mean_ms.values())
    for m in MODULE_NAMES:
        ms  = module_mean_ms[m]
        pct = 100.0 * ms / total_ms if total_ms > 0 else 0.0
        print(f"  {m:<26s}: {ms:7.2f} ms  ({pct:.1f}%)")
    print(f"  {'total (modules)':<26s}: {total_ms:7.2f} ms")
    total_wall_ms = (time.time() - start_time) * 1000.0 / n
    print(f"  {'total (wall, per sample)':<26s}: {total_wall_ms:7.2f} ms")
    print("=" * 60)

    # ===== Record one structured timing row (for the LaTeX table) =====
    mean_prep_ms  = float(np.mean(prep_times)) * 1000.0 if prep_times else 0.0
    mean_infer_ms = float(np.mean(infer_times)) * 1000.0 if infer_times else 0.0
    mean_total_ms = mean_prep_ms + mean_infer_ms
    print(f"  {'preprocess (per sample)':<26s}: {mean_prep_ms:7.2f} ms")
    print(f"  {'inference (per sample)':<26s}: {mean_infer_ms:7.2f} ms")
    print(f"  {'total (pre+infer)':<26s}: {mean_total_ms:7.2f} ms")
    print("=" * 60)

    # Map Stage-1 module names -> canonical keys used by the timing table.
    _MODULE_MAP = {
        'rgb_backbone':        'rgb_backbone',
        'depth_backbone_gt':   'depth_backbone',
        'depth_backbone_init': 'depth_backbone',
        'heatmap_decoder_s1':  'cross_modal_fusion',
        'flow_s1':             'gcm_flow',
        'pose_reg_s1':         'pose_regression',
        'topk_select':         'others',
    }
    module_breakdown = {}
    for m, key in _MODULE_MAP.items():
        module_breakdown[key] = module_breakdown.get(key, 0.0) + module_mean_ms.get(m, 0.0)

    method_label = _config.get('method') or 'Stage1-save'
    record_timing(
        method=method_label,
        dataset=_config['dataset'],
        preprocess_ms=mean_prep_ms,
        inference_ms=mean_infer_ms,
        modules=module_breakdown,
        stage='map',
        seq=_config['test_sequence'],
        split='test',
        n_samples=n,
        weights=_config.get('weights'),
        note='stage-1 mapping',
        csv_path=_config.get('timing_csv'),
    )

    return _run.result
