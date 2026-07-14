"""
GCMLoc — Stage-2 Online Localization Training Script.

Trains Stage 2 (online localization) using pre-saved GCMLoc map point clouds
(produced by train_save.py from a Stage-1 mapping checkpoint).

Workflow (3-round coarse-to-fine, matching original paper):
    Round 1: python train_loc.py with test_sequence=0 epochs=300 \\
                 max_r=10 max_t=2 weights='<mapping_ckpt>'
    Round 2: python train_loc.py with test_sequence=0 epochs=300 \\
                 max_r=2  max_t=1 weights='<round1_best.tar>'
    Round 3: python train_loc.py with test_sequence=0 epochs=300 \\
                 max_r=1  max_t=0.5 weights='<round2_best.tar>'

Architecture flags are read from the loaded checkpoint's embedded config.
If the checkpoint has no 'config' key, CLI flags are used as fallback.
"""

import math
import os
import random
import sys
import csv
import time
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mathutils
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import visibility

from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds

from camera_model_localization import CameraModel
from Dataset_kitti_localization import DatasetVisibilityKittiSingle as DatasetKitti
from Dataset_argoverse_localization import DatasetVisibilityKittiSingle as DatasetArgo
from combined_dataset import CombinedDataset
import utils_canonical as UC
from losses import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss
from models.GCMLoc.GCMLoc_localization import GCMLocLocalization
from quaternion_distances import quaternion_distance
from utils import merge_inputs, rotate_back

# 0 → KITTI Odometry   1 → Argoverse Tracking   2 → Mixed (Mixed training with both datasets in canonical space; set datasetType=2 and provide both kitti_data_folder and argo_data_folder in config)
datasetType = 2

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

SETTINGS.DISCOVER_DEPENDENCIES = "none"
SETTINGS.DISCOVER_SOURCES = "none"
ex = Experiment("GCMLoc-loc")
ex.captured_out_filter = apply_backspaces_and_linefeeds


class Logger(object):
    def __init__(self, filename, stream=None):
        self.terminal = stream if stream is not None else sys.stdout
        self.log = open(filename, 'a', encoding='utf-8', buffering=1)

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


@ex.config
def config():
    savemodel       = './checkpoints/'
    # dataset tag used in checkpoint dir name
    # set to 'argo' when training on Argoverse (also change datasetType above)
    dataset         = 'kitti'
    data_folder     = './KITTI_ODOMETRY/sequences'
    use_reflectance = False
    test_sequence   = 0
    occlusion_kernel    = 5
    occlusion_threshold = 3
    epochs          = 300
    BASE_LEARNING_RATE  = 1e-4
    loss            = 'simple'
    max_t           = 2.
    max_r           = 10.
    batch_size      = 8
    num_worker      = 3
    optimizer       = 'adam'
    resume          = None
    weights         = None   # Round 1: mapping ckpt; Round 2+: previous round best
    rescale_rot     = 1
    rescale_transl  = 1
    dropout         = 0.0
    max_depth       = 100.
    # KITTI localization: 'v2_pcl'  |  Argoverse localization: 'v2_pcl' (npy)
    # set to '' to use raw single-sweep PLY files from Argoverse
    maps_folder     = 'v2_pcl'
    # Argoverse only: root dir where npy maps are stored (leave '' for legacy layout)
    maps_root       = ''

    # ===== Canonical normalization / mixed (datasetType==2) training =====
    use_canonical      = False   # warp all RGB to one virtual camera, calib→canonical
    canon_preset       = 'A'     # 'A'=paper 768x384 | 'B'=960x480 wide
    kitti_data_folder  = ''      # mixed: KITTI root  (split by test_sequence=00)
    argo_data_folder   = ''      # mixed: Argoverse root (train1-3 / train4)
    mixed_weight_kitti = 1.0     # sampling weight (P_kitti = w_k/(w_k+w_a))
    mixed_weight_argo  = 2.0     # default → KITTI ~33%

    # ===== GCMLoc architecture (overridden by checkpoint config when weights provided) =====
    feat_dim             = 128
    heatmap_dim_s2       = 128
    vmamba_output_stage  = 2
    use_cnn_fallback     = False
    dinov2_lr_scale      = 0.01   # lr multiplier for unfrozen DINOv2 body blocks

    # ===== Architecture flags (overridden by checkpoint config when weights provided) =====
    # rgb_backbone: 'cnn', 'dinov2b' (ViT-B/14), 'dinov2s' (ViT-S/14), 'dinov2' (uses dinov2_variant)
    rgb_backbone         = 'dinov2b'
    dinov2_variant       = 'b'           # 'b'=ViT-B/14, 's'=ViT-S/14; used when rgb_backbone='dinov2'
    unfreeze_dinov2_blocks = 4
    depth_backbone       = 'cnn'
    flow_type            = 'multi_scale'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 1


def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_model_from_cfg(cfg, image_size, legacy_depth_branches=False):
    """Build GCMLocLocalization from a config dict (typically checkpoint['config'])."""
    return GCMLocLocalization(
        image_size=image_size,
        feat_dim=cfg.get('feat_dim', 128),
        heatmap_dim_s2=cfg.get('heatmap_dim_s2', 128),
        vmamba_output_stage=cfg.get('vmamba_output_stage', 2),
        use_cnn_fallback=cfg.get('use_cnn_fallback', False),
        dropout=cfg.get('dropout', 0.0),
        use_reflectance=cfg.get('use_reflectance', False),
        rgb_backbone=cfg.get('rgb_backbone', 'cnn'),
        dinov2_variant=cfg.get('dinov2_variant', 'b'),
        unfreeze_dinov2_blocks=cfg.get('unfreeze_dinov2_blocks', 0),
        depth_backbone=cfg.get('depth_backbone', 'cnn'),
        flow_type=cfg.get('flow_type', 'correlation'),
        legacy_depth_branches=legacy_depth_branches,
    )


@ex.capture
def build_optimizer(model, _config):
    """
    Build Adam optimizer with parameter groups.

    If DINOv2 blocks are unfrozen, their body params get a lower lr
    (dinov2_lr_scale) to prevent destroying pre-trained features.
    """
    base_lr         = _config['BASE_LEARNING_RATE']
    dinov2_lr_scale = _config.get('dinov2_lr_scale', 0.01)

    dinov2_body_params = []   # unfrozen DINOv2 transformer blocks
    dinov2_proj_params = []   # DINOv2 projection head (always trainable)
    other_params       = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'rgb_backbone.dino' in name:
            dinov2_body_params.append(param)
        elif 'rgb_backbone.projection' in name:
            dinov2_proj_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': dinov2_body_params, 'lr': base_lr * dinov2_lr_scale, 'name': 'dinov2_body'},
        {'params': dinov2_proj_params, 'lr': base_lr,                   'name': 'dinov2_proj'},
        {'params': other_params,       'lr': base_lr,                   'name': 'other'},
    ]
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    optimizer = optim.Adam(param_groups, weight_decay=1e-5)
    for g in param_groups:
        n = sum(p.numel() for p in g['params'])
        print(f"  {g['name']}: {n:,} params, lr={g['lr']:.2e}")

    return optimizer


@ex.capture
def train(model, optimizer, rgb_img, refl_img, target_transl, target_rot,
          loss_fn, point_clouds, loss):
    """Single training step."""
    optimizer.zero_grad()

    transl_err, rot_err, w_x, w_q = model(rgb_img, refl_img)

    if loss != 'points_distance':
        total_loss = loss_fn(target_transl, target_rot, transl_err, rot_err)
    else:
        total_loss = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    with torch.no_grad():
        t_err_cm = sum(
            torch.norm(target_transl[j] - transl_err[j]).item() * 100.
            for j in range(rgb_img.shape[0])
        ) / rgb_img.shape[0]
        r_err_deg = (
            quaternion_distance(target_rot, rot_err, rot_err.device).mean().item()
            * 180. / math.pi
        )

    return total_loss.item(), t_err_cm, r_err_deg


@ex.capture
def test(model, rgb_img, refl_img, target_transl, target_rot,
         loss_fn, camera_model, point_clouds, loss):
    """Single evaluation step."""
    with torch.no_grad():
        transl_err, rot_err, w_x, w_q = model(rgb_img, refl_img)

    if loss != 'points_distance':
        total_loss = loss_fn(target_transl, target_rot, transl_err, rot_err)
    else:
        total_loss = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)

    total_trasl_error = torch.tensor(0.0, device=target_rot.device)
    total_rot_error   = quaternion_distance(target_rot, rot_err, target_rot.device)
    total_rot_error   = total_rot_error * 180. / math.pi
    for j in range(rgb_img.shape[0]):
        total_trasl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.

    return total_loss.item(), total_trasl_error.item(), total_rot_error.sum().item()


@ex.automain
def main(_config, _run, seed):
    global EPOCH, datasetType

    my_config = dict(_config)

    # the `dataset` config string drives the selection (CLI-friendly):
    #   'kitti'→0, 'argo'→1, 'mixed'→2 (canonical KITTI+Argo)
    datasetType = {'kitti': 0, 'argo': 1, 'mixed': 2}.get(
        my_config.get('dataset'), datasetType)

    if my_config['test_sequence'] is None:
        raise TypeError('test_sequence cannot be None')

    if datasetType == 0:
        my_config['test_sequence'] = f"{my_config['test_sequence']:02d}"
    else:
        my_config['test_sequence'] = str(my_config['test_sequence'])

    # ===== Directory & Logging =====
    my_config['savemodel'] = os.path.join(my_config['savemodel'], my_config['dataset'])

    if my_config['resume'] is not None:
        save_dir  = os.path.dirname(my_config['resume'])
        timestamp = os.path.basename(save_dir)
        print(f"Resuming training from {save_dir}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rgb_tag   = my_config.get('rgb_backbone', 'cnn')
        depth_tag = my_config.get('depth_backbone', 'cnn')
        flow_tag  = my_config.get('flow_type', 'correlation')
        ds_tag    = ('mixed' if datasetType == 2 else
                     'argo' if datasetType == 1 else my_config['test_sequence'])
        save_dir  = os.path.join(
            my_config['savemodel'],
            ds_tag,
            f"{timestamp}_rgb-{rgb_tag}_depth-{depth_tag}_flow-{flow_tag}_loc",
        )
        os.makedirs(save_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(save_dir, 'train_console.log'), sys.stdout)
    sys.stderr = Logger(os.path.join(save_dir, 'train_error.log'),   sys.stderr)

    print("=" * 60)
    print("GCMLoc — Ablation Online Localization Training")
    print("=" * 60)
    print(f"Config: {_config}")
    print(f"Run dir: {timestamp}")
    print(f"Test Sequence: {my_config['test_sequence']}")

    log_path  = os.path.join(save_dir, 'train_log.csv')
    plot_path = os.path.join(save_dir, 'loss_curve.png')

    if not my_config['resume']:
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'val_loss',
                'val_t_err_cm', 'val_r_err_deg', 'lr', 'time_s', 'timestamp',
            ])

    # ===== Dataset =====
    maps_folder = my_config.get('maps_folder', 'v2_pcl')
    uc = my_config.get('use_canonical', False)
    UC.set_preset(my_config.get('canon_preset', 'A'))
    train_sampler = None

    if datasetType == 2:
        # mixed KITTI + Argoverse in canonical space
        img_shape = UC.CANON_SHAPE
        common = dict(max_r=my_config['max_r'], max_t=my_config['max_t'],
                      use_reflectance=my_config['use_reflectance'],
                      maps_folder=maps_folder, use_canonical=True)
        kitti_tr = DatasetKitti(my_config['kitti_data_folder'], split='train',
                                test_sequence='00', **common)
        argo_tr  = DatasetArgo(my_config['argo_data_folder'], split='train',
                               half_res=False, **common)
        kitti_va = DatasetKitti(my_config['kitti_data_folder'], split='test',
                                test_sequence='00', **common)
        argo_va  = DatasetArgo(my_config['argo_data_folder'], split='test',
                               half_res=False, **common)
        weights     = [my_config['mixed_weight_kitti'], my_config['mixed_weight_argo']]
        dataset     = CombinedDataset([kitti_tr, argo_tr], weights=weights)
        dataset_val = CombinedDataset([kitti_va, argo_va], weights=weights)
        train_sampler = dataset.make_sampler()
        print(f"\n[Dataset] MIXED canonical  img_shape={img_shape}  "
              f"train={len(dataset)} (kitti {len(kitti_tr)}, argo {len(argo_tr)})  "
              f"weights={weights}")
    else:
        if datasetType == 0:
            dataset_class = DatasetKitti
            img_shape = UC.CANON_SHAPE if uc else (384, 1280)
        else:
            dataset_class = DatasetArgo
            img_shape = UC.CANON_SHAPE if uc else (640, 960)

        print(f"\n[Dataset] type={'Argoverse' if datasetType == 1 else 'KITTI'}"
              f"{' canonical' if uc else ''}  img_shape={img_shape}")
        extra_kwargs = {'use_canonical': uc}
        if datasetType == 1:
            extra_kwargs['half_res'] = (not uc)   # canonical warp replaces half_res
            if my_config.get('maps_root', ''):
                extra_kwargs['maps_root'] = my_config['maps_root']

        dataset     = dataset_class(
            my_config['data_folder'], max_r=my_config['max_r'], max_t=my_config['max_t'],
            split='train', use_reflectance=my_config['use_reflectance'],
            maps_folder=maps_folder, test_sequence=my_config['test_sequence'],
            **extra_kwargs,
        )
        dataset_val = dataset_class(
            my_config['data_folder'], max_r=my_config['max_r'], max_t=my_config['max_t'],
            split='test', use_reflectance=my_config['use_reflectance'],
            maps_folder=maps_folder, test_sequence=my_config['test_sequence'],
            **extra_kwargs,
        )

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x): return _init_fn(x, seed)

    TrainImgLoader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=(train_sampler is None), sampler=train_sampler,
        batch_size=_config['batch_size'],
        num_workers=_config['num_worker'], worker_init_fn=init_fn,
        collate_fn=merge_inputs, drop_last=False, pin_memory=True,
    )
    TestImgLoader = torch.utils.data.DataLoader(
        dataset=dataset_val, shuffle=False, batch_size=_config['batch_size'],
        num_workers=_config['num_worker'], worker_init_fn=init_fn,
        collate_fn=merge_inputs, drop_last=False, pin_memory=True,
    )

    # ===== Build Model =====
    print("\nBuilding model...")

    if _config['weights'] is not None:
        print(f"Loading weights from: {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location=device, weights_only=False)

        # Architecture keys that can be explicitly overridden via CLI.
        # Must match the Sacred config() defaults exactly so we can detect
        # whether the user explicitly passed a value on the CLI.
        _arch_defaults = {
            'rgb_backbone': 'dinov2b', 'dinov2_variant': 'b', 'unfreeze_dinov2_blocks': 4,
            'depth_backbone': 'cnn', 'flow_type': 'multi_scale',
            'feat_dim': 128, 'heatmap_dim_s2': 128,
            'vmamba_output_stage': 2, 'use_cnn_fallback': False,
            'use_reflectance': False, 'dropout': 0.0,
        }

        if 'config' in checkpoint:
            ckpt_cfg = dict(checkpoint['config'])
            print("[INFO] Using architecture config from checkpoint.")
        else:
            ckpt_cfg = dict(my_config)
            print("[WARNING] Checkpoint has no 'config' key; using CLI config for architecture.")

        # If a CLI flag differs from its Sacred default, the user explicitly set it
        # → override ckpt_cfg so the intended architecture is used.
        for k, default_val in _arch_defaults.items():
            if my_config.get(k) != default_val:
                print(f"[CLI OVERRIDE] {k}: {ckpt_cfg.get(k, '?')} → {my_config[k]}")
                ckpt_cfg[k] = my_config[k]

        # Sync final architecture into my_config so saved checkpoints record it correctly.
        for k in _arch_defaults:
            if k in ckpt_cfg:
                my_config[k] = ckpt_cfg[k]

        print(f"[INFO] Architecture: rgb={ckpt_cfg.get('rgb_backbone')} "
              f"depth={ckpt_cfg.get('depth_backbone')} "
              f"flow={ckpt_cfg.get('flow_type')} "
              f"dinov2_unfreeze={ckpt_cfg.get('unfreeze_dinov2_blocks')}")

        # Detect old checkpoints where branch_lhmap had the same arch as branch_init
        state_dict = checkpoint['state_dict']
        legacy = 'depth_backbone.branch_lhmap.level0.0.0.weight' in state_dict
        if legacy:
            print("[INFO] Legacy depth branches detected — using legacy_depth_branches=True.")

        model = build_model_from_cfg(ckpt_cfg, img_shape, legacy_depth_branches=legacy)

        # strict=False:
        #   Mapping ckpt → Stage 1 keys (heatmap_decoder_s1, flow_s1, pose_reg_s1, fusion)
        #                   are unexpected (ignored).
        #   Loc ckpt     → All keys should match; if heatmap_head_s2 was HeatmapHead,
        #                   those keys are unexpected and heatmap_head_s2 starts from scratch.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")
        if missing:
            print(f"  [WARNING] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    else:
        print("[WARNING] No weights specified — model starts from random init.")
        model = build_model_from_cfg(my_config, img_shape)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    model = model.to(device)

    # ===== Optimizer & Scheduler =====
    print("\nOptimizer parameter groups:")
    optimizer = build_optimizer(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_config['epochs'], eta_min=1e-6,
    )

    # ===== Loss Function =====
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

    # ===== Resume =====
    starting_epoch = 0
    if _config['resume']:
        print(f"\nResuming from {_config['resume']}")
        ckpt_resume = torch.load(_config['resume'], map_location=device, weights_only=False)
        model.load_state_dict(ckpt_resume['state_dict'])
        optimizer.load_state_dict(ckpt_resume['optimizer'])
        starting_epoch = ckpt_resume['epoch'] + 1

    # ===== Training Loop =====
    BEST_VAL_LOSS          = float('inf')
    old_save_filename      = None
    epochs_no_improvement  = 0
    PATIENCE               = 30
    start_full_time        = time.time()

    log_writer_file = open(log_path, 'a', newline='')
    log_writer      = csv.writer(log_writer_file)

    for epoch in range(starting_epoch, _config['epochs'] + 1):
        EPOCH = epoch
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{_config['epochs']}")
        print(f"{'='*60}")

        epoch_start = time.time()
        total_train_loss = 0.
        local_loss       = 0.
        local_t_err      = 0.
        local_r_err      = 0.
        LOG_INTERVAL     = 50

        model.train()

        # ─── Training loop ───
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()

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

            loss, t_err, r_err = train(
                model, optimizer, rgb_input, lidar_input,
                sample['tr_error'], sample['rot_error'],
                loss_fn, sample['point_cloud'],
            )

            if loss != loss:
                raise ValueError("Loss is NaN")

            local_loss  += loss
            local_t_err += t_err
            local_r_err += r_err

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                ts = datetime.now().strftime('%H:%M:%S')
                spd = (time.time() - start_time) / lidar_input.shape[0]
                print(
                    f'[{ts}] Ep{epoch} [{batch_idx+1:4d}/{len(TrainImgLoader)}] '
                    f'loss={local_loss / LOG_INTERVAL:.6f}  '
                    f't_err={local_t_err / LOG_INTERVAL:.4f}cm  '
                    f'r_err={local_r_err / LOG_INTERVAL:.4f}°  '
                    f'{spd:.3f}s/sample'
                )
                _run.log_scalar("Loss", local_loss / LOG_INTERVAL,
                                epoch * len(TrainImgLoader) + batch_idx)
                local_loss  = 0.
                local_t_err = 0.
                local_r_err = 0.

            total_train_loss += loss * len(sample['rgb'])

        total_time = time.time() - epoch_start
        ts = datetime.now().strftime('%H:%M:%S')
        print(f'[{ts}] Ep{epoch} | train_loss={total_train_loss / len(dataset):.5f} | time={total_time:.1f}s')
        _run.log_scalar("Total training loss", total_train_loss / len(dataset), epoch)

        scheduler.step()

        # ─── Validation ───
        _csv_val_loss = ''
        _csv_val_t    = ''
        _csv_val_r    = ''

        if epoch % 2 == 0 or epoch > 195:
            total_test_loss = 0.
            total_test_t    = 0.
            total_test_r    = 0.
            model.eval()

            for batch_idx, sample in enumerate(TestImgLoader):
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

                loss, trasl_e, rot_e = test(
                    model, rgb_input, lidar_input,
                    sample['tr_error'], sample['rot_error'],
                    loss_fn, dataset_val.model, sample['point_cloud'],
                )

                if loss != loss:
                    raise ValueError("Loss is NaN")

                total_test_t    += trasl_e
                total_test_r    += rot_e
                total_test_loss += loss * len(sample['rgb'])

            val_loss = total_test_loss / len(dataset_val)
            val_t    = total_test_t    / len(dataset_val)
            val_r    = total_test_r    / len(dataset_val)

            print("------------------------------------")
            ts = datetime.now().strftime('%H:%M:%S')
            print(f'[{ts}] [VAL] Ep{epoch}')
            print(f'Val loss        = {val_loss:.6f}')
            print(f'Translation err = {val_t:.4f} cm')
            print(f'Rotation err    = {val_r:.4f} °')
            print("------------------------------------")

            _run.log_scalar("Val_Loss",    val_loss, epoch)
            _run.log_scalar("Val_t_error", val_t,    epoch)
            _run.log_scalar("Val_r_error", val_r,    epoch)

            _csv_val_loss = f'{val_loss:.6f}'
            _csv_val_t    = f'{val_t:.4f}'
            _csv_val_r    = f'{val_r:.4f}'

            if val_loss < BEST_VAL_LOSS:
                BEST_VAL_LOSS          = val_loss
                epochs_no_improvement  = 0

                if _config['rescale_transl'] > 0:
                    _run.result = val_t
                else:
                    _run.result = val_r

                savefilename = (
                    f'{save_dir}/checkpoint'
                    f'_r{_config["max_r"]:.2f}'
                    f'_t{_config["max_t"]:.2f}'
                    f'_e{epoch}_{val_loss:.5f}_loc_ablation.tar'
                )
                torch.save({
                    'config':      _config,
                    'epoch':       epoch,
                    'state_dict':  model.state_dict(),
                    'optimizer':   optimizer.state_dict(),
                    'train_loss':  total_train_loss / len(dataset),
                    'test_loss':   val_loss,
                }, savefilename)
                print(f'[Saved] {savefilename}')

                if old_save_filename is not None and os.path.exists(old_save_filename):
                    os.remove(old_save_filename)
                old_save_filename = savefilename
            else:
                epochs_no_improvement += 1
                print(f"[EarlyStop] No improvement for {epochs_no_improvement}/{PATIENCE} val steps.")

        # ─── Save latest (for resume) ───
        latest_filename = f'{save_dir}/checkpoint_latest_loc_ablation.tar'
        torch.save({
            'config':     _config,
            'epoch':      epoch,
            'state_dict': model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'train_loss': total_train_loss / len(dataset),
            'test_loss':  BEST_VAL_LOSS,
        }, latest_filename)

        # ─── CSV log ───
        cur_lr = optimizer.param_groups[-1]['lr']
        log_writer.writerow([
            epoch,
            f'{total_train_loss / len(dataset):.6f}',
            _csv_val_loss, _csv_val_t, _csv_val_r,
            f'{cur_lr:.2e}',
            f'{total_time:.1f}',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        ])
        log_writer_file.flush()

        # ─── Loss curve plot ───
        try:
            df = pd.read_csv(log_path)
            # Keep only the latest row per epoch (handles resume duplicates)
            df = df.drop_duplicates(subset=['epoch'], keep='last')

            plt.figure(figsize=(10, 8))

            plt.subplot(2, 1, 1)
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
            val_df = df.dropna(subset=['val_loss'])
            if not val_df.empty:
                plt.plot(val_df['epoch'], val_df['val_loss'],
                         label='Val Loss', marker='o', color='orange')
            plt.title('Training & Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()

            plt.subplot(2, 1, 2)
            if not val_df.empty:
                plt.plot(val_df['epoch'], val_df['val_t_err_cm'],
                         label='Val T-Error (cm)', marker='s', color='green')
                plt.plot(val_df['epoch'], val_df['val_r_err_deg'],
                         label='Val R-Error (deg)', marker='^', color='red')
            plt.title('Validation Translation & Rotation Errors')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        except Exception as e:
            print(f"[Warning] Failed to plot loss curve: {e}")

        # ─── Early stopping ───
        if epochs_no_improvement >= PATIENCE:
            print(f"\n[EarlyStop] Triggered after {PATIENCE} val steps without improvement.")
            break

    print(f'\nFull training time = {(time.time() - start_full_time) / 3600:.2f} HR')
    return _run.result
