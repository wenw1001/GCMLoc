"""
GCMLoc — Offline Mapping Training (Stage 1 + Stage 2, map-feature learning).

This script trains the *final* GCMLoc mapping network. The architecture is
fixed (see ARCH below); only the dataset and the training hyper-parameters are
configurable from the command line.

Architecture (fixed)
--------------------
    RGB backbone    : DINOv2 ViT-B/14 (GCMLoc-B) or ViT-S/14 (GCMLoc-S), the last
                      4 transformer blocks are unfrozen and trained with lr x 0.01
    Depth backbone  : three-branch CNN (branch_gt / branch_init / branch_lhmap)
    Stage 1         : cross-modal semantic-geometric fusion (depth x RGB)
                      -> Stage-1 heatmap decoder -> top-K (5000) point selection
    Flow matcher    : GCM-Flow multi-scale matcher (Stage 1 and Stage 2)
    Pose heads      : two pose regressors, loss = 0.6 * stage2 + 0.4 * stage1
    Optimiser       : Adam (weight_decay 5e-6), MultiStepLR [20, 50, 70] gamma 0.5

Datasets (same model, different pre-processing / image size)
-------------------------------------------------------------
    dataset=kitti : Dataset_kitti_mapping, images padded to (384, 1280),
                    local maps = <seq>/<maps_folder>/*.h5, `test_sequence` is the
                    held-out validation sequence
    dataset=argo  : Dataset_argoverse_mapping, half-resolution images (640, 960),
                    global map = <log>/map.h5 (cropped on the fly), train1-3 for
                    training / train4 for validation
    dataset=mixed : KITTI + Argoverse warped to the canonical camera
                    (canon_preset=A, 768x384); CombinedDataset samples the two
                    sources with weights mixed_weight_kitti : mixed_weight_argo

Outputs: <savemodel>/<dataset>/<seq|argo|mixed>/<timestamp>_GCMLoc-<B|S>_mapping/
    checkpoint_r{max_r}_t{max_t}_e{epoch}_{val_loss}_mapping.tar   best model
    checkpoint_latest_mapping.tar                                   for --resume
    checkpoint_..._LAST_mapping.tar                                 last epoch
    train_log.csv, loss_curve.png, train_console.log
checkpoint['config'] stores the full training config (plus ARCH); train_save.py /
train_loc.py / evaluate.py rebuild the identical network from the checkpoint.

Examples
--------
# KITTI (sequence 00 held out for validation)
python train_mapping.py with dataset=kitti \
    data_folder=./KITTI_ODOMETRY/sequences test_sequence=0 \
    epochs=120 batch_size=8 max_r=10 max_t=2

# Argoverse
python train_mapping.py with dataset=argo \
    data_folder=./data/argoverse-tracking \
    epochs=120 batch_size=8 max_r=10 max_t=2

# Mixed KITTI + Argoverse (canonical camera A)
python train_mapping.py with dataset=mixed \
    kitti_data_folder=./KITTI_ODOMETRY/sequences \
    argo_data_folder=./data/argoverse-tracking \
    mixed_weight_kitti=1.0 mixed_weight_argo=2.0 \
    epochs=120 batch_size=8 max_r=10 max_t=2

# Lightweight model (GCMLoc-S): add  dinov2_variant=s
"""

import csv
import math
import os
import random
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import mathutils
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import visibility

from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds

from camera_model_mapping import CameraModel
from Dataset_kitti_mapping import DatasetVisibilityKittiSingle as DatasetKitti
from Dataset_argoverse_mapping import DatasetVisibilityKittiSingle as DatasetArgo
from combined_dataset import CombinedDataset
import utils_canonical as UC
from losses import ProposedLoss
from models.GCMLoc import GCMLocMapping
from quaternion_distances import quaternion_distance
from utils import merge_inputs, rotate_back

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

SETTINGS.DISCOVER_DEPENDENCIES = "none"
SETTINGS.DISCOVER_SOURCES = "none"
ex = Experiment("GCMLoc-mapping")
ex.captured_out_filter = apply_backspaces_and_linefeeds


# =====================================================================
# Fixed architecture (see models/GCMLoc/GCMLoc_mapping.py). These values are
# also written into checkpoint['config'] for reference; downstream scripts
# rebuild the network from the checkpoint weights themselves.
# =====================================================================
ARCH = dict(
    feat_dim=128,                   # unified feature dimension C
    heatmap_dim_s1=64,              # Stage-1 heatmap channels
    heatmap_dim_s2=128,             # Stage-2 heatmap channels
    topk_points=5000,               # points kept per frame after Stage 1 (distilled map)
    unfreeze_dinov2_blocks=4,       # last 4 DINOv2 blocks are trainable (lr x dinov2_lr_scale)
)

DATASETS = ('kitti', 'argo', 'mixed')


@ex.config
def config():
    savemodel = './checkpoints/'    # root of the run directories
    dataset = 'kitti'               # 'kitti' | 'argo' | 'mixed'
    data_folder = './KITTI_ODOMETRY/sequences'   # KITTI / Argoverse root (single-dataset training)
    test_sequence = 0               # KITTI held-out validation sequence (int, 0 -> '00'); also used by mixed; ignored by argo
    maps_folder = 'local_maps_0.1'  # KITTI dense local maps '<seq>/<maps_folder>/*.h5' (Argoverse always uses <log>/map.h5)

    # -- model size --
    dinov2_variant = 'b'            # 'b' -> GCMLoc-B (ViT-B/14, paper default) | 's' -> GCMLoc-S (ViT-S/14)

    # -- optimisation --
    epochs = 120                    # number of training epochs
    batch_size = 8
    num_worker = 3                  # DataLoader workers
    BASE_LEARNING_RATE = 1e-4       # Adam learning rate
    dinov2_lr_scale = 0.01          # lr of the unfrozen DINOv2 blocks = base lr * scale
    lr_milestones = [20, 50, 70]    # MultiStepLR milestones (epochs)
    lr_gamma = 0.5                  # MultiStepLR decay factor
    rescale_transl = 1.0            # loss weight, translation term
    rescale_rot = 10.0              # loss weight, rotation term
    resume = None                   # path to checkpoint_latest_mapping.tar to continue a run

    # -- pose perturbation range (training augmentation / validation) --
    max_t = 2.                      # metres
    max_r = 10.                     # degrees
    max_depth = 100.                # depth normalisation constant (points beyond it are dropped)

    # -- canonical camera / mixed training --
    use_canonical = False           # optional for kitti / argo; always on for mixed
    canon_preset = 'A'              # 'A' = 768x384 (paper); must match the downstream scripts
    kitti_data_folder = ''          # mixed only: KITTI root
    argo_data_folder = ''           # mixed only: Argoverse root
    mixed_weight_kitti = 1.0        # mixed only: sampling weight of KITTI
    mixed_weight_argo = 2.0         # mixed only: sampling weight of Argoverse (1:2 -> KITTI ~ 33 %)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 1


def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Logger(object):
    """Tee stdout / stderr into a file inside the run directory."""

    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, 'a', buffering=1)

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# =====================================================================
# Model / optimiser
# =====================================================================

def build_mapping_model(image_size, dinov2_variant):
    """GCMLoc mapping network (Stage 1 + Stage 2) with the fixed ARCH."""
    return GCMLocMapping(image_size=image_size, dinov2_variant=dinov2_variant, **ARCH)


def build_optimizer(model, base_lr, dinov2_lr_scale):
    """Adam with a reduced learning rate for the unfrozen DINOv2 blocks."""
    dinov2_blocks, dinov2_proj, other = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'rgb_backbone.dino.' in name:
            dinov2_blocks.append(p)
        elif 'rgb_backbone.projection' in name:
            dinov2_proj.append(p)
        else:
            other.append(p)

    groups = [
        {'params': dinov2_blocks, 'lr': base_lr * dinov2_lr_scale, 'name': 'dinov2_blocks'},
        {'params': dinov2_proj,   'lr': base_lr,                   'name': 'dinov2_proj'},
        {'params': other,         'lr': base_lr,                   'name': 'other'},
    ]
    groups = [g for g in groups if g['params']]
    print("\nOptimizer parameter groups:")
    for g in groups:
        print(f"  {g['name']}: {sum(p.numel() for p in g['params']):,} params, lr={g['lr']:.2e}")
    return optim.Adam(groups, weight_decay=5e-6)


# =====================================================================
# Pre-processing (shared by train & val)
# =====================================================================

def preprocess_batch(sample, img_shape, max_depth):
    """
    Project the local map into the image with the GT pose and with the
    perturbed pose to build the model inputs.

    Returns dict:
        'rgb'   : RGB image
        'lidar' : depth image rendered from the perturbed (initial) pose
        'depth' : depth image rendered from the GT pose (Stage-1 supervision)
        'flagt' : validity mask of 'depth'
        'ltolu', 'ltolv' : pixel correspondence GT -> initial (point selection)
    """
    lidar_input, rgb_input, flagt_input = [], [], []
    depth_input, ltolu_input, ltolv_input = [], [], []

    sample['tr_error'] = sample['tr_error'].cuda()
    sample['rot_error'] = sample['rot_error'].cuda()

    for idx in range(len(sample['rgb'])):
        real_shape = [
            sample['rgb'][idx].shape[1],
            sample['rgb'][idx].shape[2],
            sample['rgb'][idx].shape[0],
        ]

        sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
        pcl = sample['point_cloud'][idx].clone()

        R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(sample['tr_error'][idx])
        RT = T @ R
        pc_rotated = rotate_back(pcl, RT)

        if max_depth < 100.:
            pc_rotated = pc_rotated[:, pc_rotated[0, :] < max_depth].clone()

        cam_params = sample['calib'][idx].cuda()
        cam_model = CameraModel()
        cam_model.focal_length = cam_params[:2]
        cam_model.principal_point = cam_params[2:]
        uv, uvt, depth, dt, py, px, _ = cam_model.project_pytorch(
            pc_rotated, pcl, real_shape, None
        )

        uv = uv.t().int()
        uvt = uvt.t().int()

        depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float) + 1000.
        depth_imgt = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float) + 1000.
        depth_imgt = visibility.depth_image(
            uvt.contiguous(), dt, depth_imgt, uvt.shape[0], real_shape[1], real_shape[0]
        )
        depth_img = visibility.depth_image(
            uv.contiguous(), depth, depth_img, uv.shape[0], real_shape[1], real_shape[0]
        )
        depth_imgt[depth_imgt == 1000.] = 0.
        depth_img[depth_img == 1000.] = 0.

        uv = uv.long()
        uvt = uvt.long()
        indexes = depth_img[uv[:, 1], uv[:, 0]] == depth

        flagt = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
        flagt[uvt[indexes, 1], uvt[indexes, 0]] = 1
        depth_imgt = flagt * depth_imgt

        lidarToLidaru = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
        lidarToLidaru[uvt[indexes, 1], uvt[indexes, 0]] = uv[indexes, 1].float()
        lidarToLidarv = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
        lidarToLidarv[uvt[indexes, 1], uvt[indexes, 0]] = uv[indexes, 0].float()

        depth_img /= max_depth
        depth_imgt /= max_depth

        depth_img = depth_img.unsqueeze(0)
        depth_imgt = depth_imgt.unsqueeze(0)
        flagt = flagt.unsqueeze(0)
        lidarToLidaru = lidarToLidaru.unsqueeze(0)
        lidarToLidarv = lidarToLidarv.unsqueeze(0)

        rgb = sample['rgb'][idx].cuda()
        shape_pad = [0, 0, 0, 0]
        shape_pad[3] = img_shape[0] - rgb.shape[1]
        shape_pad[1] = img_shape[1] - rgb.shape[2]

        rgb = F.pad(rgb, shape_pad)
        depth_img = F.pad(depth_img, shape_pad)
        flagt = F.pad(flagt, shape_pad)
        depth_imgt = F.pad(depth_imgt, shape_pad)
        lidarToLidaru = F.pad(lidarToLidaru, shape_pad)
        lidarToLidarv = F.pad(lidarToLidarv, shape_pad)

        rgb_input.append(rgb)
        lidar_input.append(depth_img)
        flagt_input.append(flagt)
        depth_input.append(depth_imgt)
        ltolu_input.append(lidarToLidaru)
        ltolv_input.append(lidarToLidarv)

    return {
        'lidar': torch.stack(lidar_input),
        'rgb': torch.stack(rgb_input),
        'flagt': torch.stack(flagt_input),
        'depth': torch.stack(depth_input),
        'ltolu': torch.stack(ltolu_input),
        'ltolv': torch.stack(ltolv_input),
    }


# =====================================================================
# Train / val steps
# =====================================================================

def train_step(model, optimizer, batch, target_transl, target_rot, loss_fn):
    """One optimisation step. Returns (loss, mean t_err [cm], mean r_err [deg])."""
    model.train()
    optimizer.zero_grad()

    transl_err, rot_err, t0, r0 = model(
        batch['rgb'], batch['lidar'], batch['depth'], batch['flagt'], batch['ltolu'], batch['ltolv'])

    loss_s2 = loss_fn(target_transl, target_rot, transl_err, rot_err)
    loss_s1 = loss_fn(target_transl, target_rot, t0, r0)
    total_loss = 0.6 * loss_s2 + 0.4 * loss_s1

    total_loss.backward()
    optimizer.step()

    with torch.no_grad():
        B = batch['rgb'].shape[0]
        t_err_cm = sum(torch.norm(target_transl[j] - transl_err[j]).item() * 100. for j in range(B)) / B
        r_err_deg = quaternion_distance(target_rot, rot_err, rot_err.device).mean().item() * 180. / math.pi

    return total_loss.item(), t_err_cm, r_err_deg


def val_step(model, batch, target_transl, target_rot, loss_fn):
    """One validation batch. Returns (loss, sum t_err [cm], sum r_err [deg])."""
    model.eval()
    with torch.no_grad():
        transl_err, rot_err, t0, r0 = model(
            batch['rgb'], batch['lidar'], batch['depth'], batch['flagt'], batch['ltolu'], batch['ltolv'])

    total_loss = loss_fn(target_transl, target_rot, transl_err, rot_err)

    total_trasl_error = torch.tensor(0.0, device=target_rot.device)
    total_rot_error = quaternion_distance(target_rot, rot_err, target_rot.device) * 180. / math.pi
    for j in range(batch['rgb'].shape[0]):
        total_trasl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.

    return total_loss.item(), total_trasl_error.item(), total_rot_error.sum().item()


# =====================================================================
# Dataset builders
# =====================================================================

def build_datasets(cfg, use_canonical):
    """Return (train_set, val_set, train_sampler, img_shape) for cfg['dataset']."""
    common = dict(max_r=cfg['max_r'], max_t=cfg['max_t'])
    kitti_seq = f"{int(cfg['test_sequence']):02d}"

    if cfg['dataset'] == 'mixed':
        img_shape = UC.CANON_SHAPE
        kitti_tr = DatasetKitti(cfg['kitti_data_folder'], split='train', test_sequence=kitti_seq,
                                maps_folder=cfg['maps_folder'], use_canonical=True, **common)
        kitti_va = DatasetKitti(cfg['kitti_data_folder'], split='test', test_sequence=kitti_seq,
                                maps_folder=cfg['maps_folder'], use_canonical=True, **common)
        argo_tr = DatasetArgo(cfg['argo_data_folder'], split='train',
                              half_res=False, use_canonical=True, **common)
        argo_va = DatasetArgo(cfg['argo_data_folder'], split='test',
                              half_res=False, use_canonical=True, **common)
        weights = [cfg['mixed_weight_kitti'], cfg['mixed_weight_argo']]
        train_set = CombinedDataset([kitti_tr, argo_tr], weights=weights)
        val_set = CombinedDataset([kitti_va, argo_va], weights=weights)
        sampler = train_set.make_sampler()
        print(f"\n[Dataset] MIXED canonical({cfg['canon_preset']})  img_shape={img_shape}")
        print(f"  train={len(train_set)} (kitti {len(kitti_tr)}, argo {len(argo_tr)})  weights={weights}")
        print(f"  val  ={len(val_set)} (kitti {len(kitti_va)}, argo {len(argo_va)})")
        return train_set, val_set, sampler, img_shape

    if cfg['dataset'] == 'kitti':
        img_shape = UC.CANON_SHAPE if use_canonical else (384, 1280)
        DS, name = DatasetKitti, 'KITTI'
        extra = dict(test_sequence=kitti_seq, maps_folder=cfg['maps_folder'],
                     use_canonical=use_canonical)
    else:  # argo
        img_shape = UC.CANON_SHAPE if use_canonical else (640, 960)
        DS, name = DatasetArgo, 'Argoverse'
        extra = dict(use_canonical=use_canonical, half_res=(not use_canonical))

    train_set = DS(cfg['data_folder'], split='train', **extra, **common)
    val_set = DS(cfg['data_folder'], split='test', **extra, **common)
    canon = f" canonical({cfg['canon_preset']})" if use_canonical else ''
    print(f"\n[Dataset] {name}{canon}  img_shape={img_shape}")
    print(f"  train={len(train_set)}  val={len(val_set)}")
    return train_set, val_set, None, img_shape


# =====================================================================
# Main
# =====================================================================

@ex.automain
def main(_config, _run, seed):
    global EPOCH

    cfg = dict(_config)
    if cfg['dataset'] not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {cfg['dataset']!r}")
    if cfg['dataset'] == 'mixed' and (not cfg['kitti_data_folder'] or not cfg['argo_data_folder']):
        raise ValueError("dataset=mixed requires kitti_data_folder and argo_data_folder")
    if cfg['dinov2_variant'] not in ('b', 's'):
        raise ValueError(f"dinov2_variant must be 'b' or 's', got {cfg['dinov2_variant']!r}")

    use_canonical = bool(cfg['use_canonical']) or cfg['dataset'] == 'mixed'
    UC.set_preset(cfg['canon_preset'])
    model_tag = f"GCMLoc-{cfg['dinov2_variant'].upper()}"

    # config embedded in every checkpoint (training config + fixed architecture)
    saved_config = dict(cfg)
    saved_config.update(ARCH)
    saved_config['use_canonical'] = use_canonical

    # ---------- run directory ----------
    ds_tag = {'kitti': f"{int(cfg['test_sequence']):02d}", 'argo': 'argo', 'mixed': 'mixed'}[cfg['dataset']]
    if cfg['resume']:
        save_dir = os.path.dirname(cfg['resume'])
        run_name = os.path.basename(save_dir)
    else:
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_tag}_mapping"
        save_dir = os.path.join(cfg['savemodel'], cfg['dataset'], ds_tag, run_name)
        os.makedirs(save_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(save_dir, 'train_console.log'), sys.stdout)
    sys.stderr = Logger(os.path.join(save_dir, 'train_error.log'), sys.stderr)

    print("=" * 60)
    print(f"GCMLoc — Offline Mapping Training ({model_tag})")
    print("=" * 60)
    print(f"Config: {cfg}")
    print("Architecture (fixed):")
    for k, v in ARCH.items():
        print(f"  {k:24s}= {v}")
    print(f"  {'dinov2_variant':24s}= {cfg['dinov2_variant']}")
    print(f"Save dir: {save_dir}")

    # ---------- data ----------
    train_set, val_set, train_sampler, img_shape = build_datasets(cfg, use_canonical)

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x): return _init_fn(x, seed)

    TrainImgLoader = torch.utils.data.DataLoader(
        dataset=train_set, shuffle=(train_sampler is None), sampler=train_sampler,
        batch_size=cfg['batch_size'], num_workers=cfg['num_worker'], worker_init_fn=init_fn,
        collate_fn=merge_inputs, drop_last=False, pin_memory=True,
    )
    TestImgLoader = torch.utils.data.DataLoader(
        dataset=val_set, shuffle=False,
        batch_size=cfg['batch_size'], num_workers=cfg['num_worker'], worker_init_fn=init_fn,
        collate_fn=merge_inputs, drop_last=False, pin_memory=True,
    )

    # ---------- model / optimiser ----------
    print("\nBuilding model...")
    model = build_mapping_model(img_shape, cfg['dinov2_variant'])
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")
    model = model.to(device)

    optimizer = build_optimizer(model, cfg['BASE_LEARNING_RATE'], cfg['dinov2_lr_scale'])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg['lr_milestones'], gamma=cfg['lr_gamma'])
    print(f"LR Scheduler: MultiStepLR milestones={cfg['lr_milestones']} gamma={cfg['lr_gamma']}")

    loss_fn = ProposedLoss(cfg['rescale_transl'], cfg['rescale_rot'])

    starting_epoch = 0
    if cfg['resume']:
        print(f"\nResuming from {cfg['resume']}")
        ck = torch.load(cfg['resume'], map_location=device, weights_only=False)
        model.load_state_dict(ck['state_dict'])
        optimizer.load_state_dict(ck['optimizer'])
        if 'scheduler' in ck:
            scheduler.load_state_dict(ck['scheduler'])
        starting_epoch = ck['epoch'] + 1

    def save_ckpt(path, epoch, train_loss, test_loss):
        torch.save({
            'config': saved_config,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, path)

    # ---------- training loop ----------
    BEST_VAL_LOSS = float('inf')
    BEST_VAL_EPOCH = 0
    PATIENCE = 30                   # early stopping (epochs without val improvement)
    LOG_INTERVAL = 50               # iterations between console log lines
    epochs_without_improvement = 0
    old_save_filename = None
    start_full_time = time.time()

    log_path = os.path.join(save_dir, 'train_log.csv')
    plot_path = os.path.join(save_dir, 'loss_curve.png')
    log_exists = os.path.exists(log_path)
    log_file = open(log_path, 'a', newline='', buffering=1)
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_t_err_cm', 'val_r_err_deg',
                             'lr', 'epoch_time_s', 'timestamp'])

    cur_train_loss = float('nan')
    for epoch in range(starting_epoch, cfg['epochs'] + 1):
        EPOCH = epoch
        print(f"\n{'='*60}\nEpoch {epoch}/{cfg['epochs']}    |    {run_name}\n{'='*60}")

        total_train_loss = 0.
        local_loss = local_t = local_r = 0.
        epoch_start_time = time.time()

        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            batch = preprocess_batch(sample, img_shape, cfg['max_depth'])
            loss_val, t_err, r_err = train_step(
                model, optimizer, batch, sample['tr_error'], sample['rot_error'], loss_fn)
            if loss_val != loss_val:
                raise ValueError("Loss is NaN")

            local_loss += loss_val
            local_t += t_err
            local_r += r_err
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                ts = datetime.now().strftime('%H:%M:%S')
                spd = (time.time() - start_time) / batch['lidar'].shape[0]
                print(f'[{ts}] Ep{epoch} [{batch_idx+1:4d}/{len(TrainImgLoader)}] '
                      f'loss={local_loss/LOG_INTERVAL:.6f}  t_err={local_t/LOG_INTERVAL:.4f}cm  '
                      f'r_err={local_r/LOG_INTERVAL:.4f}°  {spd:.3f}s/sample')
                _run.log_scalar("Loss", local_loss / LOG_INTERVAL, epoch * len(TrainImgLoader) + batch_idx)
                local_loss = local_t = local_r = 0.

            total_train_loss += loss_val * len(sample['rgb'])

        total_time = time.time() - epoch_start_time
        cur_train_loss = total_train_loss / len(train_set)
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Epoch {epoch} | '
              f'train_loss={cur_train_loss:.6f} | time={total_time:.1f}s')
        _run.log_scalar("Total training loss", cur_train_loss, epoch)

        scheduler.step()
        cur_lr = optimizer.param_groups[-1]['lr']       # base lr ('other' group)

        # ----- validation (every epoch) -----
        total_test_loss = total_test_t = total_test_r = 0.
        for sample in TestImgLoader:
            batch = preprocess_batch(sample, img_shape, cfg['max_depth'])
            loss_val, trasl_e, rot_e = val_step(
                model, batch, sample['tr_error'], sample['rot_error'], loss_fn)
            if loss_val != loss_val:
                raise ValueError("Loss is NaN")
            total_test_t += trasl_e
            total_test_r += rot_e
            total_test_loss += loss_val * len(sample['rgb'])

        val_loss = total_test_loss / len(val_set)
        val_t = total_test_t / len(val_set)
        val_r = total_test_r / len(val_set)
        print("------------------------------------")
        print(f'[{datetime.now().strftime("%H:%M:%S")}] [VAL] Ep{epoch}')
        print(f'Val loss = {val_loss:.6f}')
        print(f'Translation error: {val_t:.4f} cm')
        print(f'Rotation error: {val_r:.4f} °')
        print("------------------------------------")
        _run.log_scalar("Val_Loss", val_loss, epoch)
        _run.log_scalar("Val_t_error", val_t, epoch)
        _run.log_scalar("Val_r_error", val_r, epoch)

        if val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss
            BEST_VAL_EPOCH = epoch
            epochs_without_improvement = 0
            _run.result = val_t
            savefilename = (f'{save_dir}/checkpoint_r{cfg["max_r"]:.2f}_t{cfg["max_t"]:.2f}'
                            f'_e{epoch}_{val_loss:.5f}_mapping.tar')
            save_ckpt(savefilename, epoch, cur_train_loss, val_loss)
            print(f'Model saved as {savefilename}')
            if old_save_filename is not None and os.path.exists(old_save_filename):
                os.remove(old_save_filename)
            old_save_filename = savefilename
        else:
            epochs_without_improvement += 1
            print(f"[EarlyStop] No improvement for {epochs_without_improvement}/{PATIENCE} epochs.")

        save_ckpt(f'{save_dir}/checkpoint_latest_mapping.tar', epoch, cur_train_loss, BEST_VAL_LOSS)

        log_writer.writerow([epoch, f'{cur_train_loss:.6f}', f'{val_loss:.6f}', f'{val_t:.4f}',
                             f'{val_r:.4f}', f'{cur_lr:.2e}', f'{total_time:.1f}',
                             datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

        try:
            df = pd.read_csv(log_path).drop_duplicates(subset=['epoch'], keep='last')
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 1, 1)
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
            plt.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o', color='orange')
            plt.title('Training & Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
            plt.grid(True); plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(df['epoch'], df['val_t_err_cm'], label='Val T-Error (cm)', marker='s', color='green')
            plt.plot(df['epoch'], df['val_r_err_deg'], label='Val R-Error (deg)', marker='^', color='red')
            plt.title('Validation Translation & Rotation Errors'); plt.xlabel('Epoch'); plt.ylabel('Error')
            plt.grid(True); plt.legend()
            plt.tight_layout(); plt.savefig(plot_path); plt.close()
        except Exception as e:
            print(f"[Warning] Failed to plot: {e}")

        if epochs_without_improvement >= PATIENCE:
            print(f"\n[EarlyStop] Triggered at epoch {epoch}. Best was epoch {BEST_VAL_EPOCH}.")
            break

    print(f'\nFull training time = {(time.time() - start_full_time) / 3600:.2f} HR')
    last_filename = (f'{save_dir}/checkpoint_r{cfg["max_r"]:.2f}_t{cfg["max_t"]:.2f}'
                     f'_e{epoch}_LAST_mapping.tar')
    save_ckpt(last_filename, epoch, cur_train_loss, BEST_VAL_LOSS)
    print(f'Last epoch model saved as {last_filename}')
    print(f'Best val loss {BEST_VAL_LOSS:.6f} @ epoch {BEST_VAL_EPOCH}  →  {old_save_filename}')
    log_file.close()
    return _run.result
