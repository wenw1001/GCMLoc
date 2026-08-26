"""
GCMLoc — Online Localization Training (Stage 2).

Trains the online localization network on the distilled sparse maps written by
train_save.py (<sequence>/<maps_folder>/<frame>.npy). The architecture is fixed; it is
read from the `weights` checkpoint's config and validated against the weights
(GCMLoc-B / GCMLoc-S detected automatically):

    RGB backbone : DINOv2 ViT (last 4 blocks trainable, lr x dinov2_lr_scale)
    Map branch   : depth CNN branch_lhmap (all three branches kept so mapping
                   checkpoints load; only branch_lhmap runs online)
    Heatmap      : HeatmapDecoder (multi-scale U-Net, output H/4)
    Matching     : GCM-Flow -> heatmap-weighted pooling -> PoseRegressor
    Optimiser    : Adam (weight_decay 1e-5), CosineAnnealingLR to 1e-6, grad-clip 1.0

Training is coarse-to-fine over three rounds; each round starts from the previous
round's best checkpoint and narrows the initial-pose perturbation:

    Round 1: max_r=10 max_t=2    weights=<mapping checkpoint from train_mapping.py>
    Round 2: max_r=2  max_t=1    weights=<round-1 best _loc.tar>
    Round 3: max_r=1  max_t=0.6  weights=<round-2 best _loc.tar>

Datasets (must match how the maps were saved):
    dataset=kitti : (384, 1280), maps at <seq>/<maps_folder>/*.npy, `test_sequence` = validation
    dataset=argo  : half-res (640, 960), maps at <log>/<maps_folder>/*.npy, train1-3 / train4
    dataset=mixed : canonical camera (canon_preset=A, 768x384), KITTI + Argoverse sampled
                    with weights mixed_weight_kitti : mixed_weight_argo; maps must be saved
                    with use_canonical=True (e.g. save_name=v2_pcl_mix)

Outputs: <savemodel>/<dataset>/<seq|argo|mixed>/<timestamp>_GCMLoc-<B|S>_loc_r{max_r}_t{max_t}/
    checkpoint_r{max_r}_t{max_t}_e{epoch}_{val_loss}_loc.tar   best model
    checkpoint_latest_loc.tar                                   for resume=
    train_log.csv, loss_curve.png, train_console.log

Examples
--------
# KITTI, round 1 (from the mapping checkpoint)
python train_loc.py with dataset=kitti \
    data_folder=./KITTI_ODOMETRY/sequences test_sequence=0 maps_folder=v2_pcl \
    epochs=300 batch_size=8 max_r=10 max_t=2 weights=<mapping_checkpoint>.tar
# rounds 2 / 3: epochs=200, max_r=2 max_t=1 then max_r=1 max_t=0.6,
#               weights=<previous round best _loc.tar>

# Argoverse, round 1
python train_loc.py with dataset=argo \
    data_folder=./data/argoverse-tracking maps_folder=v2_pcl \
    epochs=150 batch_size=4 max_r=10 max_t=2 weights=<mapping_checkpoint>.tar

# Mixed (canonical A), round 1
python train_loc.py with dataset=mixed \
    kitti_data_folder=./KITTI_ODOMETRY/sequences \
    argo_data_folder=./data/argoverse-tracking maps_folder=v2_pcl_mix \
    epochs=100 batch_size=8 max_r=10 max_t=2 weights=<mixed_mapping_checkpoint>.tar
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

from camera_model_localization import CameraModel
from Dataset_kitti_localization import DatasetVisibilityKittiSingle as DatasetKitti
from Dataset_argoverse_localization import DatasetVisibilityKittiSingle as DatasetArgo
from combined_dataset import CombinedDataset
import utils_canonical as UC
from losses import ProposedLoss
from models.GCMLoc import load_checkpoint, build_localization_from_checkpoint
from quaternion_distances import quaternion_distance
from utils import merge_inputs, rotate_back

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

SETTINGS.DISCOVER_DEPENDENCIES = "none"
SETTINGS.DISCOVER_SOURCES = "none"
ex = Experiment("GCMLoc-loc")
ex.captured_out_filter = apply_backspaces_and_linefeeds

GRAD_CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-5


@ex.config
def config():
    savemodel = './checkpoints/'    # root of the run directories
    dataset = 'kitti'               # 'kitti' | 'argo' | 'mixed'
    data_folder = './KITTI_ODOMETRY/sequences'   # KITTI / Argoverse root (single-dataset training)
    test_sequence = 0               # KITTI held-out validation sequence (int); also used by mixed; ignored by argo
    maps_folder = 'v2_pcl'          # distilled-map folder written by train_save.py (its save_name)
    weights = None                  # REQUIRED: round 1 = mapping checkpoint; round 2+ = previous round best
    resume = None                   # checkpoint_latest_loc.tar to continue an interrupted run

    # -- optimisation --
    epochs = 300                    # rounds 2/3 use 200 (argo 150, mixed 100)
    batch_size = 8
    num_worker = 3                  # DataLoader workers
    BASE_LEARNING_RATE = 1e-4       # Adam learning rate (CosineAnnealingLR to 1e-6)
    dinov2_lr_scale = 0.01          # lr of the unfrozen DINOv2 blocks = base lr * scale
    rescale_transl = 1.0            # loss weight, translation term
    rescale_rot = 1.0               # loss weight, rotation term
    val_interval = 2                # validate every N epochs (always in the last 5)

    # -- pose perturbation range of this round --
    max_t = 2.                      # metres  (round 1: 2, round 2: 1, round 3: 0.6)
    max_r = 10.                     # degrees (round 1: 10, round 2: 2, round 3: 1)
    max_depth = 100.                # depth normalisation constant

    # -- canonical camera / mixed training --
    use_canonical = False           # must match how the maps were saved; always on for mixed
    canon_preset = 'A'              # 'A' = 768x384 (paper)
    kitti_data_folder = ''          # mixed only
    argo_data_folder = ''           # mixed only
    mixed_weight_kitti = 1.0        # mixed only: sampling weight of KITTI
    mixed_weight_argo = 2.0         # mixed only: sampling weight of Argoverse


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
# Optimiser
# =====================================================================

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
    return optim.Adam(groups, weight_decay=WEIGHT_DECAY)


# =====================================================================
# Pre-processing (shared by train & val)
# =====================================================================

def preprocess_batch(sample, img_shape, max_depth):
    """
    Project the sparse distilled map with the perturbed initial pose.

    Returns dict: 'rgb' (B, 3, H, W), 'lidar' (B, 1, H, W) projected sparse-map depth.
    """
    rgb_input, lidar_input = [], []
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
        pc_rotated = rotate_back(pcl, T @ R)

        if max_depth < 100.:
            pc_rotated = pc_rotated[:, pc_rotated[0, :] < max_depth].clone()

        cam_params = sample['calib'][idx].cuda()
        cam_model = CameraModel()
        cam_model.focal_length = cam_params[:2]
        cam_model.principal_point = cam_params[2:]
        uv, depth, py, px, _ = cam_model.project_pytorch(pc_rotated, real_shape, None)

        uv = uv.t().int()
        depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float) + 1000.
        depth_img = visibility.depth_image(
            uv.contiguous(), depth, depth_img, uv.shape[0], real_shape[1], real_shape[0])
        depth_img[depth_img == 1000.] = 0.
        depth_img = (depth_img / max_depth).unsqueeze(0)

        rgb = sample['rgb'][idx].cuda()
        pad = [0, img_shape[1] - rgb.shape[2], 0, img_shape[0] - rgb.shape[1]]
        rgb_input.append(F.pad(rgb, pad))
        lidar_input.append(F.pad(depth_img, pad))

    return {'rgb': torch.stack(rgb_input), 'lidar': torch.stack(lidar_input)}


# =====================================================================
# Train / val steps
# =====================================================================

def train_step(model, optimizer, batch, target_transl, target_rot, loss_fn):
    """One optimisation step. Returns (loss, mean t_err [cm], mean r_err [deg])."""
    model.train()
    optimizer.zero_grad()

    transl_err, rot_err = model(batch['rgb'], batch['lidar'])
    total_loss = loss_fn(target_transl, target_rot, transl_err, rot_err)

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
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
        transl_err, rot_err = model(batch['rgb'], batch['lidar'])

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
    common = dict(max_r=cfg['max_r'], max_t=cfg['max_t'], maps_folder=cfg['maps_folder'])
    kitti_seq = f"{int(cfg['test_sequence']):02d}"

    if cfg['dataset'] == 'mixed':
        img_shape = UC.CANON_SHAPE
        kitti_tr = DatasetKitti(cfg['kitti_data_folder'], split='train', test_sequence=kitti_seq,
                                use_canonical=True, **common)
        kitti_va = DatasetKitti(cfg['kitti_data_folder'], split='test', test_sequence=kitti_seq,
                                use_canonical=True, **common)
        argo_tr = DatasetArgo(cfg['argo_data_folder'], split='train',
                              half_res=False, use_canonical=True, **common)
        argo_va = DatasetArgo(cfg['argo_data_folder'], split='test',
                              half_res=False, use_canonical=True, **common)
        weights = [cfg['mixed_weight_kitti'], cfg['mixed_weight_argo']]
        train_set = CombinedDataset([kitti_tr, argo_tr], weights=weights)
        val_set = CombinedDataset([kitti_va, argo_va], weights=weights)
        sampler = train_set.make_sampler()
        print(f"\n[Dataset] MIXED canonical({cfg['canon_preset']})  img_shape={img_shape}  maps={cfg['maps_folder']}")
        print(f"  train={len(train_set)} (kitti {len(kitti_tr)}, argo {len(argo_tr)})  weights={weights}")
        print(f"  val  ={len(val_set)} (kitti {len(kitti_va)}, argo {len(argo_va)})")
        return train_set, val_set, sampler, img_shape

    if cfg['dataset'] == 'kitti':
        img_shape = UC.CANON_SHAPE if use_canonical else (384, 1280)
        DS, name = DatasetKitti, 'KITTI'
        extra = dict(test_sequence=kitti_seq, use_canonical=use_canonical)
    else:  # argo
        img_shape = UC.CANON_SHAPE if use_canonical else (640, 960)
        DS, name = DatasetArgo, 'Argoverse'
        extra = dict(use_canonical=use_canonical, half_res=(not use_canonical))

    train_set = DS(cfg['data_folder'], split='train', **extra, **common)
    val_set = DS(cfg['data_folder'], split='test', **extra, **common)
    canon = f" canonical({cfg['canon_preset']})" if use_canonical else ''
    print(f"\n[Dataset] {name}{canon}  img_shape={img_shape}  maps={cfg['maps_folder']}")
    print(f"  train={len(train_set)}  val={len(val_set)}")
    return train_set, val_set, None, img_shape


# =====================================================================
# Main
# =====================================================================

@ex.automain
def main(_config, _run, seed):
    global EPOCH

    cfg = dict(_config)
    if cfg['dataset'] not in ('kitti', 'argo', 'mixed'):
        raise ValueError(f"dataset must be 'kitti', 'argo' or 'mixed', got {cfg['dataset']!r}")
    if cfg['dataset'] == 'mixed' and (not cfg['kitti_data_folder'] or not cfg['argo_data_folder']):
        raise ValueError("dataset=mixed requires kitti_data_folder and argo_data_folder")
    if not cfg['resume'] and not cfg['weights']:
        raise ValueError("weights=<mapping or previous-round checkpoint> is required "
                         "(round 1: train_mapping.py checkpoint; later rounds: previous best _loc.tar)")

    use_canonical = bool(cfg['use_canonical']) or cfg['dataset'] == 'mixed'
    UC.set_preset(cfg['canon_preset'])

    # ---------- model (architecture from the checkpoint config, validated vs weights) ----------
    init_path = cfg['resume'] or cfg['weights']
    ckpt = load_checkpoint(init_path)
    # img_shape is needed to build the model; derive it before the datasets
    img_shape = UC.CANON_SHAPE if use_canonical else ((384, 1280) if cfg['dataset'] == 'kitti' else (640, 960))
    model, arch, missing = build_localization_from_checkpoint(ckpt, img_shape)
    model_tag = f"GCMLoc-{arch['dinov2_variant'].upper()}"

    # config embedded in every checkpoint (training config + inferred architecture)
    saved_config = dict(cfg)
    saved_config.update(arch)
    saved_config['use_canonical'] = use_canonical

    # ---------- run directory ----------
    ds_tag = {'kitti': f"{int(cfg['test_sequence']):02d}", 'argo': 'argo', 'mixed': 'mixed'}[cfg['dataset']]
    if cfg['resume']:
        save_dir = os.path.dirname(cfg['resume'])
        run_name = os.path.basename(save_dir)
    else:
        run_name = (f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_tag}"
                    f"_loc_r{cfg['max_r']:g}_t{cfg['max_t']:g}")
        save_dir = os.path.join(cfg['savemodel'], cfg['dataset'], ds_tag, run_name)
        os.makedirs(save_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(save_dir, 'train_console.log'), sys.stdout)
    sys.stderr = Logger(os.path.join(save_dir, 'train_error.log'), sys.stderr)

    print("=" * 60)
    print(f"GCMLoc — Online Localization Training ({model_tag}, "
          f"round max_r={cfg['max_r']:g} max_t={cfg['max_t']:g})")
    print("=" * 60)
    print(f"Config: {cfg}")
    print(f"Init checkpoint: {init_path}  (epoch {ckpt.get('epoch')})")
    print(f"Architecture: {arch}  missing keys: {len(missing)}")
    if missing:
        print(f"  [INFO] Random init (expected from a mapping checkpoint): "
              f"{sorted(set(k.split('.')[0] for k in missing))}")
    print(f"Save dir: {save_dir}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    model = model.to(device)

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

    # ---------- optimiser / scheduler / loss ----------
    optimizer = build_optimizer(model, cfg['BASE_LEARNING_RATE'], cfg['dinov2_lr_scale'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)
    print(f"LR Scheduler: CosineAnnealingLR T_max={cfg['epochs']} eta_min=1e-6")
    print(f"Gradient clipping: max_norm={GRAD_CLIP_NORM}")

    loss_fn = ProposedLoss(cfg['rescale_transl'], cfg['rescale_rot'])

    starting_epoch = 0
    if cfg['resume']:
        print(f"\nResuming from {cfg['resume']}")
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        starting_epoch = ckpt['epoch'] + 1

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
    PATIENCE = 30                   # early stopping (validation steps without improvement)
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

        # ----- validation (every val_interval epochs, always in the last 5) -----
        do_val = (epoch % cfg['val_interval'] == 0) or (epoch > cfg['epochs'] - 5)
        csv_val_loss = csv_val_t = csv_val_r = ''
        if do_val:
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
            csv_val_loss, csv_val_t, csv_val_r = f'{val_loss:.6f}', f'{val_t:.4f}', f'{val_r:.4f}'

            if val_loss < BEST_VAL_LOSS:
                BEST_VAL_LOSS = val_loss
                BEST_VAL_EPOCH = epoch
                epochs_without_improvement = 0
                _run.result = val_t
                savefilename = (f'{save_dir}/checkpoint_r{cfg["max_r"]:.2f}_t{cfg["max_t"]:.2f}'
                                f'_e{epoch}_{val_loss:.5f}_loc.tar')
                save_ckpt(savefilename, epoch, cur_train_loss, val_loss)
                print(f'Model saved as {savefilename}')
                if old_save_filename is not None and os.path.exists(old_save_filename):
                    os.remove(old_save_filename)
                old_save_filename = savefilename
            else:
                epochs_without_improvement += 1
                print(f"[EarlyStop] No improvement for {epochs_without_improvement}/{PATIENCE} val steps.")

        save_ckpt(f'{save_dir}/checkpoint_latest_loc.tar', epoch, cur_train_loss, BEST_VAL_LOSS)

        log_writer.writerow([epoch, f'{cur_train_loss:.6f}', csv_val_loss, csv_val_t,
                             csv_val_r, f'{cur_lr:.2e}', f'{total_time:.1f}',
                             datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

        try:
            df = pd.read_csv(log_path).drop_duplicates(subset=['epoch'], keep='last')
            val_df = df.dropna(subset=['val_loss'])
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 1, 1)
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
            if not val_df.empty:
                plt.plot(val_df['epoch'], val_df['val_loss'], label='Val Loss', marker='o', color='orange')
            plt.title('Training & Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
            plt.grid(True); plt.legend()
            plt.subplot(2, 1, 2)
            if not val_df.empty:
                plt.plot(val_df['epoch'], val_df['val_t_err_cm'], label='Val T-Error (cm)', marker='s', color='green')
                plt.plot(val_df['epoch'], val_df['val_r_err_deg'], label='Val R-Error (deg)', marker='^', color='red')
            plt.title('Validation Translation & Rotation Errors'); plt.xlabel('Epoch'); plt.ylabel('Error')
            plt.grid(True); plt.legend()
            plt.tight_layout(); plt.savefig(plot_path); plt.close()
        except Exception as e:
            print(f"[Warning] Failed to plot: {e}")

        if epochs_without_improvement >= PATIENCE:
            print(f"\n[EarlyStop] Triggered at epoch {epoch}. Best was epoch {BEST_VAL_EPOCH}.")
            break

    print(f'\nFull training time = {(time.time() - start_full_time) / 3600:.2f} HR')
    print(f'Best val loss {BEST_VAL_LOSS:.6f} @ epoch {BEST_VAL_EPOCH}  →  {old_save_filename}')
    log_file.close()
    return _run.result
