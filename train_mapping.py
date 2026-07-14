"""
GCMLoc — Stage-1 Mapping Training Script.

Trains the Stage-1 mapping network (offline map-feature construction).
The default configuration is the final GCMLoc model:

    RGB backbone : DINOv2 ViT-B/14, last 4 transformer blocks unfrozen
    Depth backbone: CNN (3-branch: gt / init / gcmloc)
    Cross-modal fusion : enabled (CrossModalFusion, depth x RGB feature)
    Flow matcher : multi-scale (MultiScaleFlowMatcher)

Single-dataset training (KITTI):
    python train_mapping.py with batch_size=8 \
        data_folder=./KITTI_ODOMETRY/sequences/ \
        test_sequence=0 epochs=120 max_r=10 max_t=2

Mixed-dataset training (KITTI + Argoverse in canonical camera space):
    python train_mapping.py with dataset=mixed use_canonical=True canon_preset=A \
        kitti_data_folder=./KITTI_ODOMETRY/sequences/ \
        argo_data_folder=<path/to/argoverse-tracking>/ \
        kitti_maps_folder=local_maps_0.1 \
        batch_size=8 epochs=120 max_r=10 max_t=2

Architecture flags (rgb_backbone / depth_backbone / use_cross_fusion /
flow_type / unfreeze_dinov2_blocks) can be overridden for ablation studies;
model_type=original rebuilds the CMRNet baseline.
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
from losses import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss
from losses.flow_loss import compute_gt_flow_multiscale, flow_supervision_loss
from models.GCMLoc.GCMLoc_mapping import GCMLocMapping
from models.GCMLoc.CMRNet_single_mapping import CMRNet as CMRNetOriginal
from quaternion_distances import quaternion_distance
from utils import merge_inputs, overlay_imgs, rotate_back

# 0 → KITTI Odometry   1 → Argoverse Tracking   2 → Mixed
datasetType = 2

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

SETTINGS.DISCOVER_DEPENDENCIES = "none"
SETTINGS.DISCOVER_SOURCES = "none"
ex = Experiment("GCMLoc-mapping")
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    savemodel = './checkpoints/'
    # dataset tag used in checkpoint directory name
    # set to 'argo' when training on Argoverse (also change datasetType above)
    dataset = 'kitti'
    data_folder = './KITTI_ODOMETRY/sequences'
    use_reflectance = False
    test_sequence = 0
    occlusion_kernel = 5
    occlusion_threshold = 3
    epochs = 120
    BASE_LEARNING_RATE = 1e-4
    loss = 'simple'
    max_t = 2.
    max_r = 10.
    batch_size = 8
    num_worker = 3
    optimizer = 'adam'
    resume = None
    weights = None
    rescale_rot = 10
    rescale_transl = 1
    dropout = 0.0
    max_depth = 100.
    # KITTI mapping: 'local_maps_0.1'  |  Argoverse mapping: 'argo_local_maps_0.1' (h5)
    # set to '' to use raw single-sweep PLY files from Argoverse
    maps_folder = 'local_maps_0.1'
    # Argoverse only: root dir where H5 maps were stored via --output_root
    # (leave '' for legacy next-to-PLY layout)
    maps_root = ''

    # ===== Canonical normalization / mixed (datasetType==2) training =====
    use_canonical      = False
    canon_preset       = 'A'                 # 'A'=paper 768x384 | 'B'=960x480 wide
    kitti_data_folder  = ''
    argo_data_folder   = ''
    kitti_maps_folder  = 'local_maps_0.1'   # KITTI per-frame h5 maps (mixed)
    mixed_weight_kitti = 1.0
    mixed_weight_argo  = 2.0                 # default → KITTI ~33%

    # ===== GCMLoc architecture config =====
    feat_dim = 128
    heatmap_dim_s1 = 64
    heatmap_dim_s2 = 128
    topk_points = 5000
    vmamba_output_stage = 2
    use_cnn_fallback = False

    # ===== Model type =====
    # 'original' : exact CMRNet_single_mapping.py (use_feat_from=2, md=4)
    # 'ablation' : GCMLocMapping with flags below
    model_type = 'ablation'

    # ===== Architecture flags (defaults = final GCMLoc model) =====
    # rgb_backbone: 'cnn', 'dinov2' (uses dinov2_variant), 'dinov2s' (ViT-S), 'dinov2b' (ViT-B)
    rgb_backbone = 'dinov2b'
    dinov2_variant = 'b'           # 'b'=ViT-B/14 (768-dim), 's'=ViT-S/14 (384-dim); used when rgb_backbone='dinov2'
    unfreeze_dinov2_blocks = 4     # 0=fully frozen, 1-12=unfreeze last N blocks
    depth_backbone = 'cnn'         # 'vmamba' or 'cnn'
    use_cross_fusion = True        # True=CrossModalFusion, False=depth only
    flow_type = 'multi_scale'      # 'multi_scale' or 'correlation'
    flow_loss_weight = 0.0         # 0=no flow loss, >0=enable flow supervision

    # ===== LR scheduler =====
    lr_scheduler = 'multistep'     # 'multistep' or 'cosine'
    lr_milestones = [20, 50, 70]   # for multistep
    lr_gamma = 0.5                 # for multistep

    # ===== DINOv2 LR =====
    dinov2_lr_scale = 0.01         # unfrozen DINOv2 blocks LR = base_lr * this
    vmamba_lr_scale = 0.1          # VMamba backbone LR = base_lr * this

    # ===== Training stability =====
    use_amp = False                 # True=fp16 AMP (faster but degrades pose accuracy)
    grad_clip = 0.0                 # max_norm for clip_grad_norm_; 0 = disabled


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 1


def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# =====================================================================
# Preprocessing helper (shared between train & validation)
# =====================================================================

def preprocess_batch(sample, img_shape, max_depth, use_reflectance, compute_flow=False):
    """
    Convert a raw batch sample into model-ready tensors.

    Returns dict with keys:
        'lidar', 'rgb', 'flagt', 'depth', 'ltolu', 'ltolv'
        'gt_flows', 'gt_masks' (only if compute_flow=True)
    """
    lidar_input, rgb_input, flagt_input = [], [], []
    depth_input, ltolu_input, ltolv_input = [], [], []

    gt_flows_batch = [[], [], []] if compute_flow else None
    gt_masks_batch = [[], [], []] if compute_flow else None

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

        reflectance = None
        if use_reflectance:
            reflectance = sample['reflectance'][idx].cuda()

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
            pc_rotated, pcl, real_shape, reflectance
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

        # GT flow for flow supervision (optional)
        if compute_flow:
            gt_flows_i, gt_masks_i = compute_gt_flow_multiscale(
                uv, uvt, indexes, img_shape[0], img_shape[1], scales=(4, 8, 16),
            )
            for s_idx in range(3):
                gt_flows_batch[s_idx].append(gt_flows_i[s_idx])
                gt_masks_batch[s_idx].append(gt_masks_i[s_idx])

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

    result = {
        'lidar': torch.stack(lidar_input),
        'rgb': torch.stack(rgb_input),
        'flagt': torch.stack(flagt_input),
        'depth': torch.stack(depth_input),
        'ltolu': torch.stack(ltolu_input),
        'ltolv': torch.stack(ltolv_input),
    }

    if compute_flow:
        result['gt_flows'] = [torch.stack(gt_flows_batch[s]) for s in range(3)]
        result['gt_masks'] = [torch.stack(gt_masks_batch[s]) for s in range(3)]

    return result


# =====================================================================
# Train & Test functions
# =====================================================================

@ex.capture
def train(model, optimizer, u, v, rgb_img, refl_img, flagt, depth,
          target_transl, target_rot, loss_fn, point_clouds, loss,
          scaler=None, grad_clip=0.0,
          gt_flows=None, gt_masks=None, flow_loss_weight=0.0):
    """Single training step. Returns (total_loss, t_err_cm, r_err_deg)."""
    model.train()
    optimizer.zero_grad()

    with torch.amp.autocast('cuda', enabled=(scaler is not None)):
        transl_err, rot_err, t0, r0 = model(rgb_img, refl_img, depth, flagt, u, v)

        if loss != 'points_distance':
            loss_s2 = loss_fn(target_transl, target_rot, transl_err, rot_err)
            loss_s1 = loss_fn(target_transl, target_rot, t0, r0)
        else:
            loss_s2 = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)
            loss_s1 = loss_fn(target_transl, target_rot, t0, r0)

        total_loss = 0.6 * loss_s2 + 0.4 * loss_s1

    # Flow supervision loss (only when multi_scale flow is used)
    if gt_flows is not None and flow_loss_weight > 0:
        _model = model.module if hasattr(model, 'module') else model
        pred_flows = getattr(_model, '_stage1_flows', None)
        if pred_flows is not None and len(pred_flows) > 0:
            loss_flow = flow_supervision_loss(pred_flows, gt_flows, gt_masks)
            total_loss = total_loss + flow_loss_weight * loss_flow

    if scaler is not None:
        scaler.scale(total_loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
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
def test(model, u, v, rgb_img, refl_img, flagt, depth,
         target_transl, target_rot, loss_fn, camera_model, point_clouds, loss,
         use_amp=False):
    """Single evaluation step."""
    model.eval()

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        transl_err, rot_err, t0, r0 = model(rgb_img, refl_img, depth, flagt, u, v)

    if loss != 'points_distance':
        total_loss = loss_fn(target_transl, target_rot, transl_err, rot_err)
    else:
        total_loss = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err)

    total_trasl_error = torch.tensor(0.0, device=target_rot.device)
    total_rot_error = quaternion_distance(target_rot, rot_err, target_rot.device)
    total_rot_error = total_rot_error * 180. / math.pi
    for j in range(rgb_img.shape[0]):
        total_trasl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.

    return total_loss.item(), total_trasl_error.item(), total_rot_error.sum().item()


# =====================================================================
# Main
# =====================================================================

@ex.automain
def main(_config, _run, seed):
    global EPOCH, datasetType

    my_config = dict(_config)

    # the `dataset` config string drives the selection (CLI-friendly):
    #   'kitti'→0, 'argo'→1, 'mixed'→2 (canonical KITTI+Argo)
    datasetType = {'kitti': 0, 'argo': 1, 'mixed': 2}.get(
        my_config.get('dataset'), datasetType)

    if my_config['test_sequence'] is None:
        raise TypeError('test_sequences cannot be None')

    uc = my_config.get('use_canonical', False)
    UC.set_preset(my_config.get('canon_preset', 'A'))
    if datasetType == 2:
        dataset_class = None                 # mixed: built below
        img_shape = UC.CANON_SHAPE
        my_config['test_sequence'] = str(my_config['test_sequence'])
    elif datasetType == 0:
        dataset_class = DatasetKitti
        my_config['test_sequence'] = f"{my_config['test_sequence']:02d}"
        img_shape = UC.CANON_SHAPE if uc else (384, 1280)
    else:
        dataset_class = DatasetArgo
        # Argoverse images are 1920×1200; half-res (×0.5) → 960×600, pad H to 640
        img_shape = UC.CANON_SHAPE if uc else (640, 960)
        # test_sequence is not used for Argoverse (val split is always the test set)
        my_config['test_sequence'] = str(my_config['test_sequence'])

    my_config["savemodel"] = os.path.join(my_config["savemodel"], my_config['dataset'])
    maps_folder = my_config.get('maps_folder', 'local_maps')

    # ==================== Save directory ====================
    if _config['resume']:
        save_dir = os.path.dirname(_config['resume'])
        run_timestamp = os.path.basename(save_dir)
    else:
        # Include model type / ablation config in folder name
        if _config['model_type'] == 'original':
            ablation_tag = 'original_CMRNet'
        else:
            ablation_tag = (
                f"rgb-{_config['rgb_backbone']}"
                f"_depth-{_config['depth_backbone']}"
                f"_fuse-{_config['use_cross_fusion']}"
                f"_flow-{_config['flow_type']}"
            )
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + ablation_tag
        base_save_dir = os.path.join(my_config["savemodel"], my_config['test_sequence'])
        save_dir = os.path.join(base_save_dir, run_timestamp)
        os.makedirs(save_dir, exist_ok=True)

    # Terminal logger
    class Logger(object):
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

    sys.stdout = Logger(os.path.join(save_dir, 'train_console.log'), sys.stdout)
    sys.stderr = Logger(os.path.join(save_dir, 'train_error.log'), sys.stderr)

    print("=" * 60)
    print("GCMLoc — Ablation Training")
    print("=" * 60)
    print(f"Config: {_config}")
    print(f"\nModel type: {_config['model_type']}")
    if _config['model_type'] == 'ablation':
        _rb = _config['rgb_backbone']
        if _rb.startswith('dinov2'):
            _variant_str = ('s' if _rb == 'dinov2s' else ('b' if _rb == 'dinov2b' else _config['dinov2_variant']))
            print(f"  RGB backbone:     {_rb} (ViT-{'S' if _variant_str == 's' else 'B'}/14)")
        else:
            print(f"  RGB backbone:     {_rb}")
        print(f"  DINOv2 unfreeze:  {_config['unfreeze_dinov2_blocks']} blocks")
        print(f"  Depth backbone:   {_config['depth_backbone']}")
        print(f"  Cross fusion:     {_config['use_cross_fusion']}")
        print(f"  Flow type:        {_config['flow_type']}")
        print(f"  Flow loss weight: {_config['flow_loss_weight']}")
    print(f"  LR scheduler:     {_config['lr_scheduler']}")
    print(f"  Save dir:         {save_dir}")

    # ==================== Dataset ====================
    train_sampler = None
    if datasetType == 2:
        # mixed KITTI + Argoverse mapping in canonical space
        print(f"\n[Dataset] MIXED canonical  img_shape={img_shape}")
        kitti_tr = DatasetKitti(my_config['kitti_data_folder'],
                                max_r=my_config['max_r'], max_t=my_config['max_t'],
                                split='train', use_reflectance=my_config['use_reflectance'],
                                maps_folder=my_config['kitti_maps_folder'],
                                test_sequence='00', use_canonical=True)
        argo_kw = {'maps_root': my_config['maps_root']} if my_config.get('maps_root', '') else {}
        argo_tr  = DatasetArgo(my_config['argo_data_folder'],
                               max_r=my_config['max_r'], max_t=my_config['max_t'],
                               split='train', use_reflectance=my_config['use_reflectance'],
                               maps_folder=maps_folder, half_res=False,
                               use_canonical=True, **argo_kw)
        kitti_va = DatasetKitti(my_config['kitti_data_folder'],
                                max_r=my_config['max_r'], max_t=my_config['max_t'],
                                split='test', use_reflectance=my_config['use_reflectance'],
                                maps_folder=my_config['kitti_maps_folder'],
                                test_sequence='00', use_canonical=True)
        argo_va  = DatasetArgo(my_config['argo_data_folder'],
                               max_r=my_config['max_r'], max_t=my_config['max_t'],
                               split='test', use_reflectance=my_config['use_reflectance'],
                               maps_folder=maps_folder, half_res=False,
                               use_canonical=True, **argo_kw)
        weights     = [my_config['mixed_weight_kitti'], my_config['mixed_weight_argo']]
        dataset     = CombinedDataset([kitti_tr, argo_tr], weights=weights)
        dataset_val = CombinedDataset([kitti_va, argo_va], weights=weights)
        train_sampler = dataset.make_sampler()
        print(f"  train={len(dataset)} (kitti {len(kitti_tr)}, argo {len(argo_tr)})  "
              f"weights={weights}")
    else:
        print(f"\n[Dataset] type={'Argoverse' if datasetType == 1 else 'KITTI'}"
              f"{' canonical' if uc else ''}  img_shape={img_shape}")
        extra_kwargs = {'use_canonical': uc}
        if datasetType == 1:
            extra_kwargs['half_res'] = (not uc)
            if my_config.get('maps_root', ''):
                extra_kwargs['maps_root'] = my_config['maps_root']
        dataset = dataset_class(
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

    batch_size = _config['batch_size']
    num_worker = _config['num_worker']

    TrainImgLoader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=(train_sampler is None), sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_worker, worker_init_fn=init_fn,
        collate_fn=merge_inputs, drop_last=False, pin_memory=True,
    )
    TestImgLoader = torch.utils.data.DataLoader(
        dataset=dataset_val, shuffle=False, batch_size=batch_size,
        num_workers=num_worker, worker_init_fn=init_fn,
        collate_fn=merge_inputs, drop_last=False, pin_memory=True,
    )

    # ==================== Build Model ====================
    print("\nBuilding model...")
    if _config['model_type'] == 'original':
        # Exact original CMRNet — same as main_single_save.py (use_feat_from=2, md=4)
        model = CMRNetOriginal(
            img_shape,
            use_feat_from=2,
            md=4,
            use_reflectance=_config['use_reflectance'],
            dropout=_config['dropout'],
        )
    else:
        model = GCMLocMapping(
            image_size=img_shape,
            feat_dim=_config['feat_dim'],
            heatmap_dim_s1=_config['heatmap_dim_s1'],
            heatmap_dim_s2=_config['heatmap_dim_s2'],
            topk_points=_config['topk_points'],
            vmamba_output_stage=_config['vmamba_output_stage'],
            use_cnn_fallback=_config['use_cnn_fallback'],
            dropout=_config['dropout'],
            use_reflectance=_config['use_reflectance'],
            rgb_backbone=_config['rgb_backbone'],
            dinov2_variant=_config['dinov2_variant'],
            unfreeze_dinov2_blocks=_config['unfreeze_dinov2_blocks'],
            depth_backbone=_config['depth_backbone'],
            use_cross_fusion=_config['use_cross_fusion'],
            flow_type=_config['flow_type'],
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {frozen_params:,}")

    model = model.to(device)

    # ==================== Optimizer ====================
    base_lr = _config['BASE_LEARNING_RATE']
    vmamba_lr_scale = _config['vmamba_lr_scale']
    dinov2_lr_scale = _config['dinov2_lr_scale']

    # Original model: single flat param group (matches main_single_save.py)
    if _config['model_type'] == 'original':
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        param_groups = [{'params': parameters, 'lr': base_lr, 'name': 'all'}]
        print(f"  all: {sum(p.numel() for p in parameters):,} params, lr={base_lr:.2e}")
    else:
        # separate LR groups per module
        dinov2_unfrozen_params = []
        dinov2_proj_params = []
        vmamba_backbone_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'rgb_backbone.dino.' in name:
                dinov2_unfrozen_params.append(param)
            elif 'rgb_backbone.projection' in name:
                dinov2_proj_params.append(param)
            elif 'depth_backbone.backbone' in name:
                vmamba_backbone_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if dinov2_unfrozen_params:
            param_groups.append({
                'params': dinov2_unfrozen_params,
                'lr': base_lr * dinov2_lr_scale,
                'name': 'dinov2_blocks',
            })
        if dinov2_proj_params:
            param_groups.append({
                'params': dinov2_proj_params,
                'lr': base_lr,
                'name': 'dinov2_proj',
            })
        if vmamba_backbone_params:
            param_groups.append({
                'params': vmamba_backbone_params,
                'lr': base_lr * vmamba_lr_scale,
                'name': 'vmamba_backbone',
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,
                'name': 'other',
            })

    print("\nOptimizer parameter groups:")
    for g in param_groups:
        n_params = sum(p.numel() for p in g['params'])
        print(f"  {g['name']}: {n_params:,} params, lr={g['lr']:.2e}")

    optimizer = optim.Adam(param_groups, weight_decay=5e-6)

    # ==================== LR Scheduler ====================
    if _config['lr_scheduler'] == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=_config['lr_milestones'],
            gamma=_config['lr_gamma'],
        )
        print(f"LR Scheduler: MultiStepLR milestones={_config['lr_milestones']} gamma={_config['lr_gamma']}")
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=_config['epochs'], eta_min=1e-6,
        )
        print(f"LR Scheduler: CosineAnnealingLR T_max={_config['epochs']}")

    # ==================== Loss Function ====================
    if _config['loss'] == 'simple':
        loss_fn = ProposedLoss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'geometric':
        loss_fn = GeometricLoss().to(device)
    elif _config['loss'] == 'points_distance':
        loss_fn = DistancePoints3D()
    elif _config['loss'] == 'L1':
        loss_fn = L1Loss(_config['rescale_transl'], _config['rescale_rot'])

    # ==================== Resume ====================
    starting_epoch = 0
    if _config['resume']:
        print(f"\nResuming from {_config['resume']}")
        checkpoint = torch.load(_config['resume'], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        starting_epoch = checkpoint['epoch'] + 1

    # ==================== Training Loop ====================
    BEST_VAL_LOSS = float('inf')
    BEST_VAL_EPOCH = 0
    PATIENCE = 30
    epochs_without_improvement = 0
    old_save_filename = None
    start_full_time = time.time()

    scaler = torch.amp.GradScaler('cuda') if _config['use_amp'] else None
    compute_flow = (
        _config['model_type'] == 'ablation'
        and _config['flow_type'] == 'multi_scale'
        and _config['flow_loss_weight'] > 0
    )
    print(f"[INFO] Flow supervision: {'ON' if compute_flow else 'OFF'}")
    print(f"[INFO] AMP (Mixed Precision): {'ON (fp16)' if _config['use_amp'] else 'OFF (fp32)'}")
    print(f"[INFO] Gradient clipping: {'max_norm=' + str(_config['grad_clip']) if _config['grad_clip'] > 0 else 'OFF'}")

    log_path = os.path.join(save_dir, 'train_log.csv')
    plot_path = os.path.join(save_dir, 'loss_curve.png')
    log_exists = os.path.exists(log_path)
    log_file = open(log_path, 'a', newline='', buffering=1)
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            'epoch', 'train_loss',
            'val_loss', 'val_t_err_cm', 'val_r_err_deg',
            'lr', 'epoch_time_s', 'timestamp',
        ])

    for epoch in range(starting_epoch, _config['epochs'] + 1):
        EPOCH = epoch
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{_config['epochs']}    |    {run_timestamp}")
        print(f"{'='*60}")

        total_train_loss = 0.
        epoch_start_time = time.time()
        model.train()

        local_loss = 0.
        local_t_err = 0.
        local_r_err = 0.
        LOG_INTERVAL = 50

        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()

            batch = preprocess_batch(
                sample, img_shape, _config['max_depth'],
                _config['use_reflectance'], compute_flow=compute_flow,
            )

            loss_val, t_err, r_err = train(
                model, optimizer,
                batch['ltolu'], batch['ltolv'],
                batch['rgb'], batch['lidar'],
                batch['flagt'], batch['depth'],
                sample['tr_error'], sample['rot_error'],
                loss_fn, sample['point_cloud'],
                scaler=scaler,
                grad_clip=_config['grad_clip'],
                gt_flows=batch.get('gt_flows'),
                gt_masks=batch.get('gt_masks'),
                flow_loss_weight=_config['flow_loss_weight'],
            )

            if loss_val != loss_val:
                raise ValueError("Loss is NaN")

            local_loss += loss_val
            local_t_err += t_err
            local_r_err += r_err

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                ts = datetime.now().strftime('%H:%M:%S')
                avg_loss = local_loss / LOG_INTERVAL
                avg_t = local_t_err / LOG_INTERVAL
                avg_r = local_r_err / LOG_INTERVAL
                spd = (time.time() - start_time) / batch['lidar'].shape[0]
                print(
                    f'[{ts}] Ep{epoch} '
                    f'[{batch_idx+1:4d}/{len(TrainImgLoader)}] '
                    f'loss={avg_loss:.6f}  '
                    f't_err={avg_t:.4f}cm  '
                    f'r_err={avg_r:.4f}°  '
                    f'{spd:.3f}s/sample'
                )
                _run.log_scalar("Loss", avg_loss, epoch * len(TrainImgLoader) + batch_idx)
                local_loss = 0.
                local_t_err = 0.
                local_r_err = 0.

            total_train_loss += loss_val * len(sample['rgb'])

        total_time = time.time() - epoch_start_time
        cur_train_loss = total_train_loss / len(dataset)
        ts = datetime.now().strftime('%H:%M:%S')
        print(f'[{ts}] Epoch {epoch} | train_loss={cur_train_loss:.6f} | time={total_time:.1f}s')
        _run.log_scalar("Total training loss", cur_train_loss, epoch)

        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']

        _csv_val_loss = ''
        _csv_val_t = ''
        _csv_val_r = ''

        # ===== Validation (every epoch) =====
        total_test_loss = 0.
        total_test_t = 0.
        total_test_r = 0.

        for batch_idx, sample in enumerate(TestImgLoader):
            batch = preprocess_batch(
                sample, img_shape, _config['max_depth'],
                _config['use_reflectance'], compute_flow=False,
            )

            loss_val, trasl_e, rot_e = test(
                model,
                batch['ltolu'], batch['ltolv'],
                batch['rgb'], batch['lidar'],
                batch['flagt'], batch['depth'],
                sample['tr_error'], sample['rot_error'],
                loss_fn, dataset_val.model, sample['point_cloud'],
                use_amp=_config['use_amp'],
            )

            if loss_val != loss_val:
                raise ValueError("Loss is NaN")

            total_test_t += trasl_e
            total_test_r += rot_e
            total_test_loss += loss_val * len(sample['rgb'])

        val_loss = total_test_loss / len(dataset_val)
        val_t = total_test_t / len(dataset_val)
        val_r = total_test_r / len(dataset_val)

        print("------------------------------------")
        ts = datetime.now().strftime('%H:%M:%S')
        print(f'[{ts}] [VAL] Ep{epoch}')
        print(f'Val loss = {val_loss:.6f}')
        print(f'Translation error: {val_t:.4f} cm')
        print(f'Rotation error: {val_r:.4f} °')
        print("------------------------------------")

        _run.log_scalar("Val_Loss", val_loss, epoch)
        _run.log_scalar("Val_t_error", val_t, epoch)
        _run.log_scalar("Val_r_error", val_r, epoch)

        _csv_val_loss = f'{val_loss:.6f}'
        _csv_val_t = f'{val_t:.4f}'
        _csv_val_r = f'{val_r:.4f}'

        if val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss
            BEST_VAL_EPOCH = epoch
            epochs_without_improvement = 0
            if _config['rescale_transl'] > 0:
                _run.result = val_t
            else:
                _run.result = val_r

            savefilename = (
                f'{save_dir}/checkpoint_r{_config["max_r"]:.2f}'
                f'_t{_config["max_t"]:.2f}_e{epoch}_{val_loss:.5f}_ablation.tar'
            )
            torch.save({
                'config': _config,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': cur_train_loss,
                'test_loss': val_loss,
            }, savefilename)
            print(f'Model saved as {savefilename}')
            if old_save_filename is not None and os.path.exists(old_save_filename):
                os.remove(old_save_filename)
            old_save_filename = savefilename
        else:
            epochs_without_improvement += 1
            print(f"[EarlyStop] No improvement for {epochs_without_improvement}/{PATIENCE} epochs.")

        # Save latest (for resume)
        latest_filename = f'{save_dir}/checkpoint_latest_ablation.tar'
        torch.save({
            'config': _config,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': cur_train_loss,
            'test_loss': BEST_VAL_LOSS,
        }, latest_filename)

        # CSV log
        log_writer.writerow([
            epoch, f'{cur_train_loss:.6f}',
            _csv_val_loss, _csv_val_t, _csv_val_r,
            f'{cur_lr:.2e}', f'{total_time:.1f}',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        ])

        # Plot
        try:
            df = pd.read_csv(log_path)
            valid_rows = []
            for _, row in df.iterrows():
                ep = int(row['epoch'])
                valid_rows = [r for r in valid_rows if int(r['epoch']) < ep]
                valid_rows.append(row)
            df = pd.DataFrame(valid_rows)

            plt.figure(figsize=(10, 8))
            plt.subplot(2, 1, 1)
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
            val_df = df.dropna(subset=['val_loss'])
            if not val_df.empty:
                plt.plot(val_df['epoch'], val_df['val_loss'], label='Val Loss', marker='o', color='orange')
            plt.title('Training & Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()

            plt.subplot(2, 1, 2)
            if not val_df.empty:
                plt.plot(val_df['epoch'], val_df['val_t_err_cm'], label='Val T-Error (cm)', marker='s', color='green')
                plt.plot(val_df['epoch'], val_df['val_r_err_deg'], label='Val R-Error (deg)', marker='^', color='red')
            plt.title('Validation Translation & Rotation Errors')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        except Exception as e:
            print(f"[Warning] Failed to plot: {e}")

        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"\n[EarlyStop] Triggered at epoch {epoch}. Best was epoch {BEST_VAL_EPOCH}.")
            break

    print(f'\nFull training time = {(time.time() - start_full_time) / 3600:.2f} HR')

    # Save final model
    last_filename = (
        f'{save_dir}/checkpoint_r{_config["max_r"]:.2f}'
        f'_t{_config["max_t"]:.2f}_e{epoch}_LAST_ablation.tar'
    )
    torch.save({
        'config': _config,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss': total_train_loss / len(dataset),
        'test_loss': BEST_VAL_LOSS,
    }, last_filename)
    print(f'Last epoch model saved as {last_filename}')

    log_file.close()
    return _run.result
