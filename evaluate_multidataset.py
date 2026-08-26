"""
GCMLoc — Multi-Dataset Iterative Evaluation (KITTI / Argoverse).

Thin wrapper around evaluate.py: runs the same iterative evaluation with one shared set
of localization checkpoints (typically mixed-trained) on each requested dataset, writing
the usual per-dataset outputs plus one combined cross-dataset table.

Output:
    <results_root>/
        kitti/  per_sample_iter*.csv  module_times.csv  summary.txt
        argo/   ...
        summary_all.txt              combined cross-dataset summary (final iteration)

Example (mixed-trained checkpoints, canonical camera):
    python evaluate_multidataset.py with "datasets=['kitti','argo']" \\
        use_canonical=True canon_preset=A maps_folder=v2_pcl_mix \\
        max_r=10 max_t=2 "weight=['<iter1>.tar','<iter2>.tar','<iter3>.tar']"
"""

import os

from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds

from evaluate import run_evaluation

SETTINGS.DISCOVER_DEPENDENCIES = "none"
SETTINGS.DISCOVER_SOURCES = "none"
ex = Experiment("GCMLoc-eval-multidataset")
ex.captured_out_filter = apply_backspaces_and_linefeeds

DEFAULT_DATA_FOLDERS = {
    'kitti': './KITTI_ODOMETRY/sequences',
    'argo': './data/argoverse-tracking',
}


@ex.config
def config():
    datasets = ['kitti', 'argo']    # datasets to evaluate, in order (list or 'kitti,argo')
    weight = None                   # REQUIRED: shared list of *_loc.tar checkpoints, one per iteration
    split = 'test'                  # evaluated split
    maps_folder = 'v2_pcl'          # distilled-map folder (train_save.py save_name), shared default
    max_t = 2.                      # initial-offset range of the test_RT csv (metres)
    max_r = 10.                     # initial-offset range of the test_RT csv (degrees)
    max_depth = 100.                # depth normalisation constant
    num_worker = 3                  # DataLoader workers
    use_canonical = False           # warp to the canonical camera (checkpoints trained with dataset=mixed)
    canon_preset = 'A'              # must match the checkpoints' training preset
    results_root = None             # parent output dir (default: eval_multidataset_<first_ckpt>/ next to the checkpoint)
    method = None                   # label of the timing_records.csv rows
    timing_csv = None               # override path of timing_records.csv

    # per-dataset overrides (None -> shared default above)
    kitti_data_folder = None
    kitti_maps_folder = None
    kitti_test_sequence = 0         # KITTI evaluated sequence
    argo_data_folder = None
    argo_maps_folder = None


@ex.automain
def main(_config):
    cfg = dict(_config)
    weight = cfg['weight']
    if not weight:
        raise ValueError("weight=<list of localization checkpoints> is required")
    if isinstance(weight, str):
        weight = [weight]
    datasets = cfg['datasets']
    if isinstance(datasets, str):
        datasets = [d.strip() for d in datasets.split(',') if d.strip()]
    for d in datasets:
        if d not in DEFAULT_DATA_FOLDERS:
            raise ValueError(f"unknown dataset {d!r}; choose from {list(DEFAULT_DATA_FOLDERS)}")

    results_root = cfg['results_root']
    if results_root is None:
        ckpt_name = os.path.splitext(os.path.basename(weight[0]))[0]
        results_root = os.path.join(os.path.dirname(weight[0]), f"eval_multidataset_{ckpt_name}")
    os.makedirs(results_root, exist_ok=True)

    all_stats = []
    for name in datasets:
        print(f"\n{'#' * 70}\n# Dataset: {name}\n{'#' * 70}")
        stats = run_evaluation({
            'dataset': name,
            'data_folder': cfg[f'{name}_data_folder'] or DEFAULT_DATA_FOLDERS[name],
            'maps_folder': cfg[f'{name}_maps_folder'] or cfg['maps_folder'],
            'test_sequence': cfg['kitti_test_sequence'],
            'split': cfg['split'], 'weight': weight,
            'max_t': cfg['max_t'], 'max_r': cfg['max_r'], 'max_depth': cfg['max_depth'],
            'num_worker': cfg['num_worker'],
            'use_canonical': cfg['use_canonical'], 'canon_preset': cfg['canon_preset'],
            'results_dir': os.path.join(results_root, name),
            'method': cfg['method'], 'timing_csv': cfg['timing_csv'],
        })
        all_stats.append(stats)

    # ---------- combined summary ----------
    path = os.path.join(results_root, 'summary_all.txt')
    with open(path, 'w') as f:
        f.write(f"GCMLoc — Multi-Dataset Evaluation (final of {len(weight)} iterations)\n")
        f.write(f"Weights     : {weight}\n")
        f.write(f"Canonical   : {cfg['use_canonical']} (preset {cfg['canon_preset']})\n")
        f.write(f"Initial RT  : max_r={cfg['max_r']:g} deg  max_t={cfg['max_t']:g} m  split={cfg['split']}\n\n")
        header = (f"{'dataset':<8} {'seq':<12} {'N':>6} {'t_mean(cm)':>11} {'t_med(cm)':>10} "
                  f"{'r_mean(deg)':>11} {'r_med(deg)':>10} {'init_t(cm)':>10} {'init_r(deg)':>11} "
                  f"{'infer(ms)':>10} {'fps':>7}")
        f.write(header + "\n" + "-" * len(header) + "\n")
        for s in all_stats:
            f.write(f"{s['dataset']:<8} {s['seq']:<12} {s['n_samples']:>6} "
                    f"{s['mean_t_cm']:>11.4f} {s['med_t_cm']:>10.4f} "
                    f"{s['mean_r_deg']:>11.4f} {s['med_r_deg']:>10.4f} "
                    f"{s['init_t_cm']:>10.2f} {s['init_r_deg']:>11.2f} "
                    f"{s['infer_ms']:>10.2f} {s['fps']:>7.2f}\n")
    print(f"\n[Saved] combined summary → {path}")
    print(open(path).read())
    return min(s['med_t_cm'] for s in all_stats)
