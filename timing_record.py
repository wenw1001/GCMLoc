"""
Shared helper for recording inference-efficiency timings.

Both evaluate.py (Stage-2 localization) and
train_save_eachtime.py (Stage-1 mapping) call `record_timing()` after a run
to append one structured row to a CSV, so you no longer have to copy
numbers out of stdout by hand.

CSV columns:
    timestamp      ISO time the row was written
    method         human label used as a table column (e.g. CMRNet, Ours-S)
    dataset        kitti | argo | itri
    stage          loc | map   (which script produced the row)
    seq            test sequence id
    split          train | val | test
    n_samples      number of samples averaged over
    preprocess_ms  mean pre-process time per sample (ms)
    inference_ms   mean inference time per sample (ms)
    total_ms       preprocess_ms + inference_ms
    mod_*_ms       per-module mean inference time per sample (ms), see
                   MODULE_DISPLAY below; these sum to inference_ms
    weights        checkpoint name(s) (basename, ';'-joined)
    note           free-form note
"""

import csv
import os
from datetime import datetime

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'timing_records.csv')

# Canonical per-module breakdown of the inference time.
# (csv_key, display_name) — display_name is used by the LaTeX table generator.
MODULE_DISPLAY = [
    ('rgb_backbone',       'RGB Backbone'),
    ('cross_modal_fusion', 'CrossModalFusion'),
    ('depth_backbone',     'Depth Backbone'),
    ('gcm_flow',           'GCM-Flow'),
    ('pose_regression',    'Pose Regression'),
    ('others',             'Others'),
]
MODULE_KEYS = [k for k, _ in MODULE_DISPLAY]
MODULE_COLUMNS = [f'mod_{k}_ms' for k in MODULE_KEYS]

FIELDS = ['timestamp', 'method', 'dataset', 'stage', 'seq', 'split',
          'n_samples', 'preprocess_ms', 'inference_ms', 'total_ms'] \
         + MODULE_COLUMNS + ['weights', 'note']


def _fmt_weights(weights):
    if weights is None:
        return ''
    if isinstance(weights, (list, tuple)):
        return ';'.join(os.path.basename(str(w)) for w in weights if w)
    return os.path.basename(str(weights))


def record_timing(method, dataset, preprocess_ms, inference_ms,
                  modules=None, stage='loc', seq=None, split=None,
                  n_samples=None, weights=None, note=None, csv_path=None):
    """Append one timing row to the records CSV and return total_ms.

    preprocess_ms / inference_ms are means *per sample* in milliseconds.
    `modules` is an optional dict keyed by MODULE_KEYS giving the per-module
    mean inference ms per sample; missing keys are treated as 0.  If 'others'
    is not supplied it is derived as inference_ms - sum(other modules).
    """
    csv_path = csv_path or DEFAULT_CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    preprocess_ms = float(preprocess_ms)
    inference_ms = float(inference_ms)
    total_ms = preprocess_ms + inference_ms

    modules = dict(modules) if modules else {}
    if 'others' not in modules:
        named = sum(float(modules.get(k, 0.0))
                    for k in MODULE_KEYS if k != 'others')
        modules['others'] = max(inference_ms - named, 0.0)

    row = {
        'timestamp':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method':        method,
        'dataset':       dataset,
        'stage':         stage,
        'seq':           '' if seq is None else seq,
        'split':         '' if split is None else split,
        'n_samples':     '' if n_samples is None else n_samples,
        'preprocess_ms': round(preprocess_ms, 3),
        'inference_ms':  round(inference_ms, 3),
        'total_ms':      round(total_ms, 3),
        'weights':       _fmt_weights(weights),
        'note':          '' if note is None else note,
    }
    for k in MODULE_KEYS:
        row[f'mod_{k}_ms'] = round(float(modules.get(k, 0.0)), 3)

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[timing] recorded: method={method} dataset={dataset} "
          f"pre={preprocess_ms:.2f}ms infer={inference_ms:.2f}ms "
          f"total={total_ms:.2f}ms -> {csv_path}")
    return total_ms
