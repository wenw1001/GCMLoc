"""
build_split.py — bake the train/test split to disk from a selected polygon.

Reads the polygon saved by select_test_region.py (test_region.json or
splits.json) and each sequence's frame_index.csv (map_x, map_y), then writes
explicit train/test timestamp lists so the dataset loader does ZERO geometry —
it just checks list membership.

Output:
    iter_campus/processed/splits.json
        {polygon: [[x,y],...],
         sequences: {<seq>: {train: [ts_ns,...], test: [ts_ns,...]}}}

Run:  python build_split.py
      python build_split.py --polygon-from test_region.json
"""
import argparse
import csv
import json
import os

import numpy as np
from matplotlib.path import Path

import itri_common as C

OUT = os.path.join(C.PROC_DIR, 'splits.json')


def _frame_index(seq):
    ts, xy = [], []
    with open(os.path.join(C.proc_seq_dir(seq), 'frame_index.csv')) as f:
        for row in csv.DictReader(f):
            ts.append(int(row['img_ts_ns']))
            xy.append((float(row['map_x']), float(row['map_y'])))
    return np.array(ts, dtype=np.int64), np.array(xy, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--polygon-from', default=None,
                    help='json file holding {"polygon": [[x,y],...]} '
                         '(default: test_region.json then splits.json)')
    args = ap.parse_args()

    candidates = ([args.polygon_from] if args.polygon_from else
                  [os.path.join(C.PROC_DIR, 'test_region.json'), OUT])
    src = next((p for p in candidates if p and os.path.isfile(p)), None)
    if src is None:
        raise SystemExit('No polygon json found. Run select_test_region.py first.')
    polygon = json.load(open(src))['polygon']
    path = Path(polygon)
    print(f'[split] polygon from {src} ({len(polygon)} vertices)')

    sequences = {}
    tot_tr = tot_te = 0
    for seq in C.list_sequences():
        ts, xy = _frame_index(seq)
        inside = path.contains_points(xy)
        # dedupe (a few poses can match the same nearest image) while keeping order
        test_ts = list(dict.fromkeys(int(x) for x in ts[inside]))
        train_ts = list(dict.fromkeys(int(x) for x in ts[~inside]))
        sequences[seq] = {'train': train_ts, 'test': test_ts}
        tot_tr += len(train_ts); tot_te += len(test_ts)
        print(f'  {seq[:34]:34s} train {len(train_ts):5d} | '
              f'test {len(test_ts):5d} ({100*len(test_ts)/len(ts):.0f}%)')
    print(f'  {"TOTAL":34s} train {tot_tr:5d} | test {tot_te:5d} '
          f'({100*tot_te/(tot_tr+tot_te):.0f}%)')

    with open(OUT, 'w') as f:
        json.dump({'polygon': polygon, 'sequences': sequences}, f, indent=2)
    print(f'[split] wrote {OUT}')


if __name__ == '__main__':
    main()
