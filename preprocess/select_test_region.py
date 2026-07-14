"""
select_test_region.py — interactively draw the region whose frames become the
TEST set, by outlining a road segment on the map + trajectories.

Usage:
    conda run --no-capture-output -n CMRNet_4090 python select_test_region.py

How to use the window:
    1. Left-click to drop polygon vertices around the road segment you want as
       the test set (you can make any shape; a long thin quad along one street
       is typical).
    2. Click near the first vertex (or complete the loop) to close the polygon.
    3. Press ENTER to confirm → prints per-sequence frame counts, saves
       test_region.json and a preview PNG, then closes.
       Press ESC to clear and redraw.

Output (the split is BAKED here, so the loader does no geometry):
    iter_campus/processed/splits.json
        {polygon: [[x,y],...],
         sequences: {<seq>: {train: [ts_ns,...], test: [ts_ns,...]}}}
    iter_campus/processed/viz/test_region_selected.png

A frame is 'test' iff its pose (x, y) lies inside the polygon — a clean spatial
hold-out with no coordinate leakage (the model never sees absolute map coords).
The dataset loader just reads the train/test timestamp lists.
"""
import csv
import json
import os

import importlib

import matplotlib
# pick an interactive backend whose GUI toolkit is actually importable
# (matplotlib.use() alone does NOT verify the binding exists)
for _mod, _bk in (('PyQt5', 'Qt5Agg'), ('PySide6', 'QtAgg'),
                  ('tkinter', 'TkAgg')):
    try:
        importlib.import_module(_mod)
        matplotlib.use(_bk)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
import numpy as np
import h5py

import itri_common as C

INFO = {
    '2024-05-21-itri-p4':                 ('tab:blue',   'rainy 雨天'),
    '2024-06-12-16-28-15-P4_itri_1':      ('tab:green',  'sunny 晴天#1'),
    '2024-06-12-17-03-45-P4_itri_5':      ('tab:orange', 'sunny 晴天#2'),
    '2024-07-25-typhoon-data-collect-p4': ('tab:red',    'typhoon 颱風'),
}

OUT_JSON = os.path.join(C.PROC_DIR, 'splits.json')
OUT_PNG  = os.path.join(C.PROC_DIR, 'viz', 'test_region_selected.png')


def _load_frame_index(seq):
    """Return (ts_ns int array, xy (N,2) float array) from frame_index.csv."""
    path = os.path.join(C.proc_seq_dir(seq), 'frame_index.csv')
    ts, xy = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts.append(int(row['img_ts_ns']))
            xy.append((float(row['map_x']), float(row['map_y'])))
    return np.array(ts, dtype=np.int64), np.array(xy, dtype=np.float64)


def main():
    print(f'[backend] {matplotlib.get_backend()}')

    # load per-frame timestamp + map xy (from frame_index.csv, which is what the
    # loader keys on — guarantees the baked split matches the loader exactly)
    ts_map, traj = {}, {}
    for seq in C.list_sequences():
        ts_map[seq], traj[seq] = _load_frame_index(seq)

    # map background (subsample)
    with h5py.File(C.MAP_H5, 'r') as hf:
        M = hf['PC'].shape[1]
        idx = np.random.choice(M, min(M, 250000), replace=False)
        idx.sort()
        bg = hf['PC'][:2, idx]

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.scatter(bg[0], bg[1], s=0.2, c='lightgray', linewidths=0, zorder=1)
    for seq, (c, lab) in INFO.items():
        t = traj[seq]
        ax.plot(t[:, 0], t[:, 1], '-', color=c, lw=1.5, label=lab, zorder=2)
    ax.set_aspect('equal')
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlabel('map x (m)'); ax.set_ylabel('map y (m)')
    ax.set_title('Draw polygon around the TEST road segment.\n'
                 'click vertices → close loop → ENTER to confirm, ESC to clear')

    state = {'verts': None}

    def on_select(verts):
        state['verts'] = list(verts)

    selector = PolygonSelector(ax, on_select, useblit=True)

    def on_key(event):
        if event.key == 'enter' and state['verts'] and len(state['verts']) >= 3:
            verts = state['verts']
            path = Path(verts)
            sequences = {}
            print('\n==== TEST region selection (split baked) ====')
            total_te = total_all = 0
            for seq in C.list_sequences():
                inside = path.contains_points(traj[seq])
                ts = ts_map[seq]
                test_ts = [int(x) for x in ts[inside]]
                train_ts = [int(x) for x in ts[~inside]]
                sequences[seq] = {'train': train_ts, 'test': test_ts}
                total_te += len(test_ts); total_all += len(ts)
                print(f'  {INFO[seq][1]:13s} train {len(train_ts):5d} | '
                      f'test {len(test_ts):5d} ({100*len(test_ts)/len(ts):.0f}%)')
            print(f'  {"TOTAL":13s} test {total_te:5d} / {total_all:5d} '
                  f'({100*total_te/total_all:.0f}%)')

            os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
            with open(OUT_JSON, 'w') as f:
                json.dump({'polygon': [[float(x), float(y)] for x, y in verts],
                           'sequences': sequences}, f, indent=2)
            print(f'[saved] {OUT_JSON}  (loader reads train/test ts lists)')

            # preview png
            for seq in C.list_sequences():
                t = traj[seq]; inside = path.contains_points(t)
                ax.plot(t[inside, 0], t[inside, 1], '.', ms=3, c='red', zorder=5)
            vv = np.array(verts + [verts[0]])
            ax.plot(vv[:, 0], vv[:, 1], '-', c='red', lw=2, zorder=6)
            os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
            fig.savefig(OUT_PNG, dpi=88, bbox_inches='tight')
            print(f'[saved] {OUT_PNG}')
            print('Close the window (or it stays open for review).')

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


if __name__ == '__main__':
    main()
