"""
build_poses.py — per-frame camera pose (cam_T_map) for every front image, by
INTERPOLATING the 100 Hz tf trajectory to each image's exact timestamp.

Why interpolate (not poses.txt, not sync_record):
  * poses.txt is self-generated and was misaligned for two sequences
    (itri_1 ~0.9 m, typhoon ~1.5 m error).
  * sync_record_original.yaml pairs each image to a tf up to 80 ms away
    (~0.25 m looser than necessary).
  * The tf dumps are 100 Hz (~10 ms spacing). An image timestamp T_img falls
    between two tf samples T0,T1; the pose at T_img is obtained by linear
    interpolation of position + SLERP of orientation. Over a ~10 ms bracket the
    motion is near-straight and <1°, so the interpolation error is sub-cm, well
    below the localization's own noise.

Coordinate chain (verified):
    tf extrinsic E           = base_link_T_map   (map→base_link)
    map_T_base   = inv(E)                          (vehicle pose in map)
    Tr_velo_to_cam           = cam_T_base_link    (== inv(tf_static front cam))
    cam_T_map = Tr_velo_to_cam @ inv(map_T_base_interp)

Frames whose timestamp is outside the tf range, or whose bracketing tf gap is
larger than --max-gap-ms (a localization dropout), are skipped.

Output per sequence (iter_campus/processed/sequences/<seq>/):
    poses_torch/<ts_ns>.npy   → 4x4 float32 cam_T_map
    frame_index.csv           → idx,img_ts_ns,bracket_ms,map_x,map_y

Run:  python build_poses.py
      python build_poses.py --seq <name> --max-gap-ms 30
"""
import argparse
import csv
import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import itri_common as C


def _load_trajectory(seq):
    """Return (ts_ns sorted int64 (N,), pos (N,3), Rotation stack) for
    map_T_base (vehicle pose in map) from every tf/map.base_link/*.json."""
    files = glob.glob(os.path.join(C.src_seq_dir(seq), 'tf/map.base_link/*.json'))
    ts, pos, quat = [], [], []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if not stem.isdigit():
            continue
        E = C.load_tf_extrinsic(seq, os.path.join('tf/map.base_link',
                                                  os.path.basename(f)))
        R_mb = E[:3, :3].T                 # map_T_base rotation
        t_mb = -R_mb @ E[:3, 3]            # map_T_base translation (vehicle pos)
        ts.append(int(stem)); pos.append(t_mb)
        quat.append(Rotation.from_matrix(R_mb).as_quat())
    ts = np.array(ts, dtype=np.int64)
    order = np.argsort(ts)
    ts = ts[order]
    pos = np.array(pos)[order]
    rots = Rotation.from_quat(np.array(quat)[order])
    # drop duplicate timestamps (Slerp requires strictly increasing)
    keep = np.concatenate([[True], np.diff(ts) > 0])
    return ts[keep], pos[keep], rots[keep]


def build_sequence(seq, max_gap_ms):
    T_cb = C.load_tr_velo_to_cam(seq)                 # cam_T_base_link
    tf_ts, tf_pos, tf_rot = _load_trajectory(seq)
    slerp = Slerp(tf_ts.astype(np.float64), tf_rot)

    out_dir = os.path.join(C.proc_seq_dir(seq), 'poses_torch')
    os.makedirs(out_dir, exist_ok=True)

    imgs = C.list_front_images(seq)
    rows = []
    skip_range = skip_gap = 0
    for ts_ns, _ in imgs:
        if ts_ns < tf_ts[0] or ts_ns > tf_ts[-1]:
            skip_range += 1
            continue
        hi = int(np.searchsorted(tf_ts, ts_ns))
        lo = max(hi - 1, 0)
        if tf_ts[hi] == ts_ns:
            lo = hi
        bracket_ms = (tf_ts[min(hi, len(tf_ts) - 1)] - tf_ts[lo]) / 1e6
        if bracket_ms > max_gap_ms:
            skip_gap += 1
            continue

        # interpolate vehicle pose map_T_base at the image time
        if lo == hi:
            pos = tf_pos[lo]
        else:
            a = (ts_ns - tf_ts[lo]) / (tf_ts[hi] - tf_ts[lo])
            pos = (1 - a) * tf_pos[lo] + a * tf_pos[hi]
        R_mb = slerp([float(ts_ns)]).as_matrix()[0]

        # cam_T_map = Tr_velo_to_cam @ inv(map_T_base)
        base_T_map = np.eye(4)
        base_T_map[:3, :3] = R_mb.T
        base_T_map[:3, 3] = -R_mb.T @ pos
        cam_T_map = T_cb @ base_T_map
        np.save(os.path.join(out_dir, f'{ts_ns}.npy'), cam_T_map.astype(np.float32))
        rows.append((len(rows), ts_ns, f'{bracket_ms:.1f}',
                     f'{pos[0]:.3f}', f'{pos[1]:.3f}'))

    with open(os.path.join(C.proc_seq_dir(seq), 'frame_index.csv'), 'w',
              newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'img_ts_ns', 'bracket_ms', 'map_x', 'map_y'])
        w.writerows(rows)

    print(f'[poses] {seq}: kept {len(rows)}/{len(imgs)}  '
          f'(skip out-of-range {skip_range}, gap>{max_gap_ms}ms {skip_gap})  '
          f'tf samples={len(tf_ts)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default=None, help='single sequence (default: all)')
    ap.add_argument('--max-gap-ms', type=float, default=30.0,
                    help='skip a frame if its bracketing tf gap exceeds this '
                         '(localization dropout). 100Hz tf → normal gap ~10ms')
    args = ap.parse_args()
    for seq in ([args.seq] if args.seq else C.list_sequences()):
        build_sequence(seq, args.max_gap_ms)
    print('[poses] done.')


if __name__ == '__main__':
    main()
