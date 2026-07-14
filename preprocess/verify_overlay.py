"""
verify_overlay.py — sanity-check the pose / calib / map chain by projecting the
global map onto the undistorted images and saving coloured overlays.

If the chain is correct, projected map points land on the real structures
(road, buildings, poles) in the image. If they are clearly shifted/rotated,
re-run build_poses.py with --pose-frame camera (or the chain needs fixing).

Run:  python verify_overlay.py --seq <name> --num 6
Outputs: iter_campus/processed/sequences/<seq>/overlay/<ts_ns>.jpg

Use:
cd preprocess
conda run --no-capture-output -n CMRNet_4090 python verify_overlay.py \
  --seq <序列名> --num 50 --stride 60 --point-size 1 --alpha 0.55 --zmax 25 --xabs 15

"""
import argparse
import json
import os

import cv2
import h5py
import numpy as np

import itri_common as C


def load_map(path):
    with h5py.File(path, 'r') as hf:
        PC = hf['PC'][:]          # (4, M)
    return PC.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--num', type=int, default=6, help='frames to render')
    ap.add_argument('--stride', type=int, default=0,
                    help='pick every Nth frame (0 = spread evenly)')
    ap.add_argument('--map', default=C.MAP_H5)
    ap.add_argument('--point-size', type=int, default=2)
    ap.add_argument('--alpha', type=float, default=0.5,
                    help='point opacity 0..1 (lower = more transparent, '
                         'background more visible). Default 0.5')
    ap.add_argument('--zmin', type=float, default=1.0, help='min depth (m)')
    ap.add_argument('--zmax', type=float, default=80.0,
                    help='max depth (m). Use a small value (e.g. 25) for a '
                         'crisp near-range alignment check')
    ap.add_argument('--xabs', type=float, default=40.0,
                    help='horizontal half-width crop (m)')
    ap.add_argument('--yabs', type=float, default=30.0,
                    help='vertical half-range crop (m)')
    args = ap.parse_args()

    seq = args.seq
    proc = C.proc_seq_dir(seq)
    with open(os.path.join(proc, 'pinhole_calib.json')) as f:
        cal = json.load(f)
    Kp = np.array([[cal['fx'], 0, cal['cx']],
                   [0, cal['fy'], cal['cy']],
                   [0, 0, 1]], dtype=np.float64)

    import csv
    with open(os.path.join(proc, 'frame_index.csv')) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f'no frames in {proc}/frame_index.csv')

    if args.stride > 0:
        picks = rows[::args.stride][:args.num]
    else:
        step = max(1, len(rows) // args.num)
        picks = rows[::step][:args.num]

    print(f'[overlay] loading map {args.map}')
    PC = load_map(args.map)        # (4, M) map frame

    out_dir = os.path.join(proc, 'overlay')
    os.makedirs(out_dir, exist_ok=True)

    for r in picks:
        ts = r['img_ts_ns']
        img_path = C.front_image_path(seq, ts)   # original rectified image
        pose_path = os.path.join(proc, 'poses_torch', f'{ts}.npy')
        if not (os.path.exists(img_path) and os.path.exists(pose_path)):
            print(f'  [skip] missing img/pose for {ts}')
            continue

        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        cam_T_map = np.load(pose_path).astype(np.float64)   # (4,4)

        cam = cam_T_map @ PC                # (4, M), optical frame
        x, y, z = cam[0], cam[1], cam[2]
        m = ((z > args.zmin) & (z < args.zmax) &
             (np.abs(x) < args.xabs) & (np.abs(y) < args.yabs))
        xc, yc, zc = x[m], y[m], z[m]

        u = (Kp[0, 0] * xc / zc + Kp[0, 2])
        v = (Kp[1, 1] * yc / zc + Kp[1, 2])
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, zc = u[inb].astype(int), v[inb].astype(int), zc[inb]

        # colour by depth (near=red, far=blue)
        d = np.clip(zc / args.zmax, 0, 1)
        colors = cv2.applyColorMap((d * 255).astype(np.uint8),
                                   cv2.COLORMAP_JET).reshape(-1, 3)

        # draw points on a separate layer, then alpha-blend only where points
        # were drawn so the background image stays crisp elsewhere.
        layer = img.copy()
        if args.point_size <= 1:
            # vectorized single-pixel draw (fast — for batch rendering)
            layer[v, u] = colors
        else:
            for ui, vi, ci in zip(u, v, colors):
                cv2.circle(layer, (ui, vi), args.point_size,
                           (int(ci[0]), int(ci[1]), int(ci[2])), -1)
        mask = np.any(layer != img, axis=2)
        blended = cv2.addWeighted(img, 1.0 - args.alpha, layer, args.alpha, 0)
        img[mask] = blended[mask]

        out_path = os.path.join(out_dir, f'{ts}.jpg')
        cv2.imwrite(out_path, img)
        print(f'  {ts}: {inb.sum():6d} pts → {out_path}')

    print(f'[overlay] done → {out_dir}')


if __name__ == '__main__':
    main()
