"""
build_calib.py — write the pinhole intrinsics for each sequence's front camera.

IMPORTANT: the stored front-camera images (camera/...f_hdr.h265/*.jpg) are
ALREADY rectified to a pinhole model — they have straight edges and a curved
gray fill at the bottom left over from an earlier fisheye→pinhole rectification.
So we do NOT undistort them again; we just record the matching intrinsics, which
are the ROS projection_matrix P (P[:3,:3]), and let the dataset loader read the
original images directly from iter_campus/raw/sequences/<seq>/camera_front.

Output per sequence (iter_campus/processed/sequences/<seq>/):
    pinhole_calib.json  → {fx, fy, cx, cy, width, height, source}

Run:  python build_calib.py
"""
import argparse
import json
import os

import itri_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default=None, help='single sequence (default: all)')
    args = ap.parse_args()

    seqs = [args.seq] if args.seq else C.list_sequences()
    for seq in seqs:
        fx, fy, cx, cy, W, H = C.load_front_pinhole(seq)
        out = {
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'width': W, 'height': H,
            'source': 'projection_matrix P (images already rectified pinhole)',
            'image_dir': 'iter_campus/raw/sequences/%s/camera_front' % seq,
        }
        os.makedirs(C.proc_seq_dir(seq), exist_ok=True)
        with open(os.path.join(C.proc_seq_dir(seq), 'pinhole_calib.json'),
                  'w') as f:
            json.dump(out, f, indent=2)
        print(f'[calib] {seq}: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} '
              f'({W}x{H})')
    print('[calib] done.')


if __name__ == '__main__':
    main()
