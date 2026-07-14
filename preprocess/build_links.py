"""
build_links.py — create iter_campus/raw/ as a symlink view of *only* the
source data this research actually uses.

This makes the data dependency explicit: everything under iter_campus/raw is a
link back into the source dataset root (SRC_ROOT); nothing is copied.

    iter_campus/raw/
        pcd_map            -> SRC/pcd_map                 (global map source)
        submaps_config.json-> SRC/submaps_config.json
        sequences/<seq>/
            camera_front   -> .../camera/...f_hdr.h265    (front images)
            calib_front.json-> .../calib/camera/...f.json (fisheye intrinsics)
            kitti_format   -> .../kitti_format            (poses/times/calib)

Run:  python build_links.py
"""
import argparse
import os

import itri_common as C


def _link(target, link_path):
    """Create/refresh a symlink at link_path → target (idempotent)."""
    if not os.path.exists(target):
        print(f'  [skip] source missing: {target}')
        return
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    if os.path.islink(link_path) or os.path.exists(link_path):
        if os.path.islink(link_path):
            os.unlink(link_path)
        else:
            print(f'  [skip] non-link exists: {link_path}')
            return
    os.symlink(target, link_path)
    print(f'  {os.path.relpath(link_path, C.DEST_ROOT)} -> {target}')


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    print(f'[links] dest: {C.RAW_DIR}')

    # Global map + config (shared across sequences)
    _link(os.path.join(C.SRC_ROOT, 'pcd_map'),
          os.path.join(C.RAW_DIR, 'pcd_map'))
    _link(os.path.join(C.SRC_ROOT, 'submaps_config.json'),
          os.path.join(C.RAW_DIR, 'submaps_config.json'))

    # Per-sequence: front camera, its calib, and the kitti_format poses
    for seq in C.list_sequences():
        seq_src = C.src_seq_dir(seq)
        seq_dst = os.path.join(C.RAW_DIR, 'sequences', seq)
        print(f'[links] sequence: {seq}')
        _link(os.path.join(seq_src, C.FRONT_CAM_REL),
              os.path.join(seq_dst, 'camera_front'))
        _link(os.path.join(seq_src, C.FRONT_CALIB_REL),
              os.path.join(seq_dst, 'calib_front.json'))
        _link(os.path.join(seq_src, 'kitti_format'),
              os.path.join(seq_dst, 'kitti_format'))

    print('[links] done.')


if __name__ == '__main__':
    main()
