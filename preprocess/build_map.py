"""
build_map.py — merge the 205 voxel-filtered submaps into a single global
point-cloud map.h5, downsampled to a coarser voxel size.

Source : SRC/pcd_map/vg05_filtered/*.npy
         structured arrays with fields ('centroids' f8[3], 'intensities' f4,
         'counts' i4), in the map frame (same frame as kitti_format/poses.txt).
         Full resolution is ~119M points at 5cm — too large to hold in RAM and
         crop per frame, so we downsample.

Output : iter_campus/processed/map.h5
            PC         (4, M) float32  → rows [x, y, z, 1] in the map frame
            intensity  (1, M) float32

The h5 layout matches the Argoverse map.h5 so the dataset loader can reuse the
same transform/crop logic.

Run:  python build_map.py --voxel 0.15
"""
import argparse
import glob
import os

import h5py
import numpy as np

import itri_common as C


def voxel_downsample(xyz, inten, voxel):
    """Keep one representative point per voxel cell (centroids already smoothed).

    xyz   : (N,3) float
    inten : (N,)  float
    """
    keys = np.floor(xyz / voxel).astype(np.int64)
    # unique voxel cells; return_index picks the first point in each cell
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx], inten[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--voxel', type=float, default=0.15,
                    help='output voxel size in metres (default 0.15)')
    ap.add_argument('--src', default=os.path.join(C.SRC_ROOT,
                    'pcd_map', 'vg05_filtered'),
                    help='directory of *.npy submaps')
    ap.add_argument('--out', default=C.MAP_H5)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, '*.npy')))
    if not files:
        raise SystemExit(f'No .npy submaps found in {args.src}')
    print(f'[map] {len(files)} submaps, voxel={args.voxel} m')

    xyz_parts, int_parts = [], []
    raw_total = 0
    for i, f in enumerate(files):
        a = np.load(f, allow_pickle=False)
        xyz = np.ascontiguousarray(a['centroids']).astype(np.float64).reshape(-1, 3)
        inten = np.ascontiguousarray(a['intensities']).astype(np.float32).reshape(-1)
        raw_total += xyz.shape[0]
        # downsample within each submap (submaps are spatially disjoint cells,
        # so per-submap downsampling == global downsampling at boundaries)
        xyz_d, int_d = voxel_downsample(xyz, inten, args.voxel)
        xyz_parts.append(xyz_d.astype(np.float32))
        int_parts.append(int_d.astype(np.float32))
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            kept = sum(p.shape[0] for p in xyz_parts)
            print(f'  [{i+1:3d}/{len(files)}] raw={raw_total:,} kept={kept:,}')

    xyz_all = np.concatenate(xyz_parts, axis=0)            # (M,3)
    int_all = np.concatenate(int_parts, axis=0)            # (M,)
    M = xyz_all.shape[0]

    PC = np.ones((4, M), dtype=np.float32)
    PC[:3, :] = xyz_all.T
    intensity = int_all.reshape(1, M).astype(np.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with h5py.File(args.out, 'w') as hf:
        hf.create_dataset('PC', data=PC, compression='gzip', compression_opts=4)
        hf.create_dataset('intensity', data=intensity,
                          compression='gzip', compression_opts=4)
        hf.attrs['voxel'] = args.voxel
        hf.attrs['num_points'] = M
        hf.attrs['frame'] = 'map (same as kitti_format/poses.txt)'

    print(f'[map] raw {raw_total:,} → kept {M:,} points '
          f'({100.0*M/raw_total:.1f}%)')
    print(f'[map] bbox min {xyz_all.min(0)}  max {xyz_all.max(0)}')
    print(f'[map] wrote {args.out}')


if __name__ == '__main__':
    main()
