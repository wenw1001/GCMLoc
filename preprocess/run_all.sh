#!/usr/bin/env bash
# run_all.sh — full ITRI-campus preprocessing pipeline.
#
# Produces, under iter_campus/:
#   raw/        symlinks to the source data we use
#   processed/  map.h5 + per-sequence {image_undistorted, poses_torch,
#               pinhole_calib.json, frame_index.csv}
#
# Usage:
#   ./run_all.sh                 # everything, all sequences
#   VOXEL=0.2 BALANCE=0.5 ./run_all.sh
#   ENV=<your_conda_env> ./run_all.sh
set -e

ENV="${ENV:-gcmloc}"
RUN="conda run --no-capture-output -n ${ENV} python"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

VOXEL="${VOXEL:-0.15}"

echo "==> 1/4 build symlink tree (iter_campus/raw)"
$RUN build_links.py

echo "==> 2/4 build global map.h5 (voxel=${VOXEL}; for verification/strategy-A)"
$RUN build_map.py --voxel "$VOXEL"

echo "==> 3/4 write pinhole calib (P; images already rectified — no undistort)"
$RUN build_calib.py

echo "==> 4/4 build per-frame cam_T_map poses"
$RUN build_poses.py

echo "==> verification overlays (first sequence)"
SEQ=$($RUN -c "import itri_common as C; print(C.list_sequences()[0])")
$RUN verify_overlay.py --seq "$SEQ" --num 6

echo "Done. Inspect iter_campus/processed/sequences/${SEQ}/overlay/*.jpg"
