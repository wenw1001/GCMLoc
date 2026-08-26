#!/bin/bash
# Batch distilled-map saving: run train_save.py (Stage-1 inference with a trained mapping
# checkpoint) over all sequences / splits of one dataset. Output:
#     <save_root>/<sequence>/<SAVE_NAME>/<frame>.npy      (5000 x 3 float32)
# which the localization stage reads with maps_folder=<SAVE_NAME>.
#
# Usage:
#     WEIGHTS=<mapping_checkpoint>.tar bash run_all_saves.sh [kitti|argo]
#
# Optional environment variables (defaults in brackets):
#     SAVE_NAME      output folder name                               [v2_pcl]
#     USE_CANONICAL  True for a checkpoint trained with dataset=mixed   [False]
#     CANON_PRESET   canonical camera preset, must match training      [A]
#     KITTI_ROOT     [./KITTI_ODOMETRY/sequences]   KITTI_SEQS  ["0 3 5 6 7 8 9"]
#     ARGO_ROOT      [./data/argoverse-tracking]
#     SAVE_ROOT      override the output root (default: the dataset root, where the
#                    localization loaders expect the maps)
#     BATCH_SIZE     [8 for kitti, 4 otherwise]
#
# Example (mixed-trained checkpoint, both datasets):
#     export WEIGHTS=checkpoints/mixed/mixed/<run>/checkpoint_..._mapping.tar
#     export SAVE_NAME=v2_pcl_mix USE_CANONICAL=True CANON_PRESET=A
#     bash run_all_saves.sh kitti && bash run_all_saves.sh argo
set -e

DATASET="${1:-kitti}"
if [ "$DATASET" != "kitti" ] && [ "$DATASET" != "argo" ]; then
    echo "Usage: WEIGHTS=<mapping_checkpoint>.tar bash run_all_saves.sh [kitti|argo]"
    exit 1
fi
if [ -z "$WEIGHTS" ] || [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: set WEIGHTS=<path to a train_mapping.py checkpoint> (got '${WEIGHTS}')"
    exit 1
fi

SAVE_NAME="${SAVE_NAME:-v2_pcl}"
USE_CANONICAL="${USE_CANONICAL:-False}"
CANON_PRESET="${CANON_PRESET:-A}"
KITTI_ROOT="${KITTI_ROOT:-./KITTI_ODOMETRY/sequences}"
KITTI_SEQS="${KITTI_SEQS:-0 3 5 6 7 8 9}"
ARGO_ROOT="${ARGO_ROOT:-./data/argoverse-tracking}"

echo "Dataset: ${DATASET}   weights: ${WEIGHTS}   save_name: ${SAVE_NAME}   canonical: ${USE_CANONICAL} (${CANON_PRESET})"

COMMON="weights=${WEIGHTS} save_name=${SAVE_NAME} use_canonical=${USE_CANONICAL} canon_preset=${CANON_PRESET}"

if [ "$DATASET" = "kitti" ]; then
    # one run per sequence: save_split=test saves sequence `test_sequence`
    for seq in ${KITTI_SEQS}; do
        echo; echo "========== KITTI sequence ${seq} =========="
        python train_save.py with dataset=kitti data_folder=${KITTI_ROOT} test_sequence=${seq} \
            save_split=test save_root=${SAVE_ROOT:-${KITTI_ROOT}} batch_size=${BATCH_SIZE:-8} ${COMMON}
    done
    echo; echo "All KITTI sequences done: ${KITTI_SEQS}"

else
    # train split (train1-3, localization training) and test split (train4, validation)
    for split in train test; do
        echo; echo "========== Argoverse split ${split} =========="
        python train_save.py with dataset=argo data_folder=${ARGO_ROOT} save_split=${split} \
            save_root=${SAVE_ROOT:-${ARGO_ROOT}} batch_size=${BATCH_SIZE:-4} ${COMMON}
    done
    echo; echo "All Argoverse splits done."

fi
