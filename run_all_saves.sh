#!/bin/bash

# Batch map-saving: run Stage-1 inference with a trained mapping checkpoint
# and export the GCMLoc map point clouds (.npy) that Stage-2 localization
# training / evaluation reads.
#
# Usage:  bash run_all_saves.sh [kitti|argo|itri]
set -e

DATASET="${1:-kitti}"

if [ "$DATASET" != "kitti" ] && [ "$DATASET" != "argo" ] && [ "$DATASET" != "itri" ]; then
    echo "Usage: bash run_all_saves.sh [kitti|argo|itri]"
    exit 1
fi

echo "Dataset: ${DATASET}"

# Path to the trained Stage-1 mapping checkpoint
WEIGHTS="./checkpoints/mapping.tar"

SAVE_NAME="v2_pcl_mix"

# Canonical-camera settings: must match the preset the WEIGHTS were
# trained with (mixed-dataset training uses preset A).
USE_CANONICAL=True
CANON_PRESET="A"

# ── KITTI ──────────────────────────────────────────────────────────
if [ "$DATASET" = "kitti" ]; then

    SEQUENCES=(0 3 5 6 7 8 9)
    SAVE_ROOT="./KITTI_ODOMETRY/sequences"

    for seq in "${SEQUENCES[@]}"; do
        echo ""
        echo "============================================================"
        echo "Saving sequence: ${seq}"
        echo "============================================================"

        python train_save.py with \
            dataset=kitti \
            batch_size=8 \
            data_folder=${SAVE_ROOT} \
            test_sequence=${seq} \
            weights="${WEIGHTS}" \
            save_root=${SAVE_ROOT} \
            save_name=${SAVE_NAME} \
            use_canonical=${USE_CANONICAL} \
            canon_preset=${CANON_PRESET}

        echo "Sequence ${seq} done."
    done

    echo ""
    echo "All KITTI sequences complete: ${SEQUENCES[*]}"

# ── Argoverse ──────────────────────────────────────────────────────
elif [ "$DATASET" = "argo" ]; then

    DATA_FOLDER="./data/argoverse-tracking"

    # train split (train1-3; used for localization training)
    echo ""
    echo "============================================================"
    echo "Saving train split (train1~3)"
    echo "============================================================"

    python train_save.py with \
        dataset=argo \
        batch_size=4 \
        data_folder=${DATA_FOLDER} \
        weights="${WEIGHTS}" \
        save_root=${DATA_FOLDER} \
        save_name=${SAVE_NAME} \
        save_split=train \
        use_canonical=${USE_CANONICAL} \
        canon_preset=${CANON_PRESET}

    echo "Train split done."

    # test split (train4; used for localization validation)
    echo ""
    echo "============================================================"
    echo "Saving test split (train4)"
    echo "============================================================"

    python train_save.py with \
        dataset=argo \
        batch_size=4 \
        data_folder=${DATA_FOLDER} \
        weights="${WEIGHTS}" \
        save_root=${DATA_FOLDER} \
        save_name=${SAVE_NAME} \
        save_split=test \
        use_canonical=${USE_CANONICAL} \
        canon_preset=${CANON_PRESET}

    echo "Test split done."

    echo ""
    echo "All Argoverse splits complete."

# ── ITRI-campus (iter_campus) ──────────────────────────────────────
else

    DATA_FOLDER="./iter_campus"
    # Note: itri uses strategy-C on-the-fly submap cropping as the map source.
    #       Saved files go to {save_root}/{seq}/{save_name}/{ts}.npy and the
    #       localization side reads iter_campus/processed/sequences/<seq>/<save_name>/,
    #       so save_root must point at processed/sequences.
    SAVE_ROOT="${DATA_FOLDER}/processed/sequences"

    # test split (used for localization inference)
    echo ""
    echo "============================================================"
    echo "Saving ITRI-campus test split"
    echo "============================================================"

    python train_save.py with \
        dataset=itri \
        batch_size=4 \
        data_folder=${DATA_FOLDER} \
        weights="${WEIGHTS}" \
        save_root=${SAVE_ROOT} \
        save_name=${SAVE_NAME} \
        save_split=test \
        use_canonical=${USE_CANONICAL} \
        canon_preset=${CANON_PRESET}

    echo "ITRI-campus test split done."

    echo ""
    echo "All ITRI-campus splits complete."

fi
