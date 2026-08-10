#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --job-name=queryocc-missing-points
#
# Generate the free-ray ("missing points") cache used as free-space supervision by the
# lidar-supervised arms. See tools/missing_points/README.md.
#
# CPU-only (no CUDA, no compiled extensions), so it runs in the same small venv as the
# index builder — no training container required:
#
#   scripts/create_missing_points.sh --dataset nuscenes --version v1.0-trainval
#
# On clusters without a CPU-only partition, submit it against the cheapest GPU:
#   sbatch --gpus-per-node=T4:1 scripts/create_missing_points.sh --dataset argoverse2 --split train
#
# The job is resumable: rerunning skips sweeps whose output already exists, so a timeout
# costs only the unfinished tail. Set CONTAINER to run inside an image instead of a venv.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$SCRIPT_DIR/env.local.sh" ] && source "$SCRIPT_DIR/env.local.sh"

PROJECT_ROOT=${PROJECT_ROOT:?"Set PROJECT_ROOT in scripts/env.local.sh"}
NUSC_DATASET_ROOT=${NUSC_DATASET_ROOT:-"$PROJECT_ROOT/nuscenes"}
AV2_DATASET_ROOT=${AV2_DATASET_ROOT:-"$PROJECT_ROOT/av2"}
NUSC_MISSING_POINTS_DIR=${NUSC_MISSING_POINTS_DIR:-"$PROJECT_ROOT/nuscenes_missing_points_cache"}
AV2_MISSING_POINTS_DIR=${AV2_MISSING_POINTS_DIR:-"$PROJECT_ROOT/av2_missing_points_cache"}
INFOS_VENV=${INFOS_VENV:-"$PROJECT_ROOT/venvs/infos"}
NUM_WORKERS=${NUM_WORKERS:-8}

# The tool takes a single --out_dir, so pick it (and the dataset root) from --dataset.
case " $* " in
    *" argoverse2 "*) OUT_DIR="$AV2_MISSING_POINTS_DIR" ;;
    *)                OUT_DIR="$NUSC_MISSING_POINTS_DIR" ;;
esac

# Host paths passed explicitly, so the venv route needs no bind mounts.
ARGS=(--num_workers "$NUM_WORKERS" --out_dir "$OUT_DIR"
      --nuscenes_data "$NUSC_DATASET_ROOT" --argoverse_data "$AV2_DATASET_ROOT"
      "$@")

if [ -n "${CONTAINER:-}" ] && [ ! -x "$INFOS_VENV/bin/python" ]; then
    echo "Running in container: $CONTAINER"
    apptainer exec \
        --env PYTHONNOUSERSITE=1 \
        --bind "$NUSC_DATASET_ROOT":"$NUSC_DATASET_ROOT" \
        --bind "$AV2_DATASET_ROOT":"$AV2_DATASET_ROOT" \
        --bind "$NUSC_MISSING_POINTS_DIR":"$NUSC_MISSING_POINTS_DIR" \
        --bind "$AV2_MISSING_POINTS_DIR":"$AV2_MISSING_POINTS_DIR" \
        --bind "$PWD":/queryocc --pwd /queryocc \
        "$CONTAINER" \
        python -u -m tools.missing_points.engine "${ARGS[@]}"
else
    echo "Running in venv: $INFOS_VENV"
    [ -x "$INFOS_VENV/bin/python" ] || {
        echo "ERROR: $INFOS_VENV/bin/python not found or not runnable."
        echo "       If your cluster provides Python via modules, load one first, e.g."
        echo "       module load Python/3.11"
        exit 1
    }
    PYTHONPATH="$PWD:${PYTHONPATH:-}" "$INFOS_VENV/bin/python" -u -m tools.missing_points.engine "${ARGS[@]}"
fi
