#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --time 04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --job-name=queryocc-infos
#
# Precompute the QueryOcc dataset index so training jobs don't rebuild it.
#
# This is CPU-only work (it walks the nuScenes database) and needs only numpy,
# pyquaternion and nuscenes-devkit — tools/create_queryocc_infos.py deliberately
# imports no torch. So it runs in a small venv, with no training container
# required:
#
#   INFOS_VENV=$PROJECT_ROOT/venvs/infos scripts/create_infos.sh --version full
#
# On clusters without a CPU-only partition, submit it against the cheapest GPU:
#   sbatch --gpus-per-node=T4:1 scripts/create_infos.sh --version full
#
# Set CONTAINER to run inside an image instead of a venv.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$SCRIPT_DIR/env.local.sh" ] && source "$SCRIPT_DIR/env.local.sh"

PROJECT_ROOT=${PROJECT_ROOT:?"Set PROJECT_ROOT in scripts/env.local.sh"}
NUSC_DATASET_ROOT=${NUSC_DATASET_ROOT:-"$PROJECT_ROOT/nuscenes"}
NUSC_INFOS_ROOT=${NUSC_INFOS_ROOT:-"$PROJECT_ROOT/infos"}
INFOS_VENV=${INFOS_VENV:-"$PROJECT_ROOT/venvs/infos"}

# Host paths are passed explicitly rather than relying on the container layout, so
# the venv route needs no bind mounts at all.
ARGS=(--data "$NUSC_DATASET_ROOT" --infos-dir "$NUSC_INFOS_ROOT" "$@")

if [ -n "${CONTAINER:-}" ] && [ -z "${INFOS_VENV_ONLY:-}" ] && [ ! -x "$INFOS_VENV/bin/python" ]; then
    echo "Running in container: $CONTAINER"
    apptainer exec \
        --env PYTHONNOUSERSITE=1 \
        --bind "$NUSC_DATASET_ROOT":"$NUSC_DATASET_ROOT" \
        --bind "$NUSC_INFOS_ROOT":"$NUSC_INFOS_ROOT" \
        --bind "$PWD":/queryocc \
        --pwd /queryocc \
        "$CONTAINER" \
        python -u -m tools.create_queryocc_infos "${ARGS[@]}"
else
    echo "Running in venv: $INFOS_VENV"
    [ -x "$INFOS_VENV/bin/python" ] || {
        echo "ERROR: $INFOS_VENV/bin/python not found or not runnable."
        echo "       If your cluster provides Python via modules, load one first, e.g."
        echo "       module load Python/3.11"
        exit 1
    }
    PYTHONPATH="$PWD:${PYTHONPATH:-}" "$INFOS_VENV/bin/python" -u -m tools.create_queryocc_infos "${ARGS[@]}"
fi
