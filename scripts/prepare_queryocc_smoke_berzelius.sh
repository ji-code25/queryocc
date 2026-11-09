#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=berzelius-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/proj/agp/users/%u/logs/%j.out
#SBATCH --account=berzelius-2026-289
#SBATCH --job-name=queryocc_prep
#

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/env.local.sh" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/scripts"
else
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
if [[ ! -f "$SCRIPT_DIR/env.local.sh" ]]; then
    echo "Missing $SCRIPT_DIR/env.local.sh; copy env.local.example and fill in the site paths" >&2
    exit 2
fi
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.local.sh"
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

: "${PROJECT_ROOT:?Set PROJECT_ROOT in scripts/env.local.sh}"
CONTAINER=${CONTAINER:-"$PROJECT_ROOT/containers/queryocc.sif"}
NUSC_DATASET_ROOT=${NUSC_DATASET_ROOT:-"$PROJECT_ROOT/nuscenes"}
NUSC_INFOS_ROOT=${NUSC_INFOS_ROOT:-"$PROJECT_ROOT/infos"}
NUSC_OCC3D_DATASET_ROOT=${NUSC_OCC3D_DATASET_ROOT:-"$PROJECT_ROOT/occ3d-nuscenes"}
PSEUDO_LABEL_ROOT=${PSEUDO_LABEL_ROOT:-"$PROJECT_ROOT/pseudo_labels/bevdetv2-nuscenes"}
PSEUDO_SEMANTICS_ROOT=${PSEUDO_SEMANTICS_ROOT:-"$PSEUDO_LABEL_ROOT/semantics"}
PSEUDO_DEPTH_ROOT=${PSEUDO_DEPTH_ROOT:-"$PSEUDO_LABEL_ROOT/depth"}
QUERYOCC_SMOKE_ROOT=${QUERYOCC_SMOKE_ROOT:-"$PROJECT_ROOT/smoke/queryocc-nusc-smoke100"}
QUERYOCC_CONVERTED_ROOT=${QUERYOCC_CONVERTED_ROOT:-"$QUERYOCC_SMOKE_ROOT/converted_pseudo_labels"}
SMOKE_INPUT_DIR="$QUERYOCC_SMOKE_ROOT/smoke_inputs"

for required_path in \
    "$CONTAINER" \
    "$NUSC_DATASET_ROOT/v1.0-trainval" \
    "$NUSC_INFOS_ROOT/bevdetv2-nuscenes_infos_train.pkl" \
    "$NUSC_INFOS_ROOT/bevdetv2-nuscenes_infos_val.pkl" \
    "$NUSC_OCC3D_DATASET_ROOT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required smoke input does not exist: $required_path" >&2
        exit 2
    fi
done

# Existing DragonBall/GaussianFlowOcc data uses a flat per-camera semantic
# layout, label 17 for sky, and a differently ordered six-camera depth .npy.
# When both legacy roots are configured, convert every currently complete
# record into QueryOcc's self-describing NPZ layout before selecting the smoke
# subset. The conversion is incremental, so rerunning this job picks up newly
# generated labels without rewriting prior records.
if [[ -n "${LEGACY_PSEUDO_SEMANTICS_ROOT:-}" || -n "${LEGACY_PSEUDO_DEPTH_ROOT:-}" ]]; then
    : "${LEGACY_PSEUDO_SEMANTICS_ROOT:?Set both legacy pseudo-label roots}"
    : "${LEGACY_PSEUDO_DEPTH_ROOT:?Set both legacy pseudo-label roots}"
    for legacy_path in "$LEGACY_PSEUDO_SEMANTICS_ROOT" "$LEGACY_PSEUDO_DEPTH_ROOT"; do
        if [[ ! -d "$legacy_path" ]]; then
            echo "Legacy pseudo-label input does not exist: $legacy_path" >&2
            exit 2
        fi
    done
    CONVERTED_ROOT="$QUERYOCC_CONVERTED_ROOT"
    mkdir -p "$CONVERTED_ROOT"
    apptainer exec \
        --bind "$PWD":/queryocc:ro \
        --bind "$NUSC_INFOS_ROOT":/source_infos:ro \
        --bind "$LEGACY_PSEUDO_SEMANTICS_ROOT":/legacy_semantics:ro \
        --bind "$LEGACY_PSEUDO_DEPTH_ROOT":/legacy_depth:ro \
        --bind "$CONVERTED_ROOT":/converted \
        --env PYTHONPATH=/queryocc \
        --pwd /queryocc \
        "$CONTAINER" \
        python tools/convert_legacy_queryocc_pseudo_labels.py \
            --infos /source_infos/bevdetv2-nuscenes_infos_train.pkl \
            --semantics-root /legacy_semantics \
            --depth-root /legacy_depth \
            --output-root /converted \
            --min-complete 100 \
            --require-scores
    PSEUDO_SEMANTICS_ROOT="$CONVERTED_ROOT/semantics"
    PSEUDO_DEPTH_ROOT="$CONVERTED_ROOT/depth"
fi

for required_path in \
    "$PSEUDO_SEMANTICS_ROOT" \
    "$PSEUDO_DEPTH_ROOT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required smoke input does not exist: $required_path" >&2
        exit 2
    fi
done

mkdir -p "$SMOKE_INPUT_DIR"

apptainer exec \
    --bind "$PWD":/queryocc:ro \
    --bind "$NUSC_DATASET_ROOT":/datasets/nuscenes:ro \
    --bind "$NUSC_INFOS_ROOT":/source_infos:ro \
    --bind "$NUSC_OCC3D_DATASET_ROOT":/datasets/occ3d-nuscenes:ro \
    --bind "$PSEUDO_SEMANTICS_ROOT":/datasets/pseudo_labels/semantics:ro \
    --bind "$PSEUDO_DEPTH_ROOT":/datasets/pseudo_labels/depth:ro \
    --bind "$SMOKE_INPUT_DIR":/smoke_inputs \
    --env PYTHONPATH=/queryocc \
    --pwd /queryocc \
    "$CONTAINER" \
    python tools/prepare_queryocc_smoke.py \
        --train-info /source_infos/bevdetv2-nuscenes_infos_train.pkl \
        --val-info /source_infos/bevdetv2-nuscenes_infos_val.pkl \
        --data-root /datasets/nuscenes \
        --semantics-root /datasets/pseudo_labels/semantics \
        --depth-root /datasets/pseudo_labels/depth \
        --gt-root /datasets/occ3d-nuscenes \
        --output-dir /smoke_inputs \
        --infos-tag queryocc-smoke \
        --train-count 100 \
        --train-sample-budget 200 \
        --val-count 32 \
        --world-size 2 \
        --samples-per-gpu 1 \
        --accumulate-grad-batches 1 \
        --check-archives

echo "Frozen QueryOcc smoke inputs: $SMOKE_INPUT_DIR"
