#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=berzelius-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/proj/agp/users/%u/logs/%j.out
#SBATCH --account=berzelius-2026-281
#SBATCH --job-name=queryocc_prepare
#

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/env.local.sh" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/scripts"
else
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
if [[ ! -f "$SCRIPT_DIR/env.local.sh" ]]; then
    echo "Missing $SCRIPT_DIR/env.local.sh; copy env.local.example and fill in the site paths" >&2
    exit 2
fi
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.local.sh"
cd "$REPO_ROOT"

: "${PROJECT_ROOT:?Set PROJECT_ROOT in scripts/env.local.sh}"
: "${LEGACY_PSEUDO_SEMANTICS_ROOT:?Set the legacy refined-label root}"
: "${LEGACY_PSEUDO_DEPTH_ROOT:?Set the legacy Metric3D root}"
: "${QUERYOCC_CONVERTED_ROOT:?Set the canonical QueryOcc output root}"

CONTAINER=${CONTAINER:-"$PROJECT_ROOT/containers/queryocc.sif"}
NUSC_INFOS_ROOT=${NUSC_INFOS_ROOT:-"$PROJECT_ROOT/infos"}
NUSC_DATASET_ROOT=${NUSC_DATASET_ROOT:-"$PROJECT_ROOT/nuscenes"}
FULL_CONVERSION_EXPECTED_TRAIN=${FULL_CONVERSION_EXPECTED_TRAIN:-28130}
CONVERSION_WORKERS=${CONVERSION_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}
FULL_TRAINING_INFOS_TAG=${FULL_TRAINING_INFOS_TAG:-queryocc-refined}

if (( CONVERSION_WORKERS < 1 )); then
    echo "CONVERSION_WORKERS must be at least 1, got $CONVERSION_WORKERS" >&2
    exit 2
fi

for required_path in \
    "$CONTAINER" \
    "$NUSC_INFOS_ROOT/bevdetv2-nuscenes_infos_train.pkl" \
    "$LEGACY_PSEUDO_SEMANTICS_ROOT" \
    "$LEGACY_PSEUDO_DEPTH_ROOT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required conversion input does not exist: $required_path" >&2
        exit 2
    fi
done

mkdir -p "$QUERYOCC_CONVERTED_ROOT"

# The completeness gate runs before any payload is written. Conversion is
# incremental and each output pair is atomically renamed, so a timed-out job can
# be resubmitted without rewriting completed samples.
apptainer exec \
    --bind "$REPO_ROOT":/queryocc:ro \
    --bind "$NUSC_INFOS_ROOT":/source_infos:ro \
    --bind "$LEGACY_PSEUDO_SEMANTICS_ROOT":/legacy_semantics:ro \
    --bind "$LEGACY_PSEUDO_DEPTH_ROOT":/legacy_depth:ro \
    --bind "$QUERYOCC_CONVERTED_ROOT":/converted \
    --env PYTHONPATH=/queryocc \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env OPENBLAS_NUM_THREADS=1 \
    --env NUMEXPR_NUM_THREADS=1 \
    --pwd /queryocc \
    "$CONTAINER" \
    python tools/convert_legacy_queryocc_pseudo_labels.py \
        --infos /source_infos/bevdetv2-nuscenes_infos_train.pkl \
        --semantics-root /legacy_semantics \
        --depth-root /legacy_depth \
        --output-root /converted \
        --min-complete "$FULL_CONVERSION_EXPECTED_TRAIN" \
        --workers "$CONVERSION_WORKERS" \
        --require-scores

echo "Full QueryOcc pseudo-label conversion complete: $QUERYOCC_CONVERTED_ROOT"

# Validate the materialized archives, blacklist records with no usable positive
# pseudo points, and write a filtered train index plus an unchanged val index.
apptainer exec \
    --bind "$REPO_ROOT":/queryocc:ro \
    --bind "$NUSC_INFOS_ROOT":/source_infos \
    --bind "$QUERYOCC_CONVERTED_ROOT":/converted \
    --env PYTHONPATH=/queryocc \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env OPENBLAS_NUM_THREADS=1 \
    --env NUMEXPR_NUM_THREADS=1 \
    --pwd /queryocc \
    "$CONTAINER" \
    python tools/prepare_queryocc_full_training.py \
        --train-info /source_infos/bevdetv2-nuscenes_infos_train.pkl \
        --val-info /source_infos/bevdetv2-nuscenes_infos_val.pkl \
        --semantics-root /converted/semantics \
        --depth-root /converted/depth \
        --infos-output-dir /source_infos \
        --artifact-dir /converted \
        --infos-tag "$FULL_TRAINING_INFOS_TAG" \
        --workers "$CONVERSION_WORKERS"

# Precompute the derived QueryOcc cache from the filtered split definition so
# the four GPU ranks do not race to rebuild it at training startup.
apptainer exec \
    --bind "$REPO_ROOT":/queryocc:ro \
    --bind "$NUSC_DATASET_ROOT":/datasets/nuscenes:ro \
    --bind "$NUSC_INFOS_ROOT":/source_infos \
    --env PYTHONPATH=/queryocc \
    --pwd /queryocc \
    "$CONTAINER" \
    python -m tools.create_queryocc_infos \
        --version full \
        --data /datasets/nuscenes \
        --infos-dir /source_infos \
        --infos-tag "$FULL_TRAINING_INFOS_TAG" \
        --overwrite

echo "Full QueryOcc preparation ready: $QUERYOCC_CONVERTED_ROOT/full_training_preflight.json"
