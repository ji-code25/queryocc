#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=berzelius-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/proj/agp/users/%u/logs/%j.out
#SBATCH --account=berzelius-2026-289
#SBATCH --job-name=queryocc_select
#

set -euo pipefail

: "${TRAIN_JOB_ID:?Export TRAIN_JOB_ID when submitting the result selector}"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/env.local.sh" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/scripts"
else
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
if [[ ! -f "$SCRIPT_DIR/env.local.sh" ]]; then
    echo "Missing $SCRIPT_DIR/env.local.sh" >&2
    exit 2
fi
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.local.sh"
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

: "${PROJECT_ROOT:?Set PROJECT_ROOT in scripts/env.local.sh}"
CONTAINER=${CONTAINER:-"$PROJECT_ROOT/containers/queryocc.sif"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT"}
RUN_NAME=${RUN_NAME:-queryocc-nusc-full16}
container_run_root="/outputs/experiments/$RUN_NAME-$TRAIN_JOB_ID"

if [[ "${QUERYOCC_LAUNCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "QueryOcc best-checkpoint selector launch preflight passed: $REPO_ROOT"
    exit 0
fi

apptainer exec \
    --bind "$REPO_ROOT":/queryocc:ro \
    --bind "$OUTPUT_DIR":/outputs \
    --env PYTHONPATH=/queryocc \
    --pwd /queryocc \
    "$CONTAINER" \
    python tools/select_best_queryocc_checkpoint.py \
        --results-dir "$container_run_root/full_validation" \
        --epochs 10 11 12 13 14 15 16 \
        --output "$container_run_root/evaluation_summary.json" \
        --best-path-output "$container_run_root/best_checkpoint.txt" \
        --best-symlink "$container_run_root/best.ckpt"

echo "Best QueryOcc checkpoint: $OUTPUT_DIR/experiments/$RUN_NAME-$TRAIN_JOB_ID/best_checkpoint.txt"
