#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=berzelius
#SBATCH --gpus-per-node=A100-SXM4-40GB:1
#SBATCH --constraint=thin
#SBATCH --cpus-per-task=5
#SBATCH --time=12:00:00
#SBATCH --output=/proj/agp/users/%u/logs/%A_%a.out
#SBATCH --account=berzelius-2026-281
#SBATCH --job-name=queryocc_fullval
#

set -euo pipefail

: "${TRAIN_JOB_ID:?Export TRAIN_JOB_ID when submitting the evaluation array}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as an array over epochs 10-16}"
EPOCH=$SLURM_ARRAY_TASK_ID
if (( EPOCH < 10 || EPOCH > 16 )); then
    echo "Evaluation epoch must be in [10, 16], got $EPOCH" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -f "$SCRIPT_DIR/env.local.sh" ]]; then
    echo "Missing $SCRIPT_DIR/env.local.sh" >&2
    exit 2
fi
set -a
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.local.sh"
set +a
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

: "${PROJECT_ROOT:?Set PROJECT_ROOT in scripts/env.local.sh}"
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT"}
FULL_TRAINING_INFOS_TAG=${FULL_TRAINING_INFOS_TAG:-queryocc-refined}
RUN_NAME=${RUN_NAME:-queryocc-nusc-full16}
RUN_ROOT="$OUTPUT_DIR/experiments/$RUN_NAME-$TRAIN_JOB_ID"
RESULTS_DIR="$RUN_ROOT/full_validation"

if [[ ! -s "$RUN_ROOT/checkpoint_manifest.json" ]]; then
    echo "Missing checkpoint manifest: $RUN_ROOT/checkpoint_manifest.json" >&2
    exit 2
fi
mapfile -t checkpoint_paths < <(find "$RUN_ROOT" -type f -path "*/checkpoints/epoch-$(printf '%02d' "$EPOCH").ckpt" -print)
if [[ "${#checkpoint_paths[@]}" -ne 1 ]]; then
    echo "Expected exactly one checkpoint for epoch $EPOCH, found ${#checkpoint_paths[@]}" >&2
    printf '%s\n' "${checkpoint_paths[@]}" >&2
    exit 2
fi

mkdir -p "$RESULTS_DIR"
checkpoint_path=${checkpoint_paths[0]}
result_path="$RESULTS_DIR/epoch-$(printf '%02d' "$EPOCH").json"
container_checkpoint="/outputs/${checkpoint_path#"$OUTPUT_DIR"/}"
container_result="/outputs/${result_path#"$OUTPUT_DIR"/}"

export NUM_GPUS=1
export WORKERS_PER_GPU=${WORKERS_PER_GPU:-5}
export QUERYOCC_SKIP_ENV_LOCAL=1

scripts/_run.sh "${RUN_NAME}-epoch${EPOCH}-fullval" queryocc-nusc \
    --no-train \
    --test \
    --no-use-logging \
    --no-allow-visualization \
    --load-ckpt-from="$container_checkpoint" \
    --evaluation-results-path="$container_result" \
    --evaluation-checkpoint-epoch="$EPOCH" \
    --trainer.devices=1 \
    --trainer.limit-test-batches=1.0 \
    --trainer.num-sanity-val-steps=0 \
    --datamodule.version=full \
    --datamodule.batch-size=1 \
    --datamodule.val-batch-size=1 \
    --datamodule.num-workers="$WORKERS_PER_GPU" \
    --datamodule.val-num-workers="$WORKERS_PER_GPU" \
    --datamodule.dataset.infos-tag="$FULL_TRAINING_INFOS_TAG"

if [[ ! -s "$result_path" ]]; then
    echo "Evaluation did not write $result_path" >&2
    exit 1
fi
echo "Full validation complete for epoch $EPOCH: $result_path"
