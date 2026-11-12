#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=berzelius
#SBATCH --gpus-per-node=A100-SXM4-80GB:4
#SBATCH --constraint=fat
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --output=/proj/agp/users/%u/logs/%j.out
#SBATCH --account=berzelius-2026-281
#SBATCH --job-name=queryocc_full16
#

set -euo pipefail

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
NUSC_INFOS_ROOT=${NUSC_INFOS_ROOT:-"$PROJECT_ROOT/infos"}
QUERYOCC_CONVERTED_ROOT=${QUERYOCC_CONVERTED_ROOT:-"$PROJECT_ROOT/pseudo_labels/queryocc-refined"}
FULL_TRAINING_INFOS_TAG=${FULL_TRAINING_INFOS_TAG:-queryocc-refined}
RUN_NAME=${RUN_NAME:-queryocc-nusc-full16}
FULL_NAME="$RUN_NAME-$SLURM_JOB_ID"
RUN_ROOT="$OUTPUT_DIR/experiments/$FULL_NAME"

for required_path in \
    "$QUERYOCC_CONVERTED_ROOT/full_training_preflight.json" \
    "$QUERYOCC_CONVERTED_ROOT/full_training_blacklist.json" \
    "$NUSC_INFOS_ROOT/${FULL_TRAINING_INFOS_TAG}_infos_train.pkl" \
    "$NUSC_INFOS_ROOT/${FULL_TRAINING_INFOS_TAG}_infos_val.pkl"; do
    if [[ ! -s "$required_path" ]]; then
        echo "Required prepared input does not exist or is empty: $required_path" >&2
        exit 2
    fi
done

if [[ "${SLURM_GPUS_ON_NODE:-}" != "4" ]]; then
    echo "Expected exactly 4 GPUs, got '${SLURM_GPUS_ON_NODE:-unset}'" >&2
    exit 2
fi

export NUM_GPUS=4
export WORKERS_PER_GPU=${WORKERS_PER_GPU:-5}
export QUERYOCC_SKIP_ENV_LOCAL=1

scripts/train_multi_gpu.sh "$RUN_NAME" queryocc-nusc \
    --no-test \
    --trainer.max-epochs=16 \
    --trainer.max-steps=-1 \
    --trainer.limit-val-batches=0 \
    --trainer.num-sanity-val-steps=0 \
    --datamodule.version=full \
    --datamodule.dataset.infos-tag="$FULL_TRAINING_INFOS_TAG"

container_run_root="/outputs/experiments/$FULL_NAME"
container_manifest="$container_run_root/checkpoint_manifest.json"
ENTRYPOINT="python -u tools/build_queryocc_checkpoint_manifest.py --run-root $container_run_root --epochs 10 11 12 13 14 15 16 --output $container_manifest --training-job-id $SLURM_JOB_ID" \
    scripts/_run.sh "${RUN_NAME}-manifest" queryocc-nusc

if [[ ! -s "$RUN_ROOT/checkpoint_manifest.json" ]]; then
    echo "Training finished without a valid checkpoint manifest" >&2
    exit 1
fi
echo "QueryOcc 16-epoch training complete: $RUN_ROOT/checkpoint_manifest.json"
