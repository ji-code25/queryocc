#!/bin/bash
#
# One-GPU end-to-end regression for QueryOcc's Occ3D-only evaluation path.
# The production pseudo trees contain train samples only, so the real val split
# deliberately has no pseudo-label or Metric3D archives.
#
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/proj/agp/users/%u/logs/%j.out
#SBATCH --account=berzelius-2026-289
#SBATCH --job-name=queryocc_occ3d_smoke

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/env.local.sh" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/scripts"
else
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

# shellcheck source=/dev/null
set -a
source "$SCRIPT_DIR/env.local.sh"
set +a

REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
FULL_TRAINING_INFOS_TAG=${FULL_TRAINING_INFOS_TAG:-queryocc-refined}

for required_path in \
    "$NUSC_INFOS_ROOT/${FULL_TRAINING_INFOS_TAG}_infos_train.pkl" \
    "$NUSC_INFOS_ROOT/${FULL_TRAINING_INFOS_TAG}_infos_val.pkl" \
    "$PSEUDO_SEMANTICS_ROOT" \
    "$PSEUDO_DEPTH_ROOT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing Occ3D-only smoke input: $required_path" >&2
        exit 2
    fi
done

export QUERYOCC_SKIP_ENV_LOCAL=1

NAME="queryocc-occ3d-only-smoke"

# Use synchronous loading for this tiny regression so QueryOcc's five-batch
# warmup iterator cannot leave a persistent worker pool partway through the
# epoch. Two real optimizer steps from the complete production train
# pseudo/depth trees exercise supervision. Epoch-1 validation and both post-fit
# tests exercise images + Occ3D only. The production val tokens have no files in
# either pseudo-label tree, so any accidental read fails the job immediately.
scripts/_run.sh "$NAME" queryocc-nusc \
    --test \
    --test-best-and-last-after-fit \
    --no-use-logging \
    --no-allow-visualization \
    --seed=0 \
    --validation-epochs 1 \
    --trainer.devices=1 \
    --trainer.accumulate-grad-batches=1 \
    --trainer.max-steps=-1 \
    --trainer.max-epochs=1 \
    --trainer.limit-val-batches=2 \
    --trainer.limit-test-batches=2 \
    --trainer.num-sanity-val-steps=0 \
    --datamodule.version=full \
    --datamodule.batch-size=1 \
    --datamodule.val-batch-size=1 \
    --datamodule.num-workers=0 \
    --datamodule.val-num-workers=0 \
    --datamodule.train-dataset-usage-num=1 \
    --datamodule.min-dataset-length=2 \
    --datamodule.dataset.infos-tag="$FULL_TRAINING_INFOS_TAG"

RUN_ROOT="$OUTPUT_DIR/experiments/${NAME}-${SLURM_JOB_ID}"
mapfile -t recovery_checkpoints < <(find "$RUN_ROOT" -type f -name last.ckpt -print)
mapfile -t best_checkpoints < <(find "$RUN_ROOT" -type f -name '*.ckpt' ! -name 'last.ckpt' -print)

if [[ "${#recovery_checkpoints[@]}" -ne 1 || "${#best_checkpoints[@]}" -ne 1 ]]; then
    echo "Expected one recovery and one Occ3D-best checkpoint under $RUN_ROOT" >&2
    printf 'recovery: %s\n' "${recovery_checkpoints[@]:-none}" >&2
    printf 'best: %s\n' "${best_checkpoints[@]:-none}" >&2
    exit 1
fi

echo "Occ3D-only smoke passed"
echo "Recovery checkpoint: ${recovery_checkpoints[0]}"
echo "Best checkpoint: ${best_checkpoints[0]}"
