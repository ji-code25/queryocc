#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/proj/agp/users/%u/logs/%j.out
#SBATCH --account=berzelius-2026-289
#SBATCH --job-name=queryocc_smoke100
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
set -a
source "$SCRIPT_DIR/env.local.sh"
set +a
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

: "${PROJECT_ROOT:?Set PROJECT_ROOT in scripts/env.local.sh}"
CONTAINER=${CONTAINER:-"$PROJECT_ROOT/containers/queryocc.sif"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT"}
QUERYOCC_SMOKE_ROOT=${QUERYOCC_SMOKE_ROOT:-"$PROJECT_ROOT/smoke/queryocc-nusc-smoke100"}
QUERYOCC_CONVERTED_ROOT=${QUERYOCC_CONVERTED_ROOT:-"$QUERYOCC_SMOKE_ROOT/converted_pseudo_labels"}
SMOKE_INPUT_DIR="$QUERYOCC_SMOKE_ROOT/smoke_inputs"
NAME="queryocc-nusc-smoke100"
FULL_NAME="$NAME-$SLURM_JOB_ID"
RUN_ROOT="$OUTPUT_DIR/smoke_runs/$FULL_NAME"
SUMMARY_JSON="$RUN_ROOT/smoke_summary.json"

if [[ -n "${LEGACY_PSEUDO_SEMANTICS_ROOT:-}" || -n "${LEGACY_PSEUDO_DEPTH_ROOT:-}" ]]; then
    export PSEUDO_SEMANTICS_ROOT="$QUERYOCC_CONVERTED_ROOT/semantics"
    export PSEUDO_DEPTH_ROOT="$QUERYOCC_CONVERTED_ROOT/depth"
fi

if [[ "${SLURM_GPUS_ON_NODE:-}" != "2" ]]; then
    echo "Expected exactly 2 GPUs, got '${SLURM_GPUS_ON_NODE:-unset}'" >&2
    exit 2
fi
for required_path in \
    "$CONTAINER" \
    "$SMOKE_INPUT_DIR/queryocc-smoke_infos_train.pkl" \
    "$SMOKE_INPUT_DIR/queryocc-smoke_infos_val.pkl" \
    "$SMOKE_INPUT_DIR/manifest.json"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required smoke input does not exist: $required_path" >&2
        exit 2
    fi
done

mkdir -p "$RUN_ROOT" "$OUTPUT_DIR"

# The frozen info directory replaces the normal full-info bind for this run. The
# manifest stores the container-visible pseudo-label paths, so the same verifier
# also proves the launcher's Grounded-SAM/Metric3D binds are correct.
export NUSC_INFOS_ROOT="$SMOKE_INPUT_DIR"
export NUM_GPUS=2
export WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
export QUERYOCC_SKIP_ENV_LOCAL=1

echo "Experiment: $FULL_NAME"
echo "Frozen inputs: $SMOKE_INPUT_DIR"
echo "Run summary: $SUMMARY_JSON"

ENTRYPOINT="python -u tools/verify_queryocc_smoke.py --input-dir /datasets/nuscenes/infos --expected-train 100 --expected-val 32 --expected-steps 100" \
    scripts/_run.sh "${NAME}-preflight" queryocc-nusc

# max_steps is a sample budget before QueryOcc's cluster scaling:
# 100 unique records are repeated to a 200-sample budget:
# 200 / (2 GPUs * batch 1/GPU * accumulation 1) = 100 optimizer steps.
# The production network, compile path, 200k-query budget and pseudo-label pipeline
# remain unchanged. In-training validation covers the 28 valid temporal anchors from
# 32 frozen records; the default post-fit test is replaced by explicit best/final tests.
scripts/_run.sh "$NAME" queryocc-nusc \
    --no-test \
    --no-use-logging \
    --no-allow-visualization \
    --seed=0 \
    --trainer.devices=2 \
    --trainer.accumulate-grad-batches=1 \
    --trainer.max-steps=200 \
    --trainer.val-check-interval=1.0 \
    --trainer.check-val-every-n-epoch=1 \
    --trainer.limit-val-batches=1.0 \
    --trainer.num-sanity-val-steps=0 \
    --datamodule.version=full \
    --datamodule.batch-size=1 \
    --datamodule.val-batch-size=1 \
    --datamodule.num-workers="$WORKERS_PER_GPU" \
    --datamodule.val-num-workers=2 \
    --datamodule.train-dataset-usage-num=100 \
    --datamodule.min-dataset-length=200 \
    --datamodule.dataset.infos-tag=queryocc-smoke \
    --datamodule.dataset.no-load-queryocc-infos-if-exists \
    --datamodule.dataset.no-save-queryocc-infos

mapfile -t checkpoints < <(find "$OUTPUT_DIR/experiments/$FULL_NAME" -type f -name last.ckpt -print)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
    echo "Expected exactly one last.ckpt under $OUTPUT_DIR/experiments/$FULL_NAME; found ${#checkpoints[@]}" >&2
    printf '%s\n' "${checkpoints[@]}" >&2
    exit 1
fi

checkpoint_path=${checkpoints[0]}
container_checkpoint="/outputs/${checkpoint_path#"$OUTPUT_DIR"/}"

mapfile -t best_checkpoints < <(
    find "$OUTPUT_DIR/experiments/$FULL_NAME" -type f -name '*.ckpt' ! -name 'last*.ckpt' -print
)
if [[ "${#best_checkpoints[@]}" -ne 1 ]]; then
    echo "Expected exactly one best checkpoint under $OUTPUT_DIR/experiments/$FULL_NAME; found ${#best_checkpoints[@]}" >&2
    printf '%s\n' "${best_checkpoints[@]}" >&2
    exit 1
fi

best_checkpoint_path=${best_checkpoints[0]}
container_best_checkpoint="/outputs/${best_checkpoint_path#"$OUTPUT_DIR"/}"
best_validation_log="$OUTPUT_DIR/experiments/$FULL_NAME/best_full_smoke_validation.log"
final_validation_log="$OUTPUT_DIR/experiments/$FULL_NAME/final_full_smoke_validation.log"

# Exercise the same best-vs-final validation protocol used after production training.
# Here "full" means every valid temporal anchor from the 32 frozen smoke records;
# the production launcher will use the complete nuScenes validation split.
scripts/_run.sh "${NAME}-best-full-val" queryocc-nusc \
    --no-train \
    --test \
    --no-use-logging \
    --no-allow-visualization \
    --no-validate-config \
    --load-ckpt-from="$container_best_checkpoint" \
    --trainer.devices=2 \
    --trainer.limit-test-batches=1.0 \
    --trainer.num-sanity-val-steps=0 \
    --datamodule.version=full \
    --datamodule.batch-size=1 \
    --datamodule.val-batch-size=1 \
    --datamodule.num-workers="$WORKERS_PER_GPU" \
    --datamodule.val-num-workers=2 \
    --datamodule.dataset.infos-tag=queryocc-smoke \
    --datamodule.dataset.no-load-queryocc-infos-if-exists \
    --datamodule.dataset.no-save-queryocc-infos \
    2>&1 | tee "$best_validation_log"

scripts/_run.sh "${NAME}-final-full-val" queryocc-nusc \
    --no-train \
    --test \
    --no-use-logging \
    --no-allow-visualization \
    --no-validate-config \
    --load-ckpt-from="$container_checkpoint" \
    --trainer.devices=2 \
    --trainer.limit-test-batches=1.0 \
    --trainer.num-sanity-val-steps=0 \
    --datamodule.version=full \
    --datamodule.batch-size=1 \
    --datamodule.val-batch-size=1 \
    --datamodule.num-workers="$WORKERS_PER_GPU" \
    --datamodule.val-num-workers=2 \
    --datamodule.dataset.infos-tag=queryocc-smoke \
    --datamodule.dataset.no-load-queryocc-infos-if-exists \
    --datamodule.dataset.no-save-queryocc-infos \
    2>&1 | tee "$final_validation_log"

container_summary="/outputs/${SUMMARY_JSON#"$OUTPUT_DIR"/}"
container_best_validation_log="/outputs/${best_validation_log#"$OUTPUT_DIR"/}"
container_final_validation_log="/outputs/${final_validation_log#"$OUTPUT_DIR"/}"
# The 32 frozen info records yield 28 valid anchors after QueryOcc removes scene
# boundaries that cannot provide the required temporal context.
ENTRYPOINT="python -u tools/collect_queryocc_smoke.py --manifest /datasets/nuscenes/infos/manifest.json --checkpoint $container_checkpoint --best-checkpoint $container_best_checkpoint --best-validation-log $container_best_validation_log --final-validation-log $container_final_validation_log --effective-val-samples 28 --output $container_summary --expected-steps 100" \
    scripts/_run.sh "${NAME}-collect" queryocc-nusc

echo "QueryOcc smoke test complete: $SUMMARY_JSON"
