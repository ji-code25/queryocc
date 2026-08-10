#!/bin/bash
# Semantic pseudo-label generation as a SLURM job array (Apptainer). One GPU
# per array task; each task processes one contiguous shard of the infos pkl
# (no split files needed — sharding is computed from the array index).
#
# Submit e.g. 16 shards:
#   sbatch --array=0-15 scripts/slurm_generate_semantics.sh
#
# Tasks are resumable: rerunning skips samples whose output already exists.
#
#SBATCH --nodes 1
# The GPU request is site-specific — pass it at submit time, e.g.
#   sbatch --gpus-per-node=A100:1 ...   (some clusters reject plain --gpus)
#   sbatch --gpus 1 ...                 (most other clusters)
#SBATCH --time 2-00:00:00
#SBATCH --output logs/%A_%a.out
#SBATCH --job-name=pseudo_semantics
#
# Locate the shared helper. Under sbatch the script runs from a SLURM spool
# copy, so $BASH_SOURCE does not point at the repo — fall back to the submit
# directory / cwd (the repo root, which these scripts already assume).
_gpl_common=""
for _d in "$(dirname "${BASH_SOURCE[0]}")" \
          "${SLURM_SUBMIT_DIR:-.}/generate_pseudo_labels/scripts" \
          "${PWD}/generate_pseudo_labels/scripts" "${PWD}/scripts"; do
    [ -f "${_d}/_common.sh" ] && { _gpl_common="${_d}/_common.sh"; break; }
done
[ -n "$_gpl_common" ] || { echo "error: cannot find scripts/_common.sh; run from the repo root" >&2; exit 1; }
# shellcheck disable=SC1090
source "$_gpl_common"

# --- configuration (override via environment / scripts/env.local.sh) -------
INFOS_PKL="${INFOS_PKL:-data/bevdetv2-nuscenes_infos_train.pkl}"
PSEUDO_LABEL_ROOT="${PSEUDO_LABEL_ROOT:-data/pseudo_labels}"
SAVE_PATH="${SEMANTICS_SAVE_PATH:-${PSEUDO_LABEL_ROOT}/semantics}"
DATA_ROOT="${DATA_ROOT:-$PWD}"
SAM_CKPT="${SAM_CKPT:-ckpts/sam_vit_h_4b8939.pth}"
# Throughput defaults (see README "Performance"): TF32 matmuls, bf16
# autocast (NOT bit-exact — argmax labels virtually identical, boundary
# pixels may flip; drop --amp bf16 for exact fp32), and 2 prompts per
# transformer pass. Backbone/text caching is on by default and self-checked
# against the stock forward on the first sample.
EXTRA_ARGS="${EXTRA_ARGS:---tf32 --amp bf16 --prompt-chunk 2}"
# ----------------------------------------------------------------------------

# Explicit NUM_SHARDS/SHARD win over the array variables, so a single
# failed task can be resubmitted without resharding the run:
#   NUM_SHARDS=16 sbatch --array=5 scripts/slurm_generate_semantics.sh
# (SLURM sets SLURM_ARRAY_TASK_COUNT=1 for a one-element array.)
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"
echo "shard ${SHARD} / ${NUM_SHARDS}  infos=${INFOS_PKL}  out=${SAVE_PATH}"

run_pipeline gpu \
    python -m generate_pseudo_labels.generate_semantics \
        --infos-pkl "$INFOS_PKL" \
        --save-path "$SAVE_PATH" \
        --data-root "$DATA_ROOT" \
        --sam-checkpoint "$SAM_CKPT" \
        --shard "$SHARD" \
        --num-shards "$NUM_SHARDS" \
        $EXTRA_ARGS
