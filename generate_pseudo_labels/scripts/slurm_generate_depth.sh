#!/bin/bash
# Metric3D depth pseudo-label generation as a SLURM job array (Apptainer).
# Metric3D is ~2 orders of magnitude cheaper per image than the
# GroundingDINO+SAM pipeline; a handful of shards is plenty:
#
#   sbatch --array=0-3 scripts/slurm_generate_depth.sh
#
# Tasks are resumable: rerunning skips samples whose output already exists.
#
#SBATCH --nodes 1
# The GPU request is site-specific — pass it at submit time, e.g.
#   sbatch --gpus-per-node=A100:1 ...   (some clusters reject plain --gpus)
#   sbatch --gpus 1 ...                 (most other clusters)
#SBATCH --time 1-00:00:00
#SBATCH --output logs/%A_%a.out
#SBATCH --job-name=pseudo_depth
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
SAVE_PATH="${DEPTH_SAVE_PATH:-${PSEUDO_LABEL_ROOT}/depth}"
DATA_ROOT="${DATA_ROOT:-$PWD}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# ----------------------------------------------------------------------------

# Explicit NUM_SHARDS/SHARD win over the array variables, so a single
# failed task can be resubmitted without resharding the run:
#   NUM_SHARDS=16 sbatch --array=5 scripts/slurm_generate_semantics.sh
# (SLURM sets SLURM_ARRAY_TASK_COUNT=1 for a one-element array.)
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"
echo "shard ${SHARD} / ${NUM_SHARDS}  infos=${INFOS_PKL}  out=${SAVE_PATH}"

run_pipeline gpu \
    python -m generate_pseudo_labels.generate_depth \
        --infos-pkl "$INFOS_PKL" \
        --save-path "$SAVE_PATH" \
        --data-root "$DATA_ROOT" \
        --shard "$SHARD" \
        --num-shards "$NUM_SHARDS" \
        $EXTRA_ARGS
