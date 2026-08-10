#!/bin/bash
# Check a job array before submitting it (one short GPU job, ~10 s/shard).
# Reproduces every per-shard startup step of slurm_generate_semantics.sh, so
# a shard cannot die on its first sample hours after submission.
#
#   NUM_SHARDS=16 sbatch scripts/slurm_preflight.sh
#   NUM_SHARDS=16 bash   scripts/slurm_preflight.sh   # inside an salloc
#
#SBATCH --nodes 1
# The GPU request is site-specific — pass it at submit time, e.g.
#   sbatch --gpus-per-node=A100:1 ...   (some clusters reject plain --gpus)
#   sbatch --gpus 1 ...                 (most other clusters)
#SBATCH --time 0-00:30:00
#SBATCH --output logs/preflight_%j.out
#SBATCH --job-name=preflight
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
# The array size you are about to submit.
NUM_SHARDS="${NUM_SHARDS:-1}"
# Keep these identical to the run you are checking (see the generators'
# EXTRA_ARGS); --tf32 changes the numbers the self-check sees.
EXTRA_ARGS="${EXTRA_ARGS:---tf32}"
# ----------------------------------------------------------------------------

run_pipeline gpu \
    python -m generate_pseudo_labels.preflight \
        --infos-pkl "$INFOS_PKL" \
        --data-root "$DATA_ROOT" \
        --save-path "$SAVE_PATH" \
        --sam-checkpoint "$SAM_CKPT" \
        --num-shards "$NUM_SHARDS" \
        $EXTRA_ARGS
