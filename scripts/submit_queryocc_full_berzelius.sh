#!/bin/bash
# Submit full preparation, 16-epoch training, seven evaluations, and selection.

set -euo pipefail

command -v sbatch >/dev/null || { echo "sbatch is unavailable; run this on Berzelius" >&2; exit 1; }

RUN_NAME=${RUN_NAME:-queryocc-nusc-full16}
prepare_job=$(sbatch --parsable scripts/convert_queryocc_pseudo_labels_berzelius.sh)
smoke_job=$(sbatch --parsable --dependency="afterok:$prepare_job" scripts/smoke_occ3d_only_berzelius.sh)
train_job=$(sbatch --parsable --dependency="afterok:$smoke_job" --export="ALL,RUN_NAME=$RUN_NAME" scripts/train_queryocc_full_berzelius.sh)
evaluation_job=$(sbatch --parsable --array=10-16 --dependency="afterok:$train_job" --export="ALL,TRAIN_JOB_ID=$train_job,RUN_NAME=$RUN_NAME" scripts/evaluate_queryocc_full_berzelius.sh)
selection_job=$(sbatch --parsable --dependency="afterok:$evaluation_job" --export="ALL,TRAIN_JOB_ID=$train_job,RUN_NAME=$RUN_NAME" scripts/select_queryocc_best_berzelius.sh)

echo "prepare_job=$prepare_job"
echo "smoke_job=$smoke_job (afterok:$prepare_job)"
echo "train_job=$train_job (afterok:$smoke_job)"
echo "evaluation_job=$evaluation_job[10-16] (afterok:$train_job)"
echo "selection_job=$selection_job (afterok:$evaluation_job)"
