#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus-per-node=4
#SBATCH --time 0-2:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --job-name=eval-queryocc
#
# Evaluate checkpoints from a run list (same list format as array_train.sh):
#   sbatch --array=1-4 scripts/arrays/array_evaluate.sh <list-dir> <list> <ckpt-path>

INPUT_DIR=$1
INPUT_FILE=$2
CKPT_NAME=${@:3}
TXT_FILE=scripts/arrays/lists/${INPUT_DIR}/${INPUT_FILE}.txt

# Dataset version: override with DATASET_VERSION=mini for smoke tests.
DATASET_VERSION=${DATASET_VERSION:-full}

NUM_GPUS=${NUM_GPUS:-4}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}

echo "[`date`] Input file: $TXT_FILE"
eval `head -n $SLURM_ARRAY_TASK_ID $TXT_FILE | tail -1`
MY_SLURM_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "[`date`] SLURM ID: $MY_SLURM_ID"
SLURM_JOB_ID=$MY_SLURM_ID

# Run name: {list}_{arm} (+ "-<slurmid>" appended by the runner script). The list prefix
# is skipped when the arm name already starts with it.
if [[ -n "$name" && "$name" == "$INPUT_FILE"* ]]; then
    jobname=$name
else
    jobname=${INPUT_FILE}_${name}
fi
echo "[`date`] Job name: $jobname"

cmd="$jobname $model $extra"
echo "[`date`] Command: $cmd"
echo "[`date`] Sleeping for up to 20 seconds to avoid resource contention"
sleep $((RANDOM % 20))
echo "[`date`] Running job"
scripts/_run.sh $cmd --trainer.devices=$NUM_GPUS \
                     --datamodule.num-workers=$WORKERS_PER_GPU \
                     --datamodule.version=$DATASET_VERSION \
                     --trainer.num_sanity_val_steps=0 \
                     --load_ckpt_from=/outputs/experiments/$CKPT_NAME \
                     --no-train
#
#EOF
