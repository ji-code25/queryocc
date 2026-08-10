#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus-per-node=4
#SBATCH --time 14:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --job-name=queryocc
#
# Launch one training arm per array index from a run list, e.g. the full release matrix:
#   sbatch --array=1-4 scripts/arrays/array_train.sh release
# Line N of scripts/arrays/lists/<list>.txt defines arm N ($name, $model, $extra).

INPUT_FILE=$1
EXTRA_CONFIG=${@:2}
TXT_FILE=scripts/arrays/lists/${INPUT_FILE}.txt

# Dataset version: override with DATASET_VERSION=mini for smoke tests.
DATASET_VERSION=${DATASET_VERSION:-full}

NUM_GPUS=${NUM_GPUS:-4}
# datamodule.num-workers is PER DDP RANK — never multiply by the GPU count. 8/GPU leaves
# headroom for the main processes and the validation loader pool on a typical
# 16-cores-per-GPU allocation; oversubscribing the cores collapses throughput even while
# GPU utilization still reads 100%. Tune via WORKERS_PER_GPU / datamodule.val_num_workers.
WORKERS_PER_GPU=${WORKERS_PER_GPU:-8}

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
scripts/_run.sh $cmd --trainer.devices=$NUM_GPUS --datamodule.num-workers=$WORKERS_PER_GPU --datamodule.version=$DATASET_VERSION --no-disable_gradient_checkpointing_on_large_node --trainer.num_sanity_val_steps=0 $EXTRA_CONFIG

#
#EOF
