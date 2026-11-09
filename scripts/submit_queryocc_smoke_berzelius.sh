#!/bin/bash
# Submit smoke-input preparation followed by the dependent GPU run.

set -euo pipefail

command -v sbatch >/dev/null || { echo "sbatch is unavailable; run this on a Slurm login node" >&2; exit 1; }

prepare_job=$(sbatch --parsable scripts/prepare_queryocc_smoke_berzelius.sh)
train_job=$(sbatch --parsable --dependency="afterok:$prepare_job" scripts/train_queryocc_smoke_berzelius.sh)

echo "prepare_job=$prepare_job"
echo "train_job=$train_job (afterok:$prepare_job)"
