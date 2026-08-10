#!/bin/bash
# Convert the queryocc docker image to an Apptainer image (queryocc_YYYYMMDD.sif)
# for use on the clusters. Builds the docker image first if it doesn't exist.

set -e

if ! command -v apptainer &> /dev/null; then
    echo "ERROR: apptainer is not installed (https://apptainer.org/docs/admin/main/installation.html)"
    exit 1
fi

if ! docker image inspect queryocc:latest &> /dev/null; then
    echo "Docker image queryocc:latest not found, building it first..."
    ./containers/build_docker.sh
fi

output_file="queryocc_$(date +%Y%m%d).sif"
if [ -f "$output_file" ]; then
    echo "Removing existing $output_file"
    rm -f "$output_file"
fi

echo "Converting docker image to $output_file (this may take several minutes)..."
apptainer build "$output_file" docker-daemon://queryocc:latest

# --cleanenv + PYTHONNOUSERSITE: apptainer mounts $HOME, so a host
# ~/.local/lib/python3.10/site-packages shadows the container's own packages and the
# check would happily report the host's torch. The cluster launchers set
# PYTHONNOUSERSITE=1 for the same reason.
echo "Verifying $output_file..."
apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 "$output_file" \
    python3.10 -c "import torch, queryocc.train; print('torch', torch.__version__, '|', torch.__file__)"

echo "Done. Usage:"
echo "  apptainer shell --nv $output_file"
echo "  apptainer exec --nv $output_file python3.10 queryocc/train.py ..."
