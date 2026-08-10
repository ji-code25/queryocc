#!/bin/bash
# Create a local uv environment (.venv) for development and testing outside docker.
#
# Installs the full python stack (torch cu126 wheels). There is no install-time
# compile step: the only native extension left is the dvr raycaster used by the
# RayIoU metrics, which torch JIT-compiles lazily on first use (needs a CUDA
# toolkit + GPU, i.e. inside the container or on a cluster node).
#
# Usage:
#   ./setup_uv_env.sh
#   source .venv/bin/activate

set -e
cd "$(dirname "$0")"

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== Creating .venv (python 3.10, same as the docker image) ==="
uv venv --python 3.10 --allow-existing .venv

# unsafe-best-match: the torch +cu126 pins only exist on the pytorch extra index,
# uv's default strategy would otherwise stop at PyPI.
echo "=== Installing python dependencies ==="
uv pip install --index-strategy unsafe-best-match "setuptools<80" pybind11 ninja
uv pip install --index-strategy unsafe-best-match -r requirements/requirements.txt
uv pip install -e .

echo ""
echo "Done. Activate with: source .venv/bin/activate"
