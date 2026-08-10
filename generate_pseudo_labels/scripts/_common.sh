#!/bin/bash
# Shared plumbing for the SLURM scripts in this directory.
#
# Every path in the pipeline comes from an environment variable with a
# repo-relative default, so nothing site-specific is baked into the scripts.
# If a file `env.local.sh` exists next to this one it is sourced first — put
# your cluster paths, container location and account there (it is gitignored;
# see README "Site configuration").
#
# Provides:
#   run_pipeline <gpu|cpu> <command...>   run inside $CONTAINER (Apptainer) if
#                                         set, otherwise in the current env
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${SCRIPT_DIR}/env.local.sh" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.local.sh"
fi

# --- generic defaults (override in the environment or in env.local.sh) ------
# Apptainer image with the generation dependencies. Empty = no container:
# run in whatever Python environment is already active.
CONTAINER="${CONTAINER:-}"
# Space-separated Apptainer --bind specs for cluster storage, e.g.
# "/data:/data /scratch:/scratch". The current working directory is always
# bound.
BIND_PATHS="${BIND_PATHS:-}"
# Persistent HF / torch.hub caches (compute nodes often have no internet;
# populate once with prefetch_weights.py).
MODEL_CACHE="${MODEL_CACHE:-}"

run_pipeline() {
    local mode="$1"; shift
    if [ -n "$CONTAINER" ] && [ ! -f "$CONTAINER" ]; then
        echo "error: CONTAINER='$CONTAINER' does not exist. Build it " \
             "(containers/build.sh) or set CONTAINER= to run in the current " \
             "Python environment." >&2
        exit 1
    fi
    local env_args=()
    if [ -n "$MODEL_CACHE" ]; then
        env_args+=(TORCH_HOME="${MODEL_CACHE}/torch"
                   HF_HOME="${MODEL_CACHE}/huggingface")
    fi

    if [ -z "$CONTAINER" ]; then
        echo "+ (no container) $*"
        env "${env_args[@]}" "$@"
        return
    fi

    local apptainer_args=(exec)
    [ "$mode" = "gpu" ] && apptainer_args+=(--nv)
    local bind
    for bind in $BIND_PATHS; do
        apptainer_args+=(--bind "$bind")
    done
    apptainer_args+=(--bind "${PWD}:${PWD}" --pwd "${PWD}")
    local kv
    for kv in "${env_args[@]}"; do
        apptainer_args+=(--env "$kv")
    done
    apptainer_args+=("$CONTAINER")
    echo "+ apptainer ${apptainer_args[*]} $*"
    apptainer "${apptainer_args[@]}" "$@"
}
