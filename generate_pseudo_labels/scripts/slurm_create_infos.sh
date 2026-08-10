#!/bin/bash
# Build the bevdetv2-style infos pkls (CPU only, one task). Needs the raw
# nuScenes tables + images on disk.
#
#   sbatch scripts/slurm_create_infos.sh
#   VERSION=v1.0-mini bash scripts/slurm_create_infos.sh   # or just run it
#
# Cheap enough (minutes for mini, <1 h for trainval) to run on a login node
# without SLURM — the #SBATCH block is only used when submitted with sbatch.
#
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --time 0-04:00:00
#SBATCH --output logs/%j.out
#SBATCH --job-name=create_infos
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
NUSC_ROOT="${NUSC_ROOT:-data/nuscenes}"
OUT_DIR="${OUT_DIR:-data}"
VERSION="${VERSION:-v1.0-trainval}"
# Where the occ3d GT lives, as stored in the pkl (see --occ-path-prefix).
OCC_PATH_PREFIX="${OCC_PATH_PREFIX:-./data/gts}"
# If set, image/lidar paths are stored relative to this directory, which
# keeps the pkls portable across machines (pass the same directory as
# --data-root to the generators). Default: absolute devkit paths.
RELATIVE_TO="${RELATIVE_TO:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# ----------------------------------------------------------------------------

mkdir -p "$OUT_DIR"
[ -n "$RELATIVE_TO" ] && EXTRA_ARGS="--relative-to $RELATIVE_TO $EXTRA_ARGS"

run_pipeline cpu \
    python -m generate_pseudo_labels.create_infos \
        --root-path "$NUSC_ROOT" \
        --version "$VERSION" \
        --out-dir "$OUT_DIR" \
        --occ-path-prefix "$OCC_PATH_PREFIX" \
        $EXTRA_ARGS
