#!/bin/bash
# Submit a SLURM job to the anvil cluster.
#
# Usage:
#   scripts/submit_anvil.sh <sbatch_path> [KEY=VAL ...] [-- <extra sbatch args>]
#
# Example:
#   scripts/submit_anvil.sh scripts/anvil_sudoku_train.sbatch \
#       EXPERIMENT=sudoku/mdlm_sudoku WANDB_ENABLED=false -- training.max_steps=100
#
# Env vars:
#   ANVIL_HOST       SSH alias or user@host for anvil (default: anvil)
#   ANVIL_REPO_DIR   Path to the sticky-diffusion clone on anvil
#                    (default: ~/projects/sticky-diffusion)

set -euo pipefail

ANVIL_HOST="${ANVIL_HOST:-anvil}"
ANVIL_REPO_DIR="${ANVIL_REPO_DIR:-~/projects/sticky-diffusion}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sbatch_path> [KEY=VAL ...] [-- <extra sbatch args>]" >&2
  exit 2
fi

sbatch_script="$1"
shift

env_exports=()
extra_args=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then
    shift
    extra_args=("$@")
    break
  fi
  if [[ "$1" != *"="* ]]; then
    echo "Expected KEY=VAL or '--', got: $1" >&2
    exit 2
  fi
  env_exports+=("$1")
  shift
done

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${branch}" == "HEAD" ]]; then
  echo "Detached HEAD; check out a branch before submitting." >&2
  exit 1
fi

echo "Pushing branch '${branch}' to origin..."
git push origin "${branch}"

export_arg="ALL"
if (( ${#env_exports[@]} > 0 )); then
  for kv in "${env_exports[@]}"; do
    export_arg+=",${kv}"
  done
fi

remote_extra_args=""
if (( ${#extra_args[@]} > 0 )); then
  for arg in "${extra_args[@]}"; do
    remote_extra_args+=" $(printf '%q' "${arg}")"
  done
fi

echo "Submitting ${sbatch_script} on ${ANVIL_HOST}..."
ssh "${ANVIL_HOST}" bash -l <<EOF
set -euo pipefail
cd ${ANVIL_REPO_DIR}
git fetch origin
git checkout "${branch}"
git pull --ff-only origin "${branch}"
sbatch --export="${export_arg}" "${sbatch_script}"${remote_extra_args}
EOF
