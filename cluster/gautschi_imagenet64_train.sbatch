#!/bin/bash -l
#SBATCH -A ruqiz
#SBATCH -p ai
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=imgnet64

set -euo pipefail

EXPERIMENT="${EXPERIMENT:-imagenet64/sjd_imagenet64}"
EVAL_CONFIG="${EVAL_CONFIG:-imagenet64}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
REPO_ROOT="${REPO_ROOT:-$HOME/sticky-diffusion}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sticky}"
ENV_PREFIX="${ENV_PREFIX:-}"
SCRATCH="${RCAC_SCRATCH:-/scratch/gautschi/${USER}}"
RUNS_BASE_DIR="${RUNS_BASE_DIR:-${SCRATCH}/sticky-diffusion/runs}"
RUN_TAG="${RUN_TAG:-${EXPERIMENT}_${SLURM_JOB_ID:-manual}}"
RUN_DIR="${RUN_DIR:-${RUNS_BASE_DIR}/${RUN_TAG}}"

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/cluster/gautschi_python_env.sh"
gautschi_resolve_python_env

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
# Point TFDS data cache to scratch.
export TFDS_DATA_DIR="${SCRATCH}/sticky-diffusion/data/tfds"

echo "repo_root=${REPO_ROOT}"
echo "run_dir=${RUN_DIR}"
echo "experiment=${EXPERIMENT}"
echo "eval_config=${EVAL_CONFIG}"
echo "wandb_enabled=${WANDB_ENABLED}"
echo "scratch=${SCRATCH}"
echo "extra_overrides=$*"

"${PYTHON_BIN}" -m sticky.cli.train \
  "experiment=${EXPERIMENT}" \
  "eval=${EVAL_CONFIG}" \
  "wandb.enabled=${WANDB_ENABLED}" \
  "hydra.run.dir=${RUN_DIR}" \
  "dataset.data_dir=${SCRATCH}/sticky-diffusion/data/tfds" \
  "eval.fid_cache_dir=${SCRATCH}/sticky-diffusion/data/fid_stats" \
  "$@"
