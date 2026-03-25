#!/bin/bash -l
#SBATCH -A cis260242-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=sudoku-train

set -euo pipefail

EXPERIMENT="${EXPERIMENT:-mdlm_sudoku_tfw_top_prob_margin}"
EVAL_CONFIG="${EVAL_CONFIG:-sudoku_mdlm}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-}"
ENV_PREFIX="${ENV_PREFIX:-/anvil/scratch/x-pjutrasdube/envs/sticky}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sticky}"
RUNS_BASE_DIR="${RUNS_BASE_DIR:-/home/x-pjutrasdube/scratch/sticky-diffusion/runs}"
RUN_TAG="${RUN_TAG:-${EXPERIMENT}_${SLURM_JOB_ID:-manual}}"
RUN_DIR="${RUN_DIR:-${RUNS_BASE_DIR}/${RUN_TAG}}"

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

if [[ -d "${ENV_PREFIX}" ]]; then
  export PATH="${ENV_PREFIX}/bin:${PATH}"
  if [[ -f "${ENV_PREFIX}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_PREFIX}/bin/activate"
  fi
  PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Could not find python. Set ENV_PREFIX or PYTHON_BIN explicitly." >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  RESOLVED_PYTHON="$(command -v "${PYTHON_BIN}" || true)"
  if [[ -z "${RESOLVED_PYTHON}" ]]; then
    echo "Python executable is not available: ${PYTHON_BIN}" >&2
    exit 1
  fi
  PYTHON_BIN="${RESOLVED_PYTHON}"
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

echo "repo_root=${REPO_ROOT}"
echo "run_dir=${RUN_DIR}"
echo "experiment=${EXPERIMENT}"
echo "eval_config=${EVAL_CONFIG}"
echo "wandb_enabled=${WANDB_ENABLED}"
echo "env_prefix=${ENV_PREFIX}"
echo "python_bin=${PYTHON_BIN}"
echo "extra_overrides=$*"

"${PYTHON_BIN}" -m sticky.entrypoints.train \
  "experiment=${EXPERIMENT}" \
  "eval=${EVAL_CONFIG}" \
  "wandb.enabled=${WANDB_ENABLED}" \
  "hydra.run.dir=${RUN_DIR}" \
  "$@"
