#!/bin/bash -l
#SBATCH -A cis260242-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --job-name=sjd-logit-temp
#SBATCH --array=0-5

set -euo pipefail

TEMPERATURES=(0.70 0.75 0.80 0.85 0.90 0.95)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#TEMPERATURES[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} is out of range for ${#TEMPERATURES[@]} temperatures." >&2
  exit 1
fi

LOGIT_TEMPERATURE="${TEMPERATURES[$TASK_ID]}"
ETA="${ETA:-0.8}"
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-10000}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-128}"
BEST_CHECKPOINT_DIR="${BEST_CHECKPOINT_DIR:-/home/x-pjutrasdube/scratch/sticky-diffusion/runs/sticky-sjd_15767233/checkpoints/best}"
FID_STATS_DIR="${FID_STATS_DIR:-/home/x-pjutrasdube/scratch/sticky-diffusion/fid_stats}"
TFDS_DATA_DIR="${TFDS_DATA_DIR:-/home/x-pjutrasdube/scratch/sticky-diffusion/tfds/cifar10/3.0.2}"
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-}"
ENV_PREFIX="${ENV_PREFIX:-/anvil/scratch/x-pjutrasdube/envs/sticky}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sticky}"
SWEEP_ID="${SWEEP_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-/home/x-pjutrasdube/scratch/sticky-diffusion/evals}"
RESULTS_ROOT="${RESULTS_ROOT:-${RESULTS_BASE_DIR}/sjd_logit_temperature_sweep_${SWEEP_ID}}"
TEMP_TAG="tau_${LOGIT_TEMPERATURE/./p}"
TASK_OUTPUT_DIR="${RESULTS_ROOT}/${TEMP_TAG}"
METRICS_PATH="${TASK_OUTPUT_DIR}/offline_eval_metrics.json"
HYDRA_DIR="${TASK_OUTPUT_DIR}/hydra"

mkdir -p "${TASK_OUTPUT_DIR}"
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

CHECKPOINT_PARENT="$(dirname "${BEST_CHECKPOINT_DIR}")"
RUN_DIR="$(dirname "${CHECKPOINT_PARENT}")"

declare -a CHECKPOINT_ARGS
if [[ -f "${RUN_DIR}/run_context.json" ]]; then
  CHECKPOINT_ARGS=(
    "offline_eval.run_dir=${RUN_DIR}"
    "offline_eval.use_run_config=true"
    "offline_eval.checkpoint_source=best"
  )
else
  CHECKPOINT_ARGS=(
    "offline_eval.checkpoint_dir=${BEST_CHECKPOINT_DIR}"
    "offline_eval.checkpoint_source=root"
  )
fi

echo "repo_root=${REPO_ROOT}"
echo "results_root=${RESULTS_ROOT}"
echo "logit_temperature=${LOGIT_TEMPERATURE}"
echo "jump_eta=${ETA}"
echo "fid_num_samples=${FID_NUM_SAMPLES}"
echo "best_checkpoint_dir=${BEST_CHECKPOINT_DIR}"
echo "env_prefix=${ENV_PREFIX}"
echo "python_bin=${PYTHON_BIN}"

"${PYTHON_BIN}" -m sticky.entrypoints.eval_checkpoint \
  experiment=sjd_cifar10 \
  eval=cifar10 \
  wandb.enabled=false \
  eval.fid_enabled=true \
  eval.is_enabled=false \
  eval.fid_num_samples="${FID_NUM_SAMPLES}" \
  eval.fid_batch_size="${FID_BATCH_SIZE}" \
  eval.fid_cache_dir="${FID_STATS_DIR}" \
  eval.fid_tfds_data_dir="${TFDS_DATA_DIR}" \
  offline_eval.force_fid=true \
  offline_eval.force_is=false \
  offline_eval.jump_eta="${ETA}" \
  offline_eval.logit_temperature="${LOGIT_TEMPERATURE}" \
  offline_eval.output_path="${METRICS_PATH}" \
  hydra.run.dir="${HYDRA_DIR}" \
  "${CHECKPOINT_ARGS[@]}"

"${PYTHON_BIN}" - <<'PY' "${METRICS_PATH}"
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
payload = json.loads(metrics_path.read_text(encoding="utf-8"))
fid = payload.get("metrics", {}).get("eval/fid")
tau = payload.get("experiment_config", {}).get("sampler_logit_temperature")
eta = payload.get("experiment_config", {}).get("jump_eta")
print(f"[sjd-sweep] tau={tau} eta={eta} fid={fid} metrics={metrics_path}", flush=True)
PY
