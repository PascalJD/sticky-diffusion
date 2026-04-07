#!/bin/bash -l
#SBATCH -A cis260242-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=sjd-cifar10-report

set -euo pipefail

ETA="${ETA:-0.8}"
LOGIT_TEMPERATURE="${LOGIT_TEMPERATURE:-1.0}"
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-50000}"
IS_NUM_SAMPLES="${IS_NUM_SAMPLES:-${FID_NUM_SAMPLES}}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-128}"
IS_BATCH_SIZE="${IS_BATCH_SIZE:-${FID_BATCH_SIZE}}"
BEST_CHECKPOINT_DIR="${BEST_CHECKPOINT_DIR:-/home/x-pjutrasdube/scratch/sticky-diffusion/runs/sticky-sjd_15767233/checkpoints/best}"
FID_STATS_DIR="${FID_STATS_DIR:-/home/x-pjutrasdube/scratch/sticky-diffusion/fid_stats}"
TFDS_DATA_DIR="${TFDS_DATA_DIR:-/home/x-pjutrasdube/scratch/sticky-diffusion/tfds/cifar10/3.0.2}"
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-}"
ENV_PREFIX="${ENV_PREFIX:-/anvil/scratch/x-pjutrasdube/envs/sticky}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sticky}"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-/home/x-pjutrasdube/scratch/sticky-diffusion/evals}"
RUN_TAG="eta_${ETA//./p}_tau_${LOGIT_TEMPERATURE//./p}"
RESULTS_ROOT="${RESULTS_ROOT:-${RESULTS_BASE_DIR}/sjd_cifar10_report_${RUN_TAG}_${SLURM_JOB_ID:-manual}}"
METRICS_PATH="${RESULTS_ROOT}/offline_eval_metrics.json"
HYDRA_DIR="${RESULTS_ROOT}/hydra"

mkdir -p "${RESULTS_ROOT}"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/anvil_python_env.sh"
sticky_resolve_python_env

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
echo "eta=${ETA}"
echo "logit_temperature=${LOGIT_TEMPERATURE}"
echo "fid_num_samples=${FID_NUM_SAMPLES}"
echo "is_num_samples=${IS_NUM_SAMPLES}"
echo "best_checkpoint_dir=${BEST_CHECKPOINT_DIR}"
echo "env_prefix=${ENV_PREFIX}"
echo "python_bin=${PYTHON_BIN}"

"${PYTHON_BIN}" -m sticky.cli.eval_checkpoint \
  experiment=cifar10/sjd_cifar10 \
  eval=cifar10_report \
  wandb.enabled=false \
  eval.fid_num_samples="${FID_NUM_SAMPLES}" \
  eval.is_num_samples="${IS_NUM_SAMPLES}" \
  eval.fid_batch_size="${FID_BATCH_SIZE}" \
  eval.is_batch_size="${IS_BATCH_SIZE}" \
  eval.fid_cache_dir="${FID_STATS_DIR}" \
  eval.fid_tfds_data_dir="${TFDS_DATA_DIR}" \
  offline_eval.force_fid=true \
  offline_eval.force_is=true \
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
metrics = payload.get("metrics", {})
fid = metrics.get("eval/fid")
is_mean = metrics.get("eval/is")
is_std = metrics.get("eval/is_std")
eta = payload.get("experiment_config", {}).get("jump_eta")
tau = payload.get("experiment_config", {}).get("sampler_logit_temperature")
print(
    f"[sjd-report] eta={eta} tau={tau} fid={fid} is={is_mean} is_std={is_std} metrics={metrics_path}",
    flush=True,
)
PY
