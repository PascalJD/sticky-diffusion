#!/bin/bash
#SBATCH --job-name=sticky_anchor_eval
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "REPO_ROOT not found: ${REPO_ROOT}" >&2
  exit 1
fi
cd "${REPO_ROOT}"
if [[ ! -f "${REPO_ROOT}/src/sticky/__init__.py" ]]; then
  echo "REPO_ROOT does not appear to be sticky-diffusion root: ${REPO_ROOT}" >&2
  exit 1
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset PYTHONHOME
ANVIL_MODULES="${ANVIL_MODULES:-}"

ensure_module_cmd() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
  command -v module >/dev/null 2>&1
}

ensure_conda_cmd() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  if ensure_module_cmd; then
    CONDA_MODULE_CANDIDATES="${CONDA_MODULE_CANDIDATES:-anaconda/2023.09 anaconda/2024.02 anaconda/2024.02-py311 anaconda}"
    for mod in ${CONDA_MODULE_CANDIDATES}; do
      if module load "${mod}" >/dev/null 2>&1; then
        if command -v conda >/dev/null 2>&1; then
          echo "Auto-loaded conda module: ${mod}"
          return 0
        fi
      fi
    done
  fi

  if [[ -f "${HOME}/.bashrc" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc" >/dev/null 2>&1 || true
  fi
  command -v conda >/dev/null 2>&1
}

if [[ -n "${ANVIL_MODULES:-}" ]]; then
  if ensure_module_cmd; then
    module --force purge || true
    for mod in ${ANVIL_MODULES}; do
      module load "${mod}"
    done
  else
    echo "ANVIL_MODULES was set but module command is unavailable." >&2
  fi
fi

CONDA_ENV="${CONDA_ENV:-sticky}"
PYTHON_BIN="${PYTHON_BIN:-}"

activate_conda_env() {
  local env_spec="$1"
  if conda activate "${env_spec}" >/dev/null 2>&1; then
    echo "Activated conda env: ${env_spec}"
    return 0
  fi

  if [[ "${env_spec}" != /* ]]; then
    local candidates=()
    if [[ -n "${SCRATCH:-}" ]]; then
      candidates+=("${SCRATCH}/envs/${env_spec}")
    fi
    candidates+=("/anvil/scratch/${USER}/envs/${env_spec}")

    local env_path
    for env_path in "${candidates[@]}"; do
      if [[ -d "${env_path}" ]] && conda activate "${env_path}" >/dev/null 2>&1; then
        echo "Activated conda env via path fallback: ${env_path}"
        return 0
      fi
    done
  fi

  return 1
}

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    if [[ ! -x "${PYTHON_BIN}" ]]; then
      echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
      return 1
    fi
    echo "Using explicit python: ${PYTHON_BIN}"
    return 0
  fi

  if [[ "${CONDA_ENV}" == /* ]] && [[ -x "${CONDA_ENV}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_ENV}/bin/python"
    echo "Using env python: ${PYTHON_BIN}"
    return 0
  fi

  if ! ensure_conda_cmd; then
    echo "conda command not found. Set CONDA_ENV to a full env path or set ANVIL_MODULES appropriately." >&2
    return 1
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! activate_conda_env "${CONDA_ENV}"; then
    echo "Failed to activate CONDA_ENV=${CONDA_ENV}. If this is a path-based env, pass its full path." >&2
    return 1
  fi

  PYTHON_BIN="$(command -v python)"
  echo "Using activated python: ${PYTHON_BIN}"
  return 0
}

if ! resolve_python_bin; then
  exit 1
fi

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

run_job_step() {
  if [[ -n "${SLURM_JOB_ID:-}" ]] && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    "$@"
    return
  fi
  if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    local gpu_count
    local srun_export
    gpu_count="${SLURM_GPUS_ON_NODE:-1}"
    srun_export="ALL"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      srun_export="${srun_export},CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    fi
    srun --ntasks=1 --nodes=1 --cpu-bind=none --gpu-bind=none --gpus-per-node="${gpu_count}" --export="${srun_export}" "$@"
  else
    "$@"
  fi
}

EXPERIMENT_CFG="${EXPERIMENT_CFG:-sjd_anchor_study_cifar10}"
EVAL_CFG="${EVAL_CFG:-sjd_anchor_study_cifar10}"
PLATFORM="${PLATFORM:-single}"
SEED="${SEED:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-}"
if [[ -z "${TRAIN_RUN_DIR}" ]]; then
  echo "TRAIN_RUN_DIR is required." >&2
  exit 1
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-final}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-null}"
USE_EMA="${USE_EMA:-true}"

NFE_BUDGET="${NFE_BUDGET:-256}"
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-10000}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-128}"
IS_ENABLED="${IS_ENABLED:-false}"
IS_NUM_SAMPLES="${IS_NUM_SAMPLES:-${FID_NUM_SAMPLES}}"
IS_BATCH_SIZE="${IS_BATCH_SIZE:-${FID_BATCH_SIZE}}"
FORCE_IS="${FORCE_IS:-false}"

ETA_OVERRIDE="${ETA_OVERRIDE:-}"
TAU_OVERRIDE="${TAU_OVERRIDE:-}"

SAMPLER_PROBE_BATCHES="${SAMPLER_PROBE_BATCHES:-0}"
SAMPLER_PROBE_BATCH_SIZE="${SAMPLER_PROBE_BATCH_SIZE:-${FID_BATCH_SIZE}}"
SAMPLER_PROBE_SEED_OFFSET="${SAMPLER_PROBE_SEED_OFFSET:-777777}"

if [[ -n "${SCRATCH:-}" ]]; then
  SCRATCH_ROOT="${SCRATCH}"
elif [[ -d "/anvil/scratch/${USER}" ]]; then
  SCRATCH_ROOT="/anvil/scratch/${USER}"
else
  SCRATCH_ROOT="${HOME}/scratch"
fi

DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/sticky-diffusion/data/cifar10}"
FID_CACHE_DIR="${FID_CACHE_DIR:-${SCRATCH_ROOT}/sticky-diffusion/data/fid_stats}"
FID_TFDS_DATA_DIR="${FID_TFDS_DATA_DIR:-${DATA_DIR}}"
RUN_DIR="${RUN_DIR:-${SCRATCH_ROOT}/sticky-diffusion/outputs/offline_eval/sjd_anchor_study}"
METRICS_PATH="${METRICS_PATH:-${RUN_DIR}/offline_eval_metrics.json}"
SUMMARY_PATH="${SUMMARY_PATH:-${RUN_DIR}/summary.json}"
CANDIDATE_NAME="${CANDIDATE_NAME:-unknown}"
SWEEP_MANIFEST="${SWEEP_MANIFEST:-}"
AGGREGATE_DIR="${AGGREGATE_DIR:-}"

mkdir -p "${RUN_DIR}" "${FID_CACHE_DIR}" "${DATA_DIR}"

echo "[$(date)] Host: $(hostname)"
echo "REPO_ROOT: ${REPO_ROOT}"
echo "Train run dir: ${TRAIN_RUN_DIR}"
echo "Checkpoint source: ${CHECKPOINT_SOURCE}"
echo "Checkpoint step: ${CHECKPOINT_STEP}"
echo "Candidate: ${CANDIDATE_NAME}"
echo "NFE budget: ${NFE_BUDGET}"
echo "FID samples: ${FID_NUM_SAMPLES}"
echo "FID batch size: ${FID_BATCH_SIZE}"
echo "IS enabled: ${IS_ENABLED}"
echo "ETA override: ${ETA_OVERRIDE:-<inherit>}"
echo "TAU override: ${TAU_OVERRIDE:-<inherit>}"
echo "Run dir: ${RUN_DIR}"
echo "Metrics path: ${METRICS_PATH}"
echo "Summary path: ${SUMMARY_PATH}"
echo

if [[ "${JAX_PLATFORMS}" == *cuda* ]]; then
  echo "Preflight JAX device check for offline eval"
  set +e
  preflight_out="$({
    echo "batch_host=$(hostname)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
    echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
    echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
    nvidia-smi -L || true
    run_job_step bash -lc '
      echo "step_host=$(hostname)"
      echo "step.CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
      echo "step.SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
      echo "step.SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
      echo "step.SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
      nvidia-smi -L || true
    '
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 \
    run_job_step "${PYTHON_BIN}" - <<'PY'
import os
import socket
import jax

print("step_host=" + socket.gethostname())
print("CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", ""))
print("SLURM_JOB_GPUS=" + os.environ.get("SLURM_JOB_GPUS", ""))
print("SLURM_STEP_GPUS=" + os.environ.get("SLURM_STEP_GPUS", ""))
print("SLURM_GPUS_ON_NODE=" + os.environ.get("SLURM_GPUS_ON_NODE", ""))
print(f"local_device_count={jax.local_device_count()}")
print("devices=" + ", ".join(str(d) for d in jax.devices()))
PY
  } 2>&1)"
  preflight_rc=$?
  set -e
  echo "${preflight_out}"
  if (( preflight_rc != 0 )); then
    echo "JAX preflight failed before offline eval start." >&2
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true
    exit 1
  fi
fi

CMD=(
  "${PYTHON_BIN}" -m sticky.entrypoints.eval_checkpoint
  "experiment=${EXPERIMENT_CFG}"
  "eval=${EVAL_CFG}"
  "experiment.training.seed=${SEED}"
  "experiment.dataset.data_dir=${DATA_DIR}"
  "experiment.runtime.platform=${PLATFORM}"
  "eval.enabled=true"
  "eval.run_at_end=true"
  "eval.fid_enabled=true"
  "eval.fid_every=0"
  "eval.fid_num_samples=${FID_NUM_SAMPLES}"
  "eval.fid_batch_size=${FID_BATCH_SIZE}"
  "eval.fid_dataset_name=cifar10"
  "eval.fid_split=train"
  "eval.fid_cache_dir=${FID_CACHE_DIR}"
  "eval.fid_tfds_data_dir=${FID_TFDS_DATA_DIR}"
  "eval.is_enabled=${IS_ENABLED}"
  "eval.is_every=0"
  "eval.is_num_samples=${IS_NUM_SAMPLES}"
  "eval.is_batch_size=${IS_BATCH_SIZE}"
  "wandb.enabled=${WANDB_ENABLED}"
  "offline_eval.run_dir=${TRAIN_RUN_DIR}"
  "offline_eval.use_run_config=true"
  "offline_eval.checkpoint_source=${CHECKPOINT_SOURCE}"
  "offline_eval.checkpoint_step=${CHECKPOINT_STEP}"
  "offline_eval.use_ema=${USE_EMA}"
  "offline_eval.sample_timesteps=${NFE_BUDGET}"
  "offline_eval.force_fid=true"
  "offline_eval.force_is=${FORCE_IS}"
  "offline_eval.sampler_probe_batches=${SAMPLER_PROBE_BATCHES}"
  "offline_eval.sampler_probe_batch_size=${SAMPLER_PROBE_BATCH_SIZE}"
  "offline_eval.sampler_probe_seed_offset=${SAMPLER_PROBE_SEED_OFFSET}"
  "offline_eval.output_path=${METRICS_PATH}"
  "hydra.run.dir=${RUN_DIR}"
)

if [[ -n "${CHECKPOINT_DIR}" ]]; then
  CMD+=("offline_eval.checkpoint_dir=${CHECKPOINT_DIR}")
fi
if [[ -n "${ETA_OVERRIDE}" ]]; then
  CMD+=("offline_eval.jump_eta=${ETA_OVERRIDE}")
fi
if [[ -n "${TAU_OVERRIDE}" ]]; then
  CMD+=("offline_eval.logit_temperature=${TAU_OVERRIDE}")
fi

run_job_step "${CMD[@]}"

"${PYTHON_BIN}" - "${METRICS_PATH}" "${SUMMARY_PATH}" "${CANDIDATE_NAME}" "${SEED}" "${NFE_BUDGET}" "${TRAIN_RUN_DIR}" "${ETA_OVERRIDE}" "${TAU_OVERRIDE}" <<'PY'
import json
import pathlib
import sys

metrics_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
candidate = sys.argv[3]
seed = int(sys.argv[4])
nfe_budget = int(sys.argv[5])
train_run_dir = sys.argv[6]
requested_eta = sys.argv[7]
requested_tau = sys.argv[8]

payload = json.loads(metrics_path.read_text(encoding="utf-8"))
metrics = payload.get("metrics", {})
experiment_config = payload.get("experiment_config", {})
checkpoint = payload.get("checkpoint", {})
evaluation = payload.get("evaluation", {})

summary = {
    "candidate": candidate,
    "seed": seed,
    "nfe_budget": nfe_budget,
    "fid": metrics.get("eval/fid"),
    "is": metrics.get("eval/is"),
    "is_std": metrics.get("eval/is_std"),
    "sample_timesteps": evaluation.get("sample_timesteps"),
    "fid_num_samples": evaluation.get("fid_num_samples"),
    "effective_eta": experiment_config.get("jump_eta"),
    "effective_logit_temperature": experiment_config.get("sampler_logit_temperature"),
    "requested_eta": None if requested_eta == "" else float(requested_eta),
    "requested_tau": None if requested_tau == "" else float(requested_tau),
    "train_run_dir": train_run_dir,
    "checkpoint_path": checkpoint.get("checkpoint_path"),
    "checkpoint_source": checkpoint.get("source"),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[offline-eval] Wrote summary to {summary_path}", flush=True)
PY

if [[ -n "${SWEEP_MANIFEST}" ]] && [[ -n "${AGGREGATE_DIR}" ]]; then
  mkdir -p "${AGGREGATE_DIR}"
  "${PYTHON_BIN}" -m sticky.entrypoints.aggregate_anchor_eval \
    --manifest "${SWEEP_MANIFEST}" \
    --output-dir "${AGGREGATE_DIR}" || true
fi
