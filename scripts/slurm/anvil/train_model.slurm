#!/bin/bash
#SBATCH --job-name=sticky_train
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=24:00:00
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
    gpu_count="${SLURM_GPUS_ON_NODE:-${REQUIRED_LOCAL_DEVICES:-1}}"
    srun_export="ALL"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      srun_export="${srun_export},CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    fi
    srun --ntasks=1 --nodes=1 --cpu-bind=none --gpu-bind=none --gpus-per-node="${gpu_count}" --export="${srun_export}" "$@"
  else
    "$@"
  fi
}

if [[ -n "${SCRATCH:-}" ]]; then
  SCRATCH_ROOT="${SCRATCH}"
elif [[ -d "/anvil/scratch/${USER}" ]]; then
  SCRATCH_ROOT="/anvil/scratch/${USER}"
else
  SCRATCH_ROOT="${HOME}/scratch"
fi

MODEL="${MODEL:-}"
if [[ -z "${MODEL}" ]]; then
  echo "MODEL is required (for example: cadd, md4, sjd)." >&2
  exit 1
fi

EXPERIMENT_CFG="${EXPERIMENT_CFG:-}"
EVAL_CFG="${EVAL_CFG:-}"
if [[ -z "${EXPERIMENT_CFG}" ]]; then
  case "${MODEL}" in
    cadd) EXPERIMENT_CFG="cadd_cifar10" ;;
    md4) EXPERIMENT_CFG="md4_cifar10" ;;
    sjd) EXPERIMENT_CFG="sjd_cifar10" ;;
    *) echo "MODEL=${MODEL} requires EXPERIMENT_CFG to be set explicitly." >&2; exit 1 ;;
  esac
fi
if [[ -z "${EVAL_CFG}" ]]; then
  case "${MODEL}" in
    cadd) EVAL_CFG="cadd_cifar10" ;;
    md4) EVAL_CFG="sjd_cifar10" ;;
    sjd) EVAL_CFG="sjd_cifar10" ;;
    *) echo "MODEL=${MODEL} requires EVAL_CFG to be set explicitly." >&2; exit 1 ;;
  esac
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/sticky-diffusion/outputs}"
DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/sticky-diffusion/data/cifar10}"
RUN_TAG="${RUN_TAG:-${MODEL}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_TAG}"

PLATFORM="${PLATFORM:-auto}" # auto | single | pmap
WANDB_ENABLED="${WANDB_ENABLED:-true}"
EVAL_ENABLED="${EVAL_ENABLED:-false}"
SEED="${SEED:-0}"
BASELINE_ARCH_114M="${BASELINE_ARCH_114M:-false}"
DISABLE_AUGMENT="${DISABLE_AUGMENT:-false}"
DISABLE_CORRECTOR="${DISABLE_CORRECTOR:-false}"

BATCH_SIZE="${BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
TRAIN_STEPS="${TRAIN_STEPS:-}"
LOG_EVERY="${LOG_EVERY:-}"
METRICS_EVERY="${METRICS_EVERY:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-}"
LOG_IMAGES_EVERY="${LOG_IMAGES_EVERY:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
WARMUP_STEPS="${WARMUP_STEPS:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
B2="${B2:-}"
MODEL_TIMESTEPS="${MODEL_TIMESTEPS:-}"
SAMPLE_TIMESTEPS="${SAMPLE_TIMESTEPS:-}"

mkdir -p "${DATA_DIR}" "${RUN_DIR}"

if [[ "${PLATFORM}" == "pmap" ]]; then
  REQUIRED_LOCAL_DEVICES="${REQUIRED_LOCAL_DEVICES:-2}"
  echo "Preflight JAX device check (required local devices: ${REQUIRED_LOCAL_DEVICES})"
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
    echo "JAX preflight failed before training start." >&2
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true
    exit 1
  fi

  detected_local_devices="$(
    echo "${preflight_out}" | awk -F= '/^local_device_count=/{print $2}' | tail -n 1
  )"
  if [[ -z "${detected_local_devices}" ]]; then
    echo "Could not parse local_device_count from JAX preflight output." >&2
    exit 1
  fi
  if (( detected_local_devices < REQUIRED_LOCAL_DEVICES )); then
    echo "Expected >= ${REQUIRED_LOCAL_DEVICES} local JAX devices, got ${detected_local_devices}." >&2
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true
    exit 1
  fi
fi

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

BASE_OVERRIDES=(
  "experiment=${EXPERIMENT_CFG}"
  "eval=${EVAL_CFG}"
  "seed=${SEED}"
  "experiment.training.seed=${SEED}"
  "wandb.enabled=${WANDB_ENABLED}"
  "experiment.runtime.platform=${PLATFORM}"
  "experiment.dataset.data_dir=${DATA_DIR}"
  "eval.enabled=${EVAL_ENABLED}"
  "hydra.run.dir=${RUN_DIR}"
)

if [[ -n "${BATCH_SIZE}" ]]; then
  BASE_OVERRIDES+=("experiment.dataset.batch_size=${BATCH_SIZE}")
fi
if [[ -n "${EVAL_BATCH_SIZE}" ]]; then
  BASE_OVERRIDES+=("experiment.dataset.eval_batch_size=${EVAL_BATCH_SIZE}")
fi
if [[ -n "${TRAIN_STEPS}" ]]; then
  BASE_OVERRIDES+=("experiment.training.num_train_steps=${TRAIN_STEPS}")
fi
if [[ -n "${LOG_EVERY}" ]]; then
  BASE_OVERRIDES+=("experiment.training.log_every_steps=${LOG_EVERY}")
fi
if [[ -n "${METRICS_EVERY}" ]]; then
  BASE_OVERRIDES+=("experiment.training.metrics_every_steps=${METRICS_EVERY}")
fi
if [[ -n "${CHECKPOINT_EVERY}" ]]; then
  BASE_OVERRIDES+=("experiment.training.checkpoint_every_steps=${CHECKPOINT_EVERY}")
fi
if [[ -n "${SAVE_FINAL_CHECKPOINT}" ]]; then
  BASE_OVERRIDES+=("experiment.training.save_final_checkpoint=${SAVE_FINAL_CHECKPOINT}")
fi
if [[ -n "${LOG_IMAGES_EVERY}" ]]; then
  BASE_OVERRIDES+=("experiment.training.log_images_every_steps=${LOG_IMAGES_EVERY}")
fi
if [[ -n "${LEARNING_RATE}" ]]; then
  BASE_OVERRIDES+=("experiment.optim.learning_rate=${LEARNING_RATE}")
fi
if [[ -n "${WARMUP_STEPS}" ]]; then
  BASE_OVERRIDES+=("experiment.optim.warmup_steps=${WARMUP_STEPS}")
fi
if [[ -n "${WEIGHT_DECAY}" ]]; then
  BASE_OVERRIDES+=("experiment.optim.weight_decay=${WEIGHT_DECAY}")
fi
if [[ -n "${B2}" ]]; then
  BASE_OVERRIDES+=("experiment.optim.b2=${B2}")
fi
if [[ -n "${MODEL_TIMESTEPS}" ]]; then
  BASE_OVERRIDES+=("experiment.model.timesteps=${MODEL_TIMESTEPS}")
fi
if [[ -n "${SAMPLE_TIMESTEPS}" ]]; then
  BASE_OVERRIDES+=("experiment.training.sample_timesteps=${SAMPLE_TIMESTEPS}")
fi
if is_truthy "${DISABLE_AUGMENT}"; then
  BASE_OVERRIDES+=("experiment.dataset.augment.enabled=false")
fi
if is_truthy "${DISABLE_CORRECTOR}"; then
  BASE_OVERRIDES+=(
    "experiment.model.corrector_enabled=false"
    "experiment.model.corrector_steps=0"
    "experiment.model.corrector_remask_frac=0.0"
  )
fi
if is_truthy "${BASELINE_ARCH_114M}"; then
  BASE_OVERRIDES+=(
    "experiment.model.image_backbone=adm_unet5d"
    "experiment.model.feature_dim=96"
    "experiment.model.ch_mult=[3,4,4]"
    "experiment.model.adm_num_res_blocks=4"
    "experiment.model.adm_attention_resolutions=[2,4]"
    "experiment.model.adm_num_heads=4"
    "experiment.model.adm_num_head_channels=64"
    "experiment.model.adm_num_heads_upsample=-1"
    "experiment.model.adm_conv_resample=true"
    "experiment.model.adm_use_scale_shift_norm=true"
    "experiment.model.adm_resblock_updown=false"
    "experiment.model.adm_use_conv_skip=false"
  )
fi

EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
EXTRA_OVERRIDES_ARR=()
if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  read -r -a EXTRA_OVERRIDES_ARR <<< "${EXTRA_OVERRIDES}"
fi

EXTRA_OVERRIDES_FILE="${EXTRA_OVERRIDES_FILE:-}"
EXTRA_OVERRIDES_FILE_ARR=()
if [[ -n "${EXTRA_OVERRIDES_FILE}" ]]; then
  if [[ ! -f "${EXTRA_OVERRIDES_FILE}" ]]; then
    echo "EXTRA_OVERRIDES_FILE not found: ${EXTRA_OVERRIDES_FILE}" >&2
    exit 1
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    if [[ -z "${line}" ]] || [[ "${line}" == \#* ]]; then
      continue
    fi
    EXTRA_OVERRIDES_FILE_ARR+=("${line}")
  done < "${EXTRA_OVERRIDES_FILE}"
fi

echo "[$(date)] Host: $(hostname)"
echo "Model: ${MODEL}"
echo "Experiment cfg: ${EXPERIMENT_CFG}"
echo "Eval cfg: ${EVAL_CFG}"
echo "REPO_ROOT: ${REPO_ROOT}"
echo "Run tag: ${RUN_TAG}"
echo "Run dir: ${RUN_DIR}"
echo "Data dir: ${DATA_DIR}"
echo "Platform: ${PLATFORM}"
echo "W&B enabled: ${WANDB_ENABLED}"
echo "Eval enabled: ${EVAL_ENABLED}"
if [[ -n "${EXTRA_OVERRIDES_FILE}" ]]; then
  echo "Extra overrides file: ${EXTRA_OVERRIDES_FILE}"
fi
echo

run_job_step "${PYTHON_BIN}" -m sticky.entrypoints.train \
  "${BASE_OVERRIDES[@]}" \
  "${EXTRA_OVERRIDES_FILE_ARR[@]}" \
  "${EXTRA_OVERRIDES_ARR[@]}"
