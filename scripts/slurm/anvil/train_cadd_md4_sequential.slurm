#!/bin/bash
#SBATCH --job-name=sticky_cadd_md4
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=72:00:00
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
  echo "REPO_ROOT does not look like sticky-diffusion: ${REPO_ROOT}" >&2
  exit 1
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

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
    CONDA_MODULE_CANDIDATES="${CONDA_MODULE_CANDIDATES:-anaconda/2023.09 anaconda/2024.02 anaconda}"
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
    module purge || true
    for mod in ${ANVIL_MODULES}; do
      module load "${mod}"
    done
  else
    echo "ANVIL_MODULES set but module command unavailable." >&2
  fi
fi

CONDA_ENV="${CONDA_ENV:-sticky}"
if ! ensure_conda_cmd; then
  echo "Conda not found. Set ANVIL_MODULES (example: anaconda/2023.09)." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

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
        echo "Activated conda env via fallback path: ${env_path}"
        return 0
      fi
    done
  fi

  return 1
}

if ! activate_conda_env "${CONDA_ENV}"; then
  echo "Failed to activate CONDA_ENV=${CONDA_ENV}" >&2
  exit 1
fi

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

if [[ -n "${SCRATCH:-}" ]]; then
  SCRATCH_ROOT="${SCRATCH}"
elif [[ -d "/anvil/scratch/${USER}" ]]; then
  SCRATCH_ROOT="/anvil/scratch/${USER}"
else
  SCRATCH_ROOT="${HOME}/scratch"
fi

DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/sticky-diffusion/data/cifar10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/sticky-diffusion/outputs/baselines}"
RUN_TAG="${RUN_TAG:-cadd_md4_baseline_$(date +%Y%m%d_%H%M%S)}"

PLATFORM="${PLATFORM:-pmap}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
SEED="${SEED:-0}"

BATCH_SIZE="${BATCH_SIZE:-256}"
BATCH_SIZE_CADD="${BATCH_SIZE_CADD:-${BATCH_SIZE}}"
BATCH_SIZE_MD4="${BATCH_SIZE_MD4:-${BATCH_SIZE}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
EVAL_BATCH_SIZE_CADD="${EVAL_BATCH_SIZE_CADD:-${BATCH_SIZE_CADD}}"
EVAL_BATCH_SIZE_MD4="${EVAL_BATCH_SIZE_MD4:-${BATCH_SIZE_MD4}}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
B2="${B2:-0.99}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-512}"

LOG_EVERY="${LOG_EVERY:-1000}"
METRICS_EVERY="${METRICS_EVERY:-1000}"
LOG_IMAGES_EVERY="${LOG_IMAGES_EVERY:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-true}"

mkdir -p "${DATA_DIR}" "${OUTPUT_ROOT}/${RUN_TAG}/cadd" "${OUTPUT_ROOT}/${RUN_TAG}/md4"

if [[ "${PLATFORM}" == "pmap" ]]; then
  REQUIRED_LOCAL_DEVICES="${REQUIRED_LOCAL_DEVICES:-${GPUS_PER_NODE:-2}}"
  echo "Preflight JAX device check (required local devices: ${REQUIRED_LOCAL_DEVICES})"
  preflight_out="$(
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 \
    python - <<'PY'
import os
import jax

print(f"local_device_count={jax.local_device_count()}")
print("devices=" + ", ".join(str(d) for d in jax.devices()))
print("CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", ""))
PY
  )"
  preflight_rc=$?
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

COMMON_OVERRIDES=(
  "wandb.enabled=${WANDB_ENABLED}"
  "experiment.runtime.platform=${PLATFORM}"
  "experiment.dataset.data_dir=${DATA_DIR}"
  "experiment.dataset.augment.enabled=false"
  "experiment.optim.learning_rate=${LEARNING_RATE}"
  "experiment.optim.warmup_steps=${WARMUP_STEPS}"
  "experiment.optim.weight_decay=${WEIGHT_DECAY}"
  "experiment.optim.b2=${B2}"
  "experiment.training.seed=${SEED}"
  "experiment.training.num_train_steps=${TRAIN_STEPS}"
  "experiment.training.log_every_steps=${LOG_EVERY}"
  "experiment.training.metrics_every_steps=${METRICS_EVERY}"
  "experiment.training.eval_every_steps=0"
  "experiment.training.checkpoint_every_steps=${CHECKPOINT_EVERY}"
  "experiment.training.save_final_checkpoint=${SAVE_FINAL_CHECKPOINT}"
  "experiment.training.log_images_every_steps=${LOG_IMAGES_EVERY}"
  "experiment.training.sample_timesteps=${DIFFUSION_STEPS}"
  "eval.enabled=false"
  "eval.run_at_end=false"
  "eval.fid_enabled=false"
  "eval.fid_every=0"
  "eval.is_enabled=false"
  "eval.is_every=0"
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
  "experiment.model.timesteps=${DIFFUSION_STEPS}"
)

EXTRA_COMMON_ARR=()
if [[ -n "${EXTRA_OVERRIDES_COMMON:-}" ]]; then
  read -r -a EXTRA_COMMON_ARR <<< "${EXTRA_OVERRIDES_COMMON}"
fi

EXTRA_CADD_ARR=()
if [[ -n "${EXTRA_OVERRIDES_CADD:-}" ]]; then
  read -r -a EXTRA_CADD_ARR <<< "${EXTRA_OVERRIDES_CADD}"
fi

EXTRA_MD4_ARR=()
if [[ -n "${EXTRA_OVERRIDES_MD4:-}" ]]; then
  read -r -a EXTRA_MD4_ARR <<< "${EXTRA_OVERRIDES_MD4}"
fi

CADD_RUN_DIR="${OUTPUT_ROOT}/${RUN_TAG}/cadd"
CADD_OVERRIDES=(
  "experiment=cadd_cifar10"
  "eval=cadd_cifar10"
  "hydra.run.dir=${CADD_RUN_DIR}"
  "experiment.dataset.batch_size=${BATCH_SIZE_CADD}"
  "experiment.dataset.eval_batch_size=${EVAL_BATCH_SIZE_CADD}"
  "experiment.model.corrector_enabled=false"
  "experiment.model.corrector_steps=0"
  "experiment.model.corrector_remask_frac=0.0"
)

MD4_RUN_DIR="${OUTPUT_ROOT}/${RUN_TAG}/md4"
MD4_OVERRIDES=(
  "experiment=md4_cifar10"
  "eval=sjd_cifar10"
  "hydra.run.dir=${MD4_RUN_DIR}"
  "experiment.dataset.batch_size=${BATCH_SIZE_MD4}"
  "experiment.dataset.eval_batch_size=${EVAL_BATCH_SIZE_MD4}"
)

MODEL_PHASE="${MODEL_PHASE:-both}" # cadd | md4 | both
case "${MODEL_PHASE}" in
  cadd|md4|both) ;;
  *)
    echo "Invalid MODEL_PHASE=${MODEL_PHASE}; expected one of: cadd, md4, both" >&2
    exit 1
    ;;
esac

echo "[$(date)] Host: $(hostname)"
echo "Run tag: ${RUN_TAG}"
echo "Data dir: ${DATA_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Platform: ${PLATFORM}"
echo "Batch size (CADD): ${BATCH_SIZE_CADD}"
echo "Batch size (MD4):  ${BATCH_SIZE_MD4}"
echo "Train steps: ${TRAIN_STEPS}"
echo "Diffusion steps: ${DIFFUSION_STEPS}"
echo "Model phase: ${MODEL_PHASE}"
echo

if [[ "${MODEL_PHASE}" == "cadd" || "${MODEL_PHASE}" == "both" ]]; then
  echo "=== Training CADD ==="
  python -m sticky.entrypoints.train \
    "${COMMON_OVERRIDES[@]}" \
    "${EXTRA_COMMON_ARR[@]}" \
    "${CADD_OVERRIDES[@]}" \
    "${EXTRA_CADD_ARR[@]}"
fi

if [[ "${MODEL_PHASE}" == "md4" || "${MODEL_PHASE}" == "both" ]]; then
  echo "=== Training MD4 ==="
  python -m sticky.entrypoints.train \
    "${COMMON_OVERRIDES[@]}" \
    "${EXTRA_COMMON_ARR[@]}" \
    "${MD4_OVERRIDES[@]}" \
    "${EXTRA_MD4_ARR[@]}"
fi

if [[ "${MODEL_PHASE}" == "both" ]]; then
  echo "[$(date)] Completed: CADD then MD4."
else
  echo "[$(date)] Completed phase: ${MODEL_PHASE}."
fi
