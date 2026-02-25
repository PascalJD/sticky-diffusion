#!/bin/bash
#SBATCH --job-name=cadd_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "REPO_ROOT not found: ${REPO_ROOT}" >&2
  exit 1
fi
cd "${REPO_ROOT}"

# Optional module list, e.g.:
#   export ANVIL_MODULES="anaconda/2023.09 cuda/12.2"
if [[ -n "${ANVIL_MODULES:-}" ]]; then
  if command -v module >/dev/null 2>&1; then
    module purge || true
    for mod in ${ANVIL_MODULES}; do
      module load "${mod}"
    done
  else
    echo "ANVIL_MODULES was set but 'module' command is unavailable." >&2
  fi
fi

CONDA_ENV="${CONDA_ENV:-sticky}"
if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Load the appropriate Anvil module first." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [[ -n "${SCRATCH:-}" ]]; then
  SCRATCH_ROOT="${SCRATCH}"
elif [[ -d "/anvil/scratch/${USER}" ]]; then
  SCRATCH_ROOT="/anvil/scratch/${USER}"
else
  SCRATCH_ROOT="${HOME}/scratch"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/sticky-diffusion/outputs/smoke}"
DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/sticky-diffusion/data/cifar10}"
RUN_TAG="${RUN_TAG:-cadd_smoke_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_TAG}"

PLATFORM="${PLATFORM:-single}" # single | pmap
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"

SMOKE_STEPS="${SMOKE_STEPS:-200}"
LOG_EVERY="${LOG_EVERY:-20}"
METRICS_EVERY="${METRICS_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-false}"
LOG_IMAGES_EVERY="${LOG_IMAGES_EVERY:-0}"
EVAL_ENABLED="${EVAL_ENABLED:-false}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

mkdir -p "${DATA_DIR}" "${RUN_DIR}"

BASE_OVERRIDES_DEFAULT="\
experiment=cadd_cifar10 \
eval=cadd_cifar10 \
wandb.enabled=${WANDB_ENABLED} \
experiment.runtime.platform=${PLATFORM} \
experiment.dataset.data_dir=${DATA_DIR} \
experiment.dataset.batch_size=${BATCH_SIZE} \
experiment.dataset.eval_batch_size=${EVAL_BATCH_SIZE} \
experiment.training.num_train_steps=${SMOKE_STEPS} \
experiment.training.log_every_steps=${LOG_EVERY} \
experiment.training.metrics_every_steps=${METRICS_EVERY} \
experiment.training.checkpoint_every_steps=${CHECKPOINT_EVERY} \
experiment.training.save_final_checkpoint=${SAVE_FINAL_CHECKPOINT} \
experiment.training.log_images_every_steps=${LOG_IMAGES_EVERY} \
eval.enabled=${EVAL_ENABLED} \
hydra.run.dir=${RUN_DIR}"

BASE_OVERRIDES="${BASE_OVERRIDES:-${BASE_OVERRIDES_DEFAULT}}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

read -r -a BASE_OVERRIDES_ARR <<< "${BASE_OVERRIDES}"
EXTRA_OVERRIDES_ARR=()
if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  read -r -a EXTRA_OVERRIDES_ARR <<< "${EXTRA_OVERRIDES}"
fi

echo "[$(date)] Host: $(hostname)"
echo "REPO_ROOT: ${REPO_ROOT}"
echo "Run dir: ${RUN_DIR}"
echo "Data dir: ${DATA_DIR}"
echo "Platform: ${PLATFORM}"
echo "Batch size: ${BATCH_SIZE}"
echo "Smoke steps: ${SMOKE_STEPS}"
echo

python -m sticky.entrypoints.train \
  "${BASE_OVERRIDES_ARR[@]}" \
  "${EXTRA_OVERRIDES_ARR[@]}"
