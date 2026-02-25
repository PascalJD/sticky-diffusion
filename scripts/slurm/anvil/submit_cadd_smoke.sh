#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "REPO_ROOT not found: ${REPO_ROOT}" >&2
  exit 1
fi
cd "${REPO_ROOT}"

if [[ -n "${SCRATCH:-}" ]]; then
  SCRATCH_ROOT="${SCRATCH}"
elif [[ -d "/anvil/scratch/${USER}" ]]; then
  SCRATCH_ROOT="/anvil/scratch/${USER}"
else
  SCRATCH_ROOT="${HOME}/scratch"
fi

PARTITION="${PARTITION:-gpu}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-00:40:00}"
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-120G}"
JOB_NAME="${JOB_NAME:-cadd_smoke}"
CONDA_ENV="${CONDA_ENV:-sticky}"
ANVIL_MODULES="${ANVIL_MODULES:-}"

RUN_TAG="${RUN_TAG:-cadd_smoke_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/sticky-diffusion/outputs/smoke}"
DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/sticky-diffusion/data/cifar10}"
LOG_DIR="${LOG_DIR:-${SCRATCH_ROOT}/sticky-diffusion/logs}"

if [[ -z "${PLATFORM:-}" ]]; then
  if (( GPUS_PER_NODE > 1 )); then
    PLATFORM="pmap"
  else
    PLATFORM="single"
  fi
else
  PLATFORM="${PLATFORM}"
fi

if [[ -z "${BATCH_SIZE:-}" ]]; then
  if [[ "${PLATFORM}" == "pmap" ]]; then
    BATCH_SIZE="512"
  else
    BATCH_SIZE="128"
  fi
fi
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"

SMOKE_STEPS="${SMOKE_STEPS:-200}"
LOG_EVERY="${LOG_EVERY:-20}"
METRICS_EVERY="${METRICS_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-false}"
LOG_IMAGES_EVERY="${LOG_IMAGES_EVERY:-0}"
EVAL_ENABLED="${EVAL_ENABLED:-false}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${DATA_DIR}"

if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  export EXTRA_OVERRIDES
fi
if [[ -n "${BASE_OVERRIDES:-}" ]]; then
  export BASE_OVERRIDES
fi

SBATCH_ARGS=(
  --partition="${PARTITION}"
  --time="${TIME_LIMIT}"
  --nodes="${NODES}"
  --ntasks="${NTASKS}"
  --gpus-per-node="${GPUS_PER_NODE}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEMORY}"
  --job-name="${JOB_NAME}"
  --output="${LOG_DIR}/${JOB_NAME}_%j.out"
  --error="${LOG_DIR}/${JOB_NAME}_%j.err"
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},ANVIL_MODULES=${ANVIL_MODULES},RUN_TAG=${RUN_TAG},OUTPUT_ROOT=${OUTPUT_ROOT},DATA_DIR=${DATA_DIR},PLATFORM=${PLATFORM},BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},SMOKE_STEPS=${SMOKE_STEPS},LOG_EVERY=${LOG_EVERY},METRICS_EVERY=${METRICS_EVERY},CHECKPOINT_EVERY=${CHECKPOINT_EVERY},SAVE_FINAL_CHECKPOINT=${SAVE_FINAL_CHECKPOINT},LOG_IMAGES_EVERY=${LOG_IMAGES_EVERY},EVAL_ENABLED=${EVAL_ENABLED},WANDB_ENABLED=${WANDB_ENABLED}"
)

if [[ -n "${ACCOUNT}" ]]; then
  SBATCH_ARGS+=(--account="${ACCOUNT}")
fi
if [[ -n "${QOS}" ]]; then
  SBATCH_ARGS+=(--qos="${QOS}")
fi

if [[ "${SBATCH_PARSABLE:-0}" == "1" ]]; then
  SBATCH_ARGS+=(--parsable)
fi

echo "Submitting CADD smoke job"
echo "  partition: ${PARTITION}"
echo "  account:   ${ACCOUNT:-<none>}"
echo "  qos:       ${QOS:-<none>}"
echo "  gpus:      ${GPUS_PER_NODE}"
echo "  platform:  ${PLATFORM}"
echo "  run tag:   ${RUN_TAG}"
echo "  run dir:   ${OUTPUT_ROOT}/${RUN_TAG}"
echo "  data dir:  ${DATA_DIR}"
echo

sbatch "${SBATCH_ARGS[@]}" scripts/slurm/anvil/train_cadd.slurm
