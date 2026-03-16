#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CALLER_PWD="${PWD}"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "REPO_ROOT not found: ${REPO_ROOT}" >&2
  exit 1
fi
cd "${REPO_ROOT}"

MODEL="${MODEL:-}"
if [[ -z "${MODEL}" ]]; then
  echo "MODEL is required (for example: cadd, md4, sjd)." >&2
  exit 1
fi

PARTITION="${PARTITION:-ai}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
CONSTRAINT="${CONSTRAINT:-}"
EXCLUDE="${EXCLUDE:-}"
if [[ -z "${EXCLUDE}" && -n "${BAD_NODES:-}" ]]; then
  EXCLUDE="${BAD_NODES}"
fi
NODELIST="${NODELIST:-}"

TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-48}"
MEMORY="${MEMORY:-480G}"
JOB_NAME="${JOB_NAME:-sticky_${MODEL}}"

CONDA_ENV="${CONDA_ENV:-sticky}"
ANVIL_MODULES="${ANVIL_MODULES:-}"
EXTRA_OVERRIDES_FILE="${EXTRA_OVERRIDES_FILE:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${EXTRA_OVERRIDES_FILE}" ]]; then
  if [[ "${EXTRA_OVERRIDES_FILE}" != /* ]]; then
    EXTRA_OVERRIDES_FILE="${CALLER_PWD}/${EXTRA_OVERRIDES_FILE}"
  fi
  if [[ ! -f "${EXTRA_OVERRIDES_FILE}" ]]; then
    echo "EXTRA_OVERRIDES_FILE not found: ${EXTRA_OVERRIDES_FILE}" >&2
    exit 1
  fi
fi

if [[ -n "${SCRATCH:-}" ]]; then
  SCRATCH_ROOT="${SCRATCH}"
elif [[ -d "/anvil/scratch/${USER}" ]]; then
  SCRATCH_ROOT="/anvil/scratch/${USER}"
else
  SCRATCH_ROOT="${HOME}/scratch"
fi

RUN_TAG="${RUN_TAG:-${MODEL}_$(date +%Y%m%d_%H%M%S)}"
STUDY_ROOT="${STUDY_ROOT:-}"
if [[ -z "${OUTPUT_ROOT:-}" && -n "${STUDY_ROOT}" ]]; then
  OUTPUT_ROOT="${STUDY_ROOT}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/sticky-diffusion/outputs}"
DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/sticky-diffusion/data/cifar10}"
if [[ -z "${LOG_DIR:-}" && -n "${STUDY_ROOT}" ]]; then
  LOG_DIR="${STUDY_ROOT}/slurm_logs"
fi
LOG_DIR="${LOG_DIR:-${SCRATCH_ROOT}/sticky-diffusion/logs}"

if [[ -z "${PLATFORM:-}" ]]; then
  if (( GPUS_PER_NODE > 1 )); then
    PLATFORM="pmap"
  else
    PLATFORM="single"
  fi
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
  --export="ALL,REPO_ROOT=${REPO_ROOT},MODEL=${MODEL},CONDA_ENV=${CONDA_ENV},ANVIL_MODULES=${ANVIL_MODULES},RUN_TAG=${RUN_TAG},OUTPUT_ROOT=${OUTPUT_ROOT},DATA_DIR=${DATA_DIR},PLATFORM=${PLATFORM},REQUIRED_LOCAL_DEVICES=${REQUIRED_LOCAL_DEVICES:-${GPUS_PER_NODE}}"
)

if [[ -n "${ACCOUNT}" ]]; then
  SBATCH_ARGS+=(--account="${ACCOUNT}")
fi
if [[ -n "${QOS}" ]]; then
  SBATCH_ARGS+=(--qos="${QOS}")
fi
if [[ -n "${CONSTRAINT}" ]]; then
  SBATCH_ARGS+=(--constraint="${CONSTRAINT}")
fi
if [[ -n "${EXCLUDE}" ]]; then
  SBATCH_ARGS+=(--exclude="${EXCLUDE}")
fi
if [[ -n "${NODELIST}" ]]; then
  SBATCH_ARGS+=(--nodelist="${NODELIST}")
fi
if [[ "${SBATCH_PARSABLE:-0}" == "1" ]]; then
  SBATCH_ARGS+=(--parsable)
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${DATA_DIR}"

echo "Submitting ${MODEL} training job"
echo "  partition: ${PARTITION}"
echo "  account:   ${ACCOUNT:-<none>}"
echo "  qos:       ${QOS:-<none>}"
echo "  constraint:${CONSTRAINT:-<none>}"
echo "  exclude:   ${EXCLUDE:-<none>}"
echo "  nodelist:  ${NODELIST:-<none>}"
echo "  gpus:      ${GPUS_PER_NODE}"
echo "  cpus:      ${CPUS_PER_TASK}"
echo "  mem:       ${MEMORY}"
echo "  time:      ${TIME_LIMIT}"
echo "  platform:  ${PLATFORM}"
echo "  run tag:   ${RUN_TAG}"
echo "  run dir:   ${OUTPUT_ROOT}/${RUN_TAG}"
if [[ -n "${EXTRA_OVERRIDES_FILE}" ]]; then
  echo "  overrides: ${EXTRA_OVERRIDES_FILE}"
fi
echo

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN sbatch'
  for arg in "${SBATCH_ARGS[@]}"; do
    printf ' %q' "${arg}"
  done
  printf ' %q\n' "${SCRIPT_DIR}/train_model.slurm"
  exit 0
fi

sbatch_out="$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/train_model.slurm")"
sbatch_rc=$?
echo "${sbatch_out}"
if (( sbatch_rc != 0 )); then
  exit "${sbatch_rc}"
fi

job_id="$(printf '%s\n' "${sbatch_out}" | grep -Eo '[0-9]+' | tail -n 1 || true)"
if [[ -n "${job_id}" ]] && command -v scontrol >/dev/null 2>&1; then
  job_info="$(scontrol show job "${job_id}" 2>/dev/null | tr '\n' ' ' || true)"
  if [[ -n "${job_info}" ]]; then
    effective_partition="$(printf '%s\n' "${job_info}" | grep -Eo 'Partition=[^ ]+' | head -n 1 | cut -d= -f2-)"
    effective_qos="$(printf '%s\n' "${job_info}" | grep -Eo 'QOS=[^ ]+' | head -n 1 | cut -d= -f2-)"
    effective_time_limit="$(printf '%s\n' "${job_info}" | grep -Eo 'TimeLimit=[^ ]+' | head -n 1 | cut -d= -f2-)"
    effective_time_min="$(printf '%s\n' "${job_info}" | grep -Eo 'TimeMin=[^ ]+' | head -n 1 | cut -d= -f2-)"
    echo "Effective Slurm assignment:"
    echo "  job id:    ${job_id}"
    echo "  partition: ${effective_partition:-<unknown>}"
    echo "  qos:       ${effective_qos:-<unknown>}"
    echo "  timelimit: ${effective_time_limit:-<unknown>}"
    echo "  timemin:   ${effective_time_min:-<unknown>}"
  fi
fi
