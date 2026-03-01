#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "REPO_ROOT not found: ${REPO_ROOT}" >&2
  exit 1
fi
cd "${REPO_ROOT}"

PARTITION="${PARTITION:-ai}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
# Optional. Leave empty unless you confirmed a valid feature name via sinfo/scontrol.
CONSTRAINT="${CONSTRAINT:-}"

SPLIT_JOBS="${SPLIT_JOBS:-1}" # 1=true -> submit CADD then MD4 with dependency
TIME_LIMIT="${TIME_LIMIT:-72:00:00}" # used when SPLIT_JOBS=0
TIME_LIMIT_CADD="${TIME_LIMIT_CADD:-24:00:00}"
TIME_LIMIT_MD4="${TIME_LIMIT_MD4:-24:00:00}"
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEMORY="${MEMORY:-240G}"
JOB_NAME="${JOB_NAME:-sticky_cadd_md4}"
JOB_NAME_CADD="${JOB_NAME_CADD:-${JOB_NAME}_cadd}"
JOB_NAME_MD4="${JOB_NAME_MD4:-${JOB_NAME}_md4}"
RUN_TAG="${RUN_TAG:-cadd_md4_baseline_$(date +%Y%m%d_%H%M%S)}"

CONDA_ENV="${CONDA_ENV:-sticky}"
ANVIL_MODULES="${ANVIL_MODULES:-}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

COMMON_SBATCH_ARGS=(
  --partition="${PARTITION}"
  --nodes="${NODES}"
  --ntasks="${NTASKS}"
  --gpus-per-node="${GPUS_PER_NODE}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEMORY}"
)

if [[ -n "${ACCOUNT}" ]]; then
  COMMON_SBATCH_ARGS+=(--account="${ACCOUNT}")
fi

if [[ -n "${QOS}" ]]; then
  COMMON_SBATCH_ARGS+=(--qos="${QOS}")
fi

if [[ -n "${CONSTRAINT}" ]]; then
  COMMON_SBATCH_ARGS+=(--constraint="${CONSTRAINT}")
fi

echo "Submitting sequential CADD->MD4 baseline job"
echo "  partition: ${PARTITION}"
echo "  account:   ${ACCOUNT:-<none>}"
echo "  qos:       ${QOS:-<none>}"
echo "  constraint:${CONSTRAINT:-<none>}"
echo "  gpus:      ${GPUS_PER_NODE}"
echo "  cpus:      ${CPUS_PER_TASK}"
echo "  mem:       ${MEMORY}"
if [[ "${SPLIT_JOBS}" == "1" || "${SPLIT_JOBS}" == "true" ]]; then
  echo "  mode:      split jobs (afterok dependency)"
  echo "  cadd time: ${TIME_LIMIT_CADD}"
  echo "  md4 time:  ${TIME_LIMIT_MD4}"
else
  echo "  mode:      single allocation"
  echo "  time:      ${TIME_LIMIT}"
fi
echo "  run tag:   ${RUN_TAG}"
echo

BASE_EXPORT="ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},ANVIL_MODULES=${ANVIL_MODULES},RUN_TAG=${RUN_TAG}"

if [[ "${SPLIT_JOBS}" == "1" || "${SPLIT_JOBS}" == "true" ]]; then
  CADD_SBATCH_ARGS=(
    "${COMMON_SBATCH_ARGS[@]}"
    --time="${TIME_LIMIT_CADD}"
    --job-name="${JOB_NAME_CADD}"
    --output="${LOG_DIR}/${JOB_NAME_CADD}_%j.out"
    --error="${LOG_DIR}/${JOB_NAME_CADD}_%j.err"
    --export="${BASE_EXPORT},MODEL_PHASE=cadd"
  )

  cadd_submit_out="$(
    sbatch --parsable "${CADD_SBATCH_ARGS[@]}" scripts/slurm/anvil/train_cadd_md4_sequential.slurm
  )"
  cadd_job_id="${cadd_submit_out%%;*}"

  MD4_SBATCH_ARGS=(
    "${COMMON_SBATCH_ARGS[@]}"
    --dependency="afterok:${cadd_job_id}"
    --time="${TIME_LIMIT_MD4}"
    --job-name="${JOB_NAME_MD4}"
    --output="${LOG_DIR}/${JOB_NAME_MD4}_%j.out"
    --error="${LOG_DIR}/${JOB_NAME_MD4}_%j.err"
    --export="${BASE_EXPORT},MODEL_PHASE=md4"
  )

  md4_submit_out="$(
    sbatch --parsable "${MD4_SBATCH_ARGS[@]}" scripts/slurm/anvil/train_cadd_md4_sequential.slurm
  )"
  md4_job_id="${md4_submit_out%%;*}"

  if [[ "${SBATCH_PARSABLE:-0}" == "1" ]]; then
    echo "${cadd_job_id}"
    echo "${md4_job_id}"
  else
    echo "Submitted CADD job: ${cadd_job_id}"
    echo "Submitted MD4 job:  ${md4_job_id} (afterok:${cadd_job_id})"
  fi
else
  SBATCH_ARGS=(
    "${COMMON_SBATCH_ARGS[@]}"
    --time="${TIME_LIMIT}"
    --job-name="${JOB_NAME}"
    --output="${LOG_DIR}/${JOB_NAME}_%j.out"
    --error="${LOG_DIR}/${JOB_NAME}_%j.err"
    --export="${BASE_EXPORT},MODEL_PHASE=both"
  )
  if [[ "${SBATCH_PARSABLE:-0}" == "1" ]]; then
    SBATCH_ARGS+=(--parsable)
  fi
  sbatch "${SBATCH_ARGS[@]}" scripts/slurm/anvil/train_cadd_md4_sequential.slurm
fi
