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
CONSTRAINT="${CONSTRAINT:-h100}"

TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEMORY="${MEMORY:-240G}"
JOB_NAME="${JOB_NAME:-sticky_cadd_md4}"

CONDA_ENV="${CONDA_ENV:-sticky}"
ANVIL_MODULES="${ANVIL_MODULES:-}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

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
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},ANVIL_MODULES=${ANVIL_MODULES}"
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

if [[ "${SBATCH_PARSABLE:-0}" == "1" ]]; then
  SBATCH_ARGS+=(--parsable)
fi

echo "Submitting sequential CADD->MD4 baseline job"
echo "  partition: ${PARTITION}"
echo "  account:   ${ACCOUNT:-<none>}"
echo "  qos:       ${QOS:-<none>}"
echo "  constraint:${CONSTRAINT:-<none>}"
echo "  gpus:      ${GPUS_PER_NODE}"
echo "  cpus:      ${CPUS_PER_TASK}"
echo "  mem:       ${MEMORY}"
echo "  time:      ${TIME_LIMIT}"
echo

sbatch "${SBATCH_ARGS[@]}" scripts/slurm/anvil/train_cadd_md4_sequential.slurm
