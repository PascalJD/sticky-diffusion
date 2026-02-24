#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <manifest.txt> [extra sbatch args ...]" >&2
  exit 1
fi

MANIFEST="$1"
shift || true

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

RUN_COUNT=$(awk 'NF && $1 !~ /^#/' "${MANIFEST}" | wc -l | tr -d ' ')
if [[ "${RUN_COUNT}" -le 0 ]]; then
  echo "No runnable entries in manifest: ${MANIFEST}" >&2
  exit 1
fi

ARRAY_MAX_PARALLEL="${ARRAY_MAX_PARALLEL:-16}"
ARRAY_SPEC="0-$((RUN_COUNT - 1))%${ARRAY_MAX_PARALLEL}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

PARTITION="${PARTITION:-gpu}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
MEMORY="${MEMORY:-0}"
CONDA_ENV="${CONDA_ENV:-sticky}"
REPO_ROOT="${REPO_ROOT:-$PWD}"
SBATCH_PARSABLE="${SBATCH_PARSABLE:-0}"

SBATCH_ARGS=(
  --partition="${PARTITION}"
  --time="${TIME_LIMIT}"
  --nodes="${NODES}"
  --gpus-per-node="${GPUS_PER_NODE}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEMORY}"
  --array="${ARRAY_SPEC}"
  --export="ALL,MANIFEST=${MANIFEST},REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV}"
)

if [[ -n "${ACCOUNT}" ]]; then
  SBATCH_ARGS+=(--account="${ACCOUNT}")
fi
if [[ -n "${QOS}" ]]; then
  SBATCH_ARGS+=(--qos="${QOS}")
fi

if [[ $# -gt 0 ]]; then
  SBATCH_ARGS+=("$@")
fi

if [[ "${SBATCH_PARSABLE}" == "1" ]]; then
  SBATCH_ARGS+=(--parsable)
  sbatch "${SBATCH_ARGS[@]}" scripts/slurm/anvil/train_sjd_array.slurm
else
  echo "Submitting ${RUN_COUNT} runs as array ${ARRAY_SPEC}"
  echo "Partition=${PARTITION} Account=${ACCOUNT:-<none>} QOS=${QOS:-<none>}"
  sbatch "${SBATCH_ARGS[@]}" scripts/slurm/anvil/train_sjd_array.slurm
fi
