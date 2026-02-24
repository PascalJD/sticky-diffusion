#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "REPO_ROOT not found: ${REPO_ROOT}" >&2
  exit 1
fi
cd "${REPO_ROOT}"

mkdir -p logs manifests

TUNING_TAG="${TUNING_TAG:-sjd_stage1_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_DIR="${MANIFEST_DIR:-manifests/${TUNING_TAG}}"
mkdir -p "${MANIFEST_DIR}"

STAGE1_MANIFEST="${STAGE1_MANIFEST:-${MANIFEST_DIR}/stage1_eta_p.txt}"

# Stage-1 sweep knobs.
STAGE1_ETAS="${STAGE1_ETAS:-0.9,0.85,0.8,0.75}"
STAGE1_P_VALUES="${STAGE1_P_VALUES:-0.5,1,2,3}"
STAGE1_TEMPERATURE="${STAGE1_TEMPERATURE:-1.0}"
STAGE1_ANCHOR_MODE="${STAGE1_ANCHOR_MODE:-fixed}"
STAGE1_SEEDS="${STAGE1_SEEDS:-0}"

# Slurm knobs.
PARTITION="${PARTITION:-gpu}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
MEMORY="${MEMORY:-0}"
ARRAY_MAX_PARALLEL="${ARRAY_MAX_PARALLEL:-16}"
CONDA_ENV="${CONDA_ENV:-sticky}"

# Optional module list to load inside Slurm jobs.
ANVIL_MODULES="${ANVIL_MODULES:-}"

JOB_NAME="${JOB_NAME:-sjd_stage1}"

echo "TUNING_TAG=${TUNING_TAG}"
echo "Manifest dir: ${MANIFEST_DIR}"
echo "Generating Stage-1 manifest (eta x p)..."
python scripts/slurm/anvil/generate_sjd_manifest.py \
  --etas "${STAGE1_ETAS}" \
  --p-values "${STAGE1_P_VALUES}" \
  --temperature "${STAGE1_TEMPERATURE}" \
  --anchor-mode "${STAGE1_ANCHOR_MODE}" \
  --seeds "${STAGE1_SEEDS}" \
  --output "${STAGE1_MANIFEST}"

RUN_COUNT="$(awk 'NF && $1 !~ /^#/' "${STAGE1_MANIFEST}" | wc -l | tr -d ' ')"
echo "Stage-1 run count: ${RUN_COUNT}"

JOB_ID="$(
  SBATCH_PARSABLE=1 \
  PARTITION="${PARTITION}" \
  ACCOUNT="${ACCOUNT}" \
  QOS="${QOS}" \
  TIME_LIMIT="${TIME_LIMIT}" \
  NODES="${NODES}" \
  GPUS_PER_NODE="${GPUS_PER_NODE}" \
  CPUS_PER_TASK="${CPUS_PER_TASK}" \
  MEMORY="${MEMORY}" \
  ARRAY_MAX_PARALLEL="${ARRAY_MAX_PARALLEL}" \
  CONDA_ENV="${CONDA_ENV}" \
  ANVIL_MODULES="${ANVIL_MODULES}" \
  REPO_ROOT="${REPO_ROOT}" \
  TUNING_TAG="${TUNING_TAG}" \
  bash scripts/slurm/anvil/submit_sjd_tuning.sh "${STAGE1_MANIFEST}" \
    --job-name="${JOB_NAME}"
)"

echo "Submitted Stage-1 array job: ${JOB_ID}"
echo "Manifest: ${STAGE1_MANIFEST}"
echo "Outputs root: outputs/tuning/${TUNING_TAG}"
