#!/bin/bash -l
#SBATCH -A cis260242-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=debug-python-env

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/tmp/python_env_debug}"
mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

LOG_PATH="${OUTPUT_DIR}/python_env_debug_${SLURM_JOB_ID:-manual}.log"
echo "Writing diagnostics to ${LOG_PATH}"

bash "${REPO_ROOT}/cluster/debug_python_env.sh" | tee "${LOG_PATH}"
