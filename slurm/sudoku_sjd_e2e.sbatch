#!/bin/bash -l
#SBATCH -A cis260242-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=sjd_sudoku_e2e
#SBATCH --output=/home/x-pjutrasdube/scratch/sticky-diffusion/runs/%x_%j/slurm-%j.out
#SBATCH --error=/home/x-pjutrasdube/scratch/sticky-diffusion/runs/%x_%j/slurm-%j.err

# Parameterized launcher for the SJD Sudoku e2e + matched baseline runs.
# Usage:
#   sbatch slurm/sudoku_sjd_e2e.sbatch sudoku/sjd_sudoku_e2e
#   sbatch --job-name=sjd_sudoku_e2e_baseline \
#          slurm/sudoku_sjd_e2e.sbatch sudoku/sjd_sudoku_e2e_baseline
#
# The default --job-name above targets the e2e run; override --job-name when
# launching the baseline so the run directory and wandb run name follow.

set -euo pipefail

EXPERIMENT="${1:-sudoku/sjd_sudoku_e2e}"
REPO_ROOT="/home/x-pjutrasdube/projects/sticky-diffusion"
ENV_PREFIX="/anvil/scratch/x-pjutrasdube/envs/sticky"
RUN_DIR="/home/x-pjutrasdube/scratch/sticky-diffusion/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

mkdir -p "${RUN_DIR}"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) ==="
date -Is
nvidia-smi
echo "experiment=${EXPERIMENT}"
echo "run_dir=${RUN_DIR}"
echo "repo_root=${REPO_ROOT}"
echo "env_prefix=${ENV_PREFIX}"
echo "repo HEAD: $(cd "${REPO_ROOT}" && git rev-parse HEAD)"

unset PYTHONHOME || true
unset PYTHONPATH || true
export PYTHONPATH="${REPO_ROOT}/src"
export HYDRA_FULL_ERROR=1
# SCRATCH is already set by Anvil profile; the Sudoku dataset loader resolves
# data/sudoku -> ${SCRATCH}/sticky-diffusion/data/sudoku automatically.

cd "${REPO_ROOT}"
"${ENV_PREFIX}/bin/python" -m sticky.cli.train \
    experiment="${EXPERIMENT}" \
    hydra.run.dir="${RUN_DIR}" \
    wandb.run_name="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

echo "=== done ==="
date -Is
