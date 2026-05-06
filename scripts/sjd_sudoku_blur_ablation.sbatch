#!/bin/bash
#SBATCH --job-name=sjd_sudoku_blur_ablation
#SBATCH --account=ruqiz
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:h100:2
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/home/pjutrasd/sticky-diffusion
SCRATCH_ROOT=/scratch/gautschi/pjutrasd/sticky-diffusion
TFDS_ROOT=${SCRATCH_ROOT}/data/tfds
RUN_TAG=sjd_sudoku_blur_ablation_${SLURM_JOB_ID}
RUN_DIR=${SCRATCH_ROOT}/runs/sudoku/${RUN_TAG}

mkdir -p "${RUN_DIR}" "${TFDS_ROOT}"

module purge
module load conda
source activate /home/pjutrasd/.conda/envs/sticky

export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH:-}
export TFDS_DATA_DIR=${TFDS_ROOT}
export WANDB_DIR=${RUN_DIR}
export HYDRA_FULL_ERROR=1

cd "${PROJECT_ROOT}"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) | blur ablation: W_I,full,row_only,row_col ==="
echo "RUN_DIR=${RUN_DIR}"
nvidia-smi || true

srun python -m sticky.cli.train --multirun \
  experiment=sudoku/sjd_sudoku_blur \
  "+blur_ablation=W_I,full,row_only,row_col" \
  "hydra.sweep.dir=${RUN_DIR}" \
  "hydra.sweep.subdir=\${blur_ablation}" \
  wandb.enabled=true \
  wandb.group=sjd_sudoku_blur_ablation
