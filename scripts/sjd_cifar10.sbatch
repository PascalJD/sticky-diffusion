#!/bin/bash
#SBATCH --job-name=sjd_cifar10
#SBATCH --account=ruqiz
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:h100:2
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/home/pjutrasd/sticky-diffusion
SCRATCH_ROOT=/scratch/gautschi/pjutrasd/sticky-diffusion
TFDS_ROOT=${SCRATCH_ROOT}/data/tfds
FID_CACHE=${SCRATCH_ROOT}/data/fid_stats
RUN_DIR=${SCRATCH_ROOT}/runs/cifar10/sjd_cifar10_${SLURM_JOB_ID}

mkdir -p "${RUN_DIR}" "${TFDS_ROOT}" "${FID_CACHE}"

module purge
module load conda
source activate /home/pjutrasd/.conda/envs/sticky

export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH:-}
export TFDS_DATA_DIR=${TFDS_ROOT}
export WANDB_DIR=${RUN_DIR}
export HYDRA_FULL_ERROR=1

cd "${PROJECT_ROOT}"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) ==="
echo "RUN_DIR=${RUN_DIR}"
nvidia-smi || true

srun python -m sticky.cli.train \
  experiment=cifar10/sjd_cifar10 \
  wandb.enabled=true \
  hydra.run.dir="${RUN_DIR}" \
  experiment.dataset.data_dir="${TFDS_ROOT}" \
  eval.fid_cache_dir="${FID_CACHE}" \
  eval.fid_tfds_data_dir="${TFDS_ROOT}"
