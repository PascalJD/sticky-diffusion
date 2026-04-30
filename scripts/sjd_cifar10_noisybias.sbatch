#!/bin/bash
#SBATCH --job-name=sjd_cifar10_nb
#SBATCH --account=ruqiz
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:h100:2
#SBATCH --time=3-00:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%A_%a.out
#SBATCH --error=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%A_%a.err

set -euo pipefail

ETAS=(0.85 1.0)
ETA=${ETAS[${SLURM_ARRAY_TASK_ID}]}

PROJECT_ROOT=/home/pjutrasd/sticky-diffusion
SCRATCH_ROOT=/scratch/gautschi/pjutrasd/sticky-diffusion
TFDS_ROOT=${SCRATCH_ROOT}/data/tfds
FID_CACHE=${SCRATCH_ROOT}/data/fid_stats
RUN_TAG=sjd_cifar10_nb_eta${ETA}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
RUN_DIR=${SCRATCH_ROOT}/runs/cifar10/${RUN_TAG}

mkdir -p "${RUN_DIR}" "${TFDS_ROOT}" "${FID_CACHE}"

module purge
module load conda
source activate /home/pjutrasd/.conda/envs/sticky

export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH:-}
export TFDS_DATA_DIR=${TFDS_ROOT}
export WANDB_DIR=${RUN_DIR}
export HYDRA_FULL_ERROR=1

cd "${PROJECT_ROOT}"

echo "=== Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) | eta=${ETA} ==="
echo "RUN_DIR=${RUN_DIR}"
nvidia-smi || true

srun python -m sticky.cli.train \
  experiment=cifar10/sjd_cifar10 \
  experiment.forward.jump.eta=${ETA} \
  +experiment.model.use_noisy_input_bias=true \
  experiment.training.num_microbatches=1 \
  training.max_steps=100000 \
  wandb.enabled=true \
  wandb.run_name="sjd_cifar10_nb_eta${ETA}" \
  "+wandb.tags=[cifar10,sjd,noisy_input_bias,eta${ETA}]" \
  hydra.run.dir="${RUN_DIR}" \
  experiment.dataset.data_dir="${TFDS_ROOT}" \
  eval.fid_cache_dir="${FID_CACHE}" \
  eval.fid_tfds_data_dir="${TFDS_ROOT}"
