#!/bin/bash
#SBATCH --job-name=eval_cifar10_fid50k_nn
#SBATCH --account=ruqiz
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:h100:2
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%A_%a.out
#SBATCH --error=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%A_%a.err

set -euo pipefail

RUNS=(
  /scratch/gautschi/pjutrasd/sticky-diffusion/runs/cifar10/sjd_cifar10_eta1.0_mb2_nib_9654763_0
  /scratch/gautschi/pjutrasd/sticky-diffusion/runs/cifar10/sjd_cifar10_eta0.85_mb2_nib_9654763_1
)
RUN_DIR=${RUNS[${SLURM_ARRAY_TASK_ID}]}
RUN_TAG=$(basename "${RUN_DIR}")

PROJECT_ROOT=/home/pjutrasd/sticky-diffusion
SCRATCH_ROOT=/scratch/gautschi/pjutrasd/sticky-diffusion
TFDS_ROOT=${SCRATCH_ROOT}/data/tfds
FID_CACHE=${SCRATCH_ROOT}/data/fid_stats
HYDRA_OUT=${RUN_DIR}/offline_eval_fid50k
METRICS_OUT=${RUN_DIR}/metrics/offline_eval_fid50k.json

mkdir -p "${HYDRA_OUT}" "${RUN_DIR}/metrics"

module purge
module load conda
source activate /home/pjutrasd/.conda/envs/sticky

export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH:-}
export TFDS_DATA_DIR=${TFDS_ROOT}
export WANDB_DIR=${HYDRA_OUT}
export HYDRA_FULL_ERROR=1

cd "${PROJECT_ROOT}"

echo "=== Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) | run=${RUN_TAG} ==="
echo "RUN_DIR=${RUN_DIR}"
echo "METRICS_OUT=${METRICS_OUT}"
nvidia-smi || true

srun python -m sticky.cli.eval_checkpoint \
  experiment=cifar10/sjd_cifar10 \
  eval=cifar10 \
  offline_eval.run_dir="${RUN_DIR}" \
  offline_eval.checkpoint_source=best \
  offline_eval.use_run_config=true \
  offline_eval.use_ema=true \
  offline_eval.output_path="${METRICS_OUT}" \
  eval.fid_enabled=true \
  eval.fid_num_samples=50000 \
  eval.fid_cache_dir="${FID_CACHE}" \
  eval.fid_tfds_data_dir="${TFDS_ROOT}" \
  experiment.dataset.data_dir="${TFDS_ROOT}" \
  hydra.run.dir="${HYDRA_OUT}"
