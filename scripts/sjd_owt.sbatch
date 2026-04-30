#!/bin/bash
#SBATCH --job-name=sjd_owt
#SBATCH --account=ruqiz
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:h100:4
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/home/pjutrasd/sticky-diffusion
SCRATCH_ROOT=/scratch/gautschi/pjutrasd/sticky-diffusion
RUN_TAG=sjd_owt_${SLURM_JOB_ID}
RUN_DIR=${SCRATCH_ROOT}/runs/openwebtext/${RUN_TAG}

mkdir -p "${RUN_DIR}"

module purge
module load conda
source activate /home/pjutrasd/.conda/envs/sticky

export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH:-}
export WANDB_DIR=${RUN_DIR}
export HYDRA_FULL_ERROR=1

cd "${PROJECT_ROOT}"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) | SJD OWT (anchor-agnostic baseline) ==="
echo "RUN_DIR=${RUN_DIR}"
nvidia-smi || true

# 4x H100, per-device BS 64 -> global BS 256, 100k steps, eta=1.0, frozen GPT-2 anchors.
# eta=1.0 removes jump-kernel state-dependency (CADD limit), so this run has no
# state-dependency at all — pure time-only baseline.
# Hazard weighting defaults to anchor-agnostic baseline (forward/hazard_weighting=none).
# eval.enabled=false skips the text_basic sample-generation pipeline, which
# OOMs at JIT-compile (sampler allocates a (NFE x batch x seq_len x vocab)
# tensor ~ 2.5 TB for OWT). Likelihood eval (controlled separately via
# cfg.training.likelihood_eval_every_steps) is unaffected.
srun python -m sticky.cli.train \
  experiment=openwebtext/sjd_openwebtext \
  experiment.training.num_train_steps=100000 \
  experiment.dataset.batch_size=64 \
  experiment.forward.jump.eta=1.0 \
  experiment.model.anchor.learnable=false \
  eval.enabled=false \
  wandb.enabled=true \
  wandb.run_name="${RUN_TAG}" \
  "+wandb.tags=[openwebtext,sjd,eta1.0,no_fsjd,100k]" \
  hydra.run.dir="${RUN_DIR}"
