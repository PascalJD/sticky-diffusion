#!/bin/bash
#SBATCH --job-name=sjd_owt_fsjd
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
RUN_TAG=sjd_owt_fsjd_${SLURM_JOB_ID}
RUN_DIR=${SCRATCH_ROOT}/runs/openwebtext/${RUN_TAG}

mkdir -p "${RUN_DIR}"

module purge
module load conda
source activate /home/pjutrasd/.conda/envs/sticky

export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH:-}
export WANDB_DIR=${RUN_DIR}
export HYDRA_FULL_ERROR=1

cd "${PROJECT_ROOT}"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) | F-SJD OWT (frequency-weighted hazard) ==="
echo "RUN_DIR=${RUN_DIR}"
nvidia-smi || true

# 4x H100, per-device BS 64 -> global BS 256, 100k steps, eta=1.0, frozen GPT-2 anchors.
# eta=1.0 removes jump-kernel state-dependency, so the only state-dep mechanism
# active is the frequency-weighted forward hazard: lambda_t(a) = beta(t) * w(a)
# with w(a) = -log p_0(a) / H(p_0)  (entropy-normalized so E_{p_0}[w] = 1, which
# preserves the expected per-token corruption rate vs the eta=1 baseline).
# Hazard-weighting group is mounted at package experiment.forward.hazard_weighting,
# so the override needs the @-package syntax (a bare `forward/hazard_weighting=...`
# at root is rejected). eval.enabled=false skips the OOMing text_basic sampler.
srun python -m sticky.cli.train \
  experiment=openwebtext/sjd_openwebtext \
  forward/hazard_weighting@experiment.forward.hazard_weighting=frequency \
  experiment.forward.hazard_weighting.log_w_path=/scratch/gautschi/pjutrasd/sticky-diffusion/data/anchors/openwebtext_log_w_entropy.npz \
  experiment.training.num_train_steps=100000 \
  experiment.dataset.batch_size=64 \
  experiment.forward.jump.eta=1.0 \
  experiment.model.anchor.learnable=false \
  eval.enabled=false \
  wandb.enabled=true \
  wandb.run_name="${RUN_TAG}" \
  "+wandb.tags=[openwebtext,sjd,eta1.0,fsjd,entropy_norm,100k]" \
  hydra.run.dir="${RUN_DIR}"
