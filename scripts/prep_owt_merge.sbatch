#!/bin/bash
#SBATCH --job-name=prep_owt_merge
#SBATCH --account=ruqiz
#SBATCH --partition=cpu
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%j.err

# Merge step. Runs only after the shard array's afterok dependency clears
# (see scripts/launch_owt_prep.sh). Streams shard .bin files into the two
# (N, 1024) int32 .npy files the dataloader expects, then computes the
# F-SJD frequency log_w table from the merged train file.

set -euo pipefail

PROJECT_ROOT=/home/pjutrasd/sticky-diffusion
SCRATCH_ROOT=/scratch/gautschi/pjutrasd/sticky-diffusion
DATA_DIR=${SCRATCH_ROOT}/data/openwebtext
SHARDS_DIR=${DATA_DIR}/shards
ANCHORS_DIR=${SCRATCH_ROOT}/data/anchors

mkdir -p "${ANCHORS_DIR}" "${SCRATCH_ROOT}/slurm_logs"

module purge
module load conda
source activate /home/pjutrasd/.conda/envs/sticky

export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH:-}

cd "${PROJECT_ROOT}"

echo "=== Merge ${SLURM_JOB_ID} on $(hostname) | $(date) ==="

TRAIN_NPY=${DATA_DIR}/openwebtext_gpt2_1024_train.npy
EVAL_NPY=${DATA_DIR}/openwebtext_gpt2_1024_eval.npy
LOG_W_OUT=${ANCHORS_DIR}/openwebtext_log_w.npz

# 1. Merge bins into the (N, 1024) int32 .npy files.
python tools/merge_owt_shards.py \
  --train-bins "${SHARDS_DIR}/train_0.bin" \
               "${SHARDS_DIR}/train_1.bin" \
               "${SHARDS_DIR}/train_2.bin" \
               "${SHARDS_DIR}/train_3.bin" \
  --val-bins   "${SHARDS_DIR}/val.bin" \
  --out-dir "${DATA_DIR}" \
  --seq-len 1024

# 2. Compute frequency log_w from the merged train tokens (F-SJD weights).
if [[ -f "${LOG_W_OUT}" ]]; then
  echo "log_w already at ${LOG_W_OUT} — skipping"
else
  python tools/compute_anchor_frequencies.py \
    --tokens "${TRAIN_NPY}" \
    --vocab-size 50257 \
    --out "${LOG_W_OUT}"
fi

echo ""
echo "=== Merge complete | $(date) ==="
ls -lh \
  "${ANCHORS_DIR}/gpt2_wte.npz" \
  "${TRAIN_NPY}" \
  "${EVAL_NPY}" \
  "${LOG_W_OUT}"
