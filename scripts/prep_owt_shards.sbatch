#!/bin/bash
#SBATCH --job-name=prep_owt_shard
#SBATCH --account=ruqiz
#SBATCH --partition=cpu
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=03:30:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%A_%a.out
#SBATCH --error=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%A_%a.err

# OWT pretokenize shard. 5-task array:
#   0..3 = train shards (~2M docs each, total 7,913,769 docs)
#   4    = val (last 100k docs)
# Each task streams int32 tokens to a per-shard .bin via the patched
# tools/prepare_openwebtext.py --out-bin mode. Memory bounded to one
# tokenize batch (~6 MB), so OOM is impossible regardless of total size.

set -euo pipefail

PROJECT_ROOT=/home/pjutrasd/sticky-diffusion
SCRATCH_ROOT=/scratch/gautschi/pjutrasd/sticky-diffusion
SHARDS_DIR=${SCRATCH_ROOT}/data/openwebtext/shards
HF_CACHE=${SCRATCH_ROOT}/hf_cache

mkdir -p "${SHARDS_DIR}" "${SCRATCH_ROOT}/slurm_logs"

# Symlink ${PROJECT_ROOT}/data -> ${SCRATCH_ROOT}/data so dataset config relative paths resolve.
if [[ ! -e "${PROJECT_ROOT}/data" ]]; then
  ln -s "${SCRATCH_ROOT}/data" "${PROJECT_ROOT}/data"
fi

module purge
module load conda
source activate /home/pjutrasd/.conda/envs/sticky

export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH:-}
export HF_HOME=${HF_CACHE}
export HF_DATASETS_CACHE=${HF_CACHE}/datasets
export TRANSFORMERS_CACHE=${HF_CACHE}/transformers
export RAYON_NUM_THREADS=32
export OMP_NUM_THREADS=32
export TOKENIZERS_PARALLELISM=true

cd "${PROJECT_ROOT}"

# Per-task shard layout. Counts derived from OWT total of 8,013,769 docs and
# 100k val (MDLM convention). Train shards split the first 7,913,769 docs
# into four ~equal pieces.
case "${SLURM_ARRAY_TASK_ID}" in
  0) START=0;        STOP=1978442; OUT="${SHARDS_DIR}/train_0.bin" ;;
  1) START=1978442;  STOP=3956884; OUT="${SHARDS_DIR}/train_1.bin" ;;
  2) START=3956884;  STOP=5935326; OUT="${SHARDS_DIR}/train_2.bin" ;;
  3) START=5935326;  STOP=7913769; OUT="${SHARDS_DIR}/train_3.bin" ;;
  4) START=7913769;  STOP=8013769; OUT="${SHARDS_DIR}/val.bin" ;;
  *) echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

echo "=== Shard ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) | $(date) ==="
echo "  range: docs [${START}, ${STOP})"
echo "  out:   ${OUT}"

srun python tools/prepare_openwebtext.py \
  --start "${START}" \
  --stop "${STOP}" \
  --out-bin "${OUT}"

echo "=== Shard done | $(date) ==="
ls -lh "${OUT}"
