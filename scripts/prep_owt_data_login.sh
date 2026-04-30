#!/bin/bash
# Login-node runner for SJD/F-SJD OWT data prep. Mirrors prep_owt_data.sbatch
# but runs directly on the Gautschi front-end (192 cores / 768 GB), which is
# vastly faster than waiting 5 days in the `cpu` partition standby queue.
#
# Suggested launch (detached screen, see end of file for one-liner):
#   screen -S owt_prep -dm bash scripts/prep_owt_data_login.sh
#
# Steps (idempotent — re-runs skip finished work):
#   1. extract GPT-2 wte                 -> data/anchors/gpt2_wte.npz
#   2. tokenize OpenWebText (GPT-2 BPE)  -> data/openwebtext/{train,val}.npy
#   3. rename to config-expected names   -> openwebtext_gpt2_1024_{train,eval}.npy
#   4. compute frequency log_w           -> data/anchors/openwebtext_log_w.npz

set -euo pipefail

PROJECT_ROOT=/home/pjutrasd/sticky-diffusion
SCRATCH_ROOT=/scratch/gautschi/pjutrasd/sticky-diffusion
DATA_ROOT=${SCRATCH_ROOT}/data
OWT_DIR=${DATA_ROOT}/openwebtext
ANCHORS_DIR=${DATA_ROOT}/anchors
HF_CACHE=${SCRATCH_ROOT}/hf_cache
LOG_DIR=${SCRATCH_ROOT}/slurm_logs
LOG_FILE=${LOG_DIR}/prep_owt_data_login.log

mkdir -p "${OWT_DIR}" "${ANCHORS_DIR}" "${HF_CACHE}" "${LOG_DIR}"

# Symlink ${PROJECT_ROOT}/data -> ${SCRATCH_ROOT}/data so dataset config relative paths resolve.
if [[ ! -e "${PROJECT_ROOT}/data" ]]; then
  ln -s "${DATA_ROOT}" "${PROJECT_ROOT}/data"
  echo "Created symlink ${PROJECT_ROOT}/data -> ${DATA_ROOT}"
elif [[ -L "${PROJECT_ROOT}/data" ]]; then
  echo "Existing data symlink: $(readlink ${PROJECT_ROOT}/data)"
else
  echo "WARNING: ${PROJECT_ROOT}/data exists and is not a symlink — leaving alone"
fi

module purge
module load conda
source activate /home/pjutrasd/.conda/envs/sticky

export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH:-}
export HF_HOME=${HF_CACHE}
export HF_DATASETS_CACHE=${HF_CACHE}/datasets
export TRANSFORMERS_CACHE=${HF_CACHE}/transformers
# Cap parallelism on the shared login node — be a good citizen.
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export RAYON_NUM_THREADS=32
export TOKENIZERS_PARALLELISM=true

cd "${PROJECT_ROOT}"

echo "=== OWT data prep (login-node) | $(date) | host=$(hostname) ==="
echo "DATA_ROOT=${DATA_ROOT}"
echo "HF_HOME=${HF_HOME}"
echo "Threads cap: 32"

WTE_OUT=${ANCHORS_DIR}/gpt2_wte.npz
TRAIN_RAW=${OWT_DIR}/train.npy
VAL_RAW=${OWT_DIR}/val.npy
TRAIN_FINAL=${OWT_DIR}/openwebtext_gpt2_1024_train.npy
EVAL_FINAL=${OWT_DIR}/openwebtext_gpt2_1024_eval.npy
LOG_W_OUT=${ANCHORS_DIR}/openwebtext_log_w.npz

# --- Step 1: extract GPT-2 wte ---
if [[ -f "${WTE_OUT}" ]]; then
  echo "[1/4] GPT-2 wte already at ${WTE_OUT} — skipping"
else
  echo "[1/4] Extracting GPT-2 wte -> ${WTE_OUT}"
  python tools/extract_gpt2_embeddings.py --out "${WTE_OUT}"
fi

# --- Step 2: pretokenize OpenWebText ---
if [[ -f "${TRAIN_FINAL}" && -f "${EVAL_FINAL}" ]]; then
  echo "[2/4] Pretokenized OWT already at ${TRAIN_FINAL}, ${EVAL_FINAL} — skipping"
elif [[ -f "${TRAIN_RAW}" && -f "${VAL_RAW}" ]]; then
  echo "[2/4] Raw tokenized files already at ${TRAIN_RAW}, ${VAL_RAW} — skipping tokenize"
else
  echo "[2/4] Tokenizing OpenWebText -> ${OWT_DIR}/{train,val}.npy"
  python tools/prepare_openwebtext.py --out-dir "${OWT_DIR}" --seq-len 1024
fi

# --- Step 3: rename train.npy -> openwebtext_gpt2_1024_train.npy, val.npy -> ..._eval.npy ---
if [[ -f "${TRAIN_RAW}" && ! -f "${TRAIN_FINAL}" ]]; then
  echo "[3/4] Renaming ${TRAIN_RAW} -> ${TRAIN_FINAL}"
  mv "${TRAIN_RAW}" "${TRAIN_FINAL}"
fi
if [[ -f "${VAL_RAW}" && ! -f "${EVAL_FINAL}" ]]; then
  echo "[3/4] Renaming ${VAL_RAW} -> ${EVAL_FINAL}"
  mv "${VAL_RAW}" "${EVAL_FINAL}"
fi
if [[ ! -f "${TRAIN_FINAL}" || ! -f "${EVAL_FINAL}" ]]; then
  echo "ERROR: expected ${TRAIN_FINAL} and ${EVAL_FINAL} after step 3" >&2
  exit 1
fi
echo "[3/4] OWT splits in place: ${TRAIN_FINAL}, ${EVAL_FINAL}"

# --- Step 4: compute frequency log_w ---
if [[ -f "${LOG_W_OUT}" ]]; then
  echo "[4/4] log_w already at ${LOG_W_OUT} — skipping"
else
  echo "[4/4] Computing log_w -> ${LOG_W_OUT}"
  python tools/compute_anchor_frequencies.py \
    --tokens "${TRAIN_FINAL}" \
    --vocab-size 50257 \
    --out "${LOG_W_OUT}"
fi

echo ""
echo "=== Prep complete | $(date) ==="
ls -lh "${WTE_OUT}" "${TRAIN_FINAL}" "${EVAL_FINAL}" "${LOG_W_OUT}"
