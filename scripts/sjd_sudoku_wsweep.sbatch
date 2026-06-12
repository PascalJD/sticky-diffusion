#!/bin/bash
# SLURM array launcher — sjd_sudoku_wsweep (16 arms).
#
# LOOKUP TABLE ORDER:
#   Indices 0–14: rho-major, seed-minor grid
#     rho  0.00: seeds 0,1,2  → indices  0, 1, 2
#     rho  0.10: seeds 0,1,2  → indices  3, 4, 5
#     rho  0.20: seeds 0,1,2  → indices  6, 7, 8
#     rho  0.35: seeds 0,1,2  → indices  9,10,11
#     rho  0.50: seeds 0,1,2  → indices 12,13,14
#   Index 15: rho0-learn arm (rho=0.0, seed=0, anchor.learnable=true)
#
# DRY-RUN PREVIEW (all 16 arms):
#   for i in $(seq 0 15); do DRY_RUN=1 SLURM_ARRAY_TASK_ID=$i bash scripts/sjd_sudoku_wsweep.sbatch; done
#
# SUBMIT SINGLE ARM (e.g. calibration-first: index 0):
#   sbatch --array=0 scripts/sjd_sudoku_wsweep.sbatch
#
# SUBMIT FULL SWEEP:
#   sbatch --array=0-15 scripts/sjd_sudoku_wsweep.sbatch
#
# VENUE / RESOURCE OVERRIDES (use sbatch CLI flags — they take precedence over
#   in-file #SBATCH directives; env vars like SLURM_ACCOUNT do NOT affect them):
#   sbatch -A <account> -p <partition> --gres=gpu:h100:2 --array=0-15 scripts/sjd_sudoku_wsweep.sbatch
#
#   Single-GPU variant (--gres=gpu:h100:1, ~2x per-run wallclock):
#   sbatch -A <account> -p <partition> --gres=gpu:h100:1 --array=0-15 scripts/sjd_sudoku_wsweep.sbatch
#   16 single-GPU jobs often clear the queue faster than 16 dual-GPU jobs.
#
#   NOTE: --output / --error paths below are Gautschi-specific
#   (/scratch/gautschi/…/slurm_logs/). On other venues you MUST override them
#   with -o/-e CLI flags, or create the equivalent directory first.
#
# ---------------------------------------------------------------------------
# PRE-SUBMISSION CHECKLIST
#   1. Stage data (see DATA STAGING section below).
#   2. Create the SLURM log directory — SLURM will NOT create it automatically;
#      every task will fail silently with no log if the directory is missing:
#        mkdir -p <scratch>/slurm_logs
#      On Gautschi the default path is:
#        mkdir -p /scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs
#   3. Run a calibration arm first (index 0) to confirm steps/sec and plateau.
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# CALIBRATION-FIRST PROTOCOL
#   Launch index 0 (rho=0.0, seed=0) alone:
#     sbatch --array=0 scripts/sjd_sudoku_wsweep.sbatch
#   Monitor plateau via:
#     metrics/sudoku_sjd_progress.csv  →  eval/linear_topk_probability/solve_rate
#   Before fanning out, choose:
#     - Step budget:  training.max_steps=<N>   (NOTE: root-level, NOT
#                     experiment.training.max_steps — the experiment.-prefixed
#                     form will error; set via EXTRA_OVERRIDES below)
#     - Eval cost:    eval.sudoku_num_batches=<M>
#                     experiment.training.eval_every_steps=<K>
#   Check that the calibration run's steps/sec implies the full step budget
#   fits within --time; there is NO resume path on TIMEOUT.
#   Then submit the remaining arms:
#     sbatch --array=1-15 scripts/sjd_sudoku_wsweep.sbatch
#
# ---------------------------------------------------------------------------
# NO-REQUEUE WARNING
#   Arms are NOT resumable. A requeued or restarted arm reuses hydra.run.dir,
#   restarts from step 0, and crashes at the first periodic checkpoint
#   (overwrite=False). Wipe or rename RUN_DIR before resubmitting an arm.
#
# DATA STAGING (run once before any submission):
#   python tools/prepare_sudoku_data.py
#   Concurrent pulls are unreliable (best-effort advisory lock on Lustre);
#   stage once first.
#   Verify staged files (resolved path on Gautschi when $SCRATCH is set;
#   see src/sticky/data/sudoku/paths.py — resolve_data_root() uses
#   $SCRATCH/sticky-diffusion/data/sudoku/ when $SCRATCH is set):
#     sha256sum \
#       /scratch/gautschi/pjutrasd/sticky-diffusion/data/sudoku/Sudoku-test-data.npy \
#       /scratch/gautschi/pjutrasd/sticky-diffusion/data/sudoku/Sudoku-train-data.npy
#   Expected sha256:
#     13800f8c9675dfc840afa5029e1c5b1ce831d9b4c16ccd39a4acbaa1b7872c58  Sudoku-test-data.npy
#     82eaba71daa9d446cc125f587531b6d94c82ef9f2b13fdd16002f27086f84000  Sudoku-train-data.npy
#   (On clusters without $SCRATCH the loader falls back to repo-relative
#    data/sudoku/ resolved against the original cwd at import time.)
#
# PRECISION NOTE
#   The rho=0 arm's W=I einsum is bitwise-identical to no-blur on CPU and at
#   highest JAX float precision. On GPU with TF32 matmuls it is only
#   near-identical (all sweep arms share the same einsum path, so in-sweep
#   comparisons are unaffected). Set JAX_DEFAULT_MATMUL_PRECISION=float32
#   only if a bitwise rho0-vs-legacy-baseline comparison is ever needed.
# ---------------------------------------------------------------------------

#SBATCH --job-name=sjd_sudoku_wsweep
#SBATCH --account=ruqiz
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:h100:2
# Check calibration steps/sec: full step budget MUST fit within this limit;
# there is no resume path on TIMEOUT.
#SBATCH --time=6:00:00
#SBATCH --array=0-15
#SBATCH --no-requeue
#SBATCH --output=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%A_%a.out
#SBATCH --error=/scratch/gautschi/pjutrasd/sticky-diffusion/slurm_logs/%x-%A_%a.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Path / env knobs
# ---------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/home/pjutrasd/sticky-diffusion}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/gautschi/pjutrasd/sticky-diffusion}"
RUNS_BASE_DIR="${RUNS_BASE_DIR:-${SCRATCH_ROOT}/runs/sudoku}"
TFDS_ROOT="${TFDS_ROOT:-${SCRATCH_ROOT}/data/tfds}"
CONDA_ENV="${CONDA_ENV:-/home/pjutrasd/.conda/envs/sticky}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
# Extra Hydra overrides appended verbatim, e.g.:
#   EXTRA_OVERRIDES="training.max_steps=50000 eval.sudoku_num_batches=4"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# ---------------------------------------------------------------------------
# Lookup table: array index → (rho, seed, learnable, run_name)
# Order: rho-major, seed-minor (indices 0-14); index 15 = rho0-learn arm.
# ---------------------------------------------------------------------------
RHO_TABLE=( 0.0 0.0 0.0  0.1 0.1 0.1  0.2 0.2 0.2  0.35 0.35 0.35  0.5 0.5 0.5  0.0 )
SEED_TABLE=(  0   1   2    0   1   2    0   1   2     0    1    2    0   1   2    0  )
LEARN_TABLE=(false false false false false false false false false false false false false false false true)
NAME_TABLE=(
  "sjd_sudoku_wsweep_rho0.0_seed0"
  "sjd_sudoku_wsweep_rho0.0_seed1"
  "sjd_sudoku_wsweep_rho0.0_seed2"
  "sjd_sudoku_wsweep_rho0.1_seed0"
  "sjd_sudoku_wsweep_rho0.1_seed1"
  "sjd_sudoku_wsweep_rho0.1_seed2"
  "sjd_sudoku_wsweep_rho0.2_seed0"
  "sjd_sudoku_wsweep_rho0.2_seed1"
  "sjd_sudoku_wsweep_rho0.2_seed2"
  "sjd_sudoku_wsweep_rho0.35_seed0"
  "sjd_sudoku_wsweep_rho0.35_seed1"
  "sjd_sudoku_wsweep_rho0.35_seed2"
  "sjd_sudoku_wsweep_rho0.5_seed0"
  "sjd_sudoku_wsweep_rho0.5_seed1"
  "sjd_sudoku_wsweep_rho0.5_seed2"
  "sjd_sudoku_wsweep_rho0.0_seed0_learn"
)

# ---------------------------------------------------------------------------
# Resolve this task's parameters
# ---------------------------------------------------------------------------

# Assert all four lookup tables have equal lengths.
_N_RHO=${#RHO_TABLE[@]}
_N_SEED=${#SEED_TABLE[@]}
_N_LEARN=${#LEARN_TABLE[@]}
_N_NAME=${#NAME_TABLE[@]}
if [ "${_N_RHO}" -ne "${_N_SEED}" ] || [ "${_N_RHO}" -ne "${_N_LEARN}" ] || [ "${_N_RHO}" -ne "${_N_NAME}" ]; then
  echo "ERROR: lookup table length mismatch: RHO=${_N_RHO} SEED=${_N_SEED} LEARN=${_N_LEARN} NAME=${_N_NAME}" >&2
  exit 1
fi

# Guard: refuse to run a real arm without SLURM_ARRAY_TASK_ID.
# DRY_RUN=1 previews are allowed to default to index 0.
if [ -z "${SLURM_ARRAY_TASK_ID+x}" ]; then
  if [ "${DRY_RUN:-0}" != "1" ]; then
    echo "ERROR: refusing to run a real arm without SLURM_ARRAY_TASK_ID;" \
         "set DRY_RUN=1 for previews." >&2
    exit 1
  fi
  IDX=0
else
  IDX="${SLURM_ARRAY_TASK_ID}"
fi

# Bounds check: 0 <= IDX < table length.
if [ "${IDX}" -lt 0 ] || [ "${IDX}" -ge "${_N_RHO}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${IDX} is out of bounds [0, $((${_N_RHO} - 1))]." >&2
  exit 1
fi

RHO="${RHO_TABLE[$IDX]}"
SEED="${SEED_TABLE[$IDX]}"
LEARNABLE="${LEARN_TABLE[$IDX]}"
RUN_NAME="${NAME_TABLE[$IDX]}"
RUN_DIR="${RUNS_BASE_DIR}/${RUN_NAME}"

# Build the per-arm override list
OVERRIDES=(
  "experiment=sudoku/sjd_sudoku_wsweep"
  "experiment.forward.blur.rho=${RHO}"
  "experiment.training.seed=${SEED}"
  "wandb.run_name=${RUN_NAME}"
  "hydra.run.dir=${RUN_DIR}"
  "wandb.enabled=${WANDB_ENABLED}"
)

if [ "${LEARNABLE}" = "true" ]; then
  OVERRIDES+=( "experiment.model.anchor.learnable=true" )
fi

# Append any caller-supplied extra overrides (glob-safe, no word-splitting).
if [ -n "${EXTRA_OVERRIDES}" ]; then
  read -r -a EXTRA_ARR <<< "${EXTRA_OVERRIDES}"
  OVERRIDES+=( "${EXTRA_ARR[@]}" )
fi

# ---------------------------------------------------------------------------
# DRY_RUN mode: print fully-resolved command and exit
# ---------------------------------------------------------------------------
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "=== DRY_RUN idx=${IDX} rho=${RHO} seed=${SEED} learnable=${LEARNABLE} run_name=${RUN_NAME} ==="
  echo "    run_dir=${RUN_DIR}"
  echo "    cmd: python -m sticky.cli.train ${OVERRIDES[*]}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Cluster environment setup (mirrors sjd_sudoku_blur_ablation.sbatch pattern)
# ---------------------------------------------------------------------------
mkdir -p "${RUN_DIR}" "${TFDS_ROOT}"

module purge
module load conda
source activate "${CONDA_ENV}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TFDS_DATA_DIR="${TFDS_ROOT}"
export WANDB_DIR="${RUN_DIR}"
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Resolved-config echo (printed to SLURM .out log)
# ---------------------------------------------------------------------------
echo "=== Job ${SLURM_JOB_ID:-local} array_task=${IDX} on $(hostname) ==="
echo "    rho=${RHO}  seed=${SEED}  learnable=${LEARNABLE}"
echo "    run_name=${RUN_NAME}"
echo "    run_dir=${RUN_DIR}"
echo "    wandb_enabled=${WANDB_ENABLED}"
echo "    overrides: ${OVERRIDES[*]}"
nvidia-smi || true

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
python -m sticky.cli.train "${OVERRIDES[@]}"
