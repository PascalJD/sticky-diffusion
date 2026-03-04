#!/bin/bash
#SBATCH --job-name=sticky_eval_ckpt
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "REPO_ROOT not found: ${REPO_ROOT}" >&2
  exit 1
fi
cd "${REPO_ROOT}"
if [[ ! -f "${REPO_ROOT}/src/sticky/__init__.py" ]]; then
  echo "REPO_ROOT does not appear to be sticky-diffusion root: ${REPO_ROOT}" >&2
  exit 1
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset PYTHONHOME

ensure_module_cmd() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
  command -v module >/dev/null 2>&1
}

ensure_conda_cmd() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  if ensure_module_cmd; then
    CONDA_MODULE_CANDIDATES="${CONDA_MODULE_CANDIDATES:-anaconda/2023.09 anaconda/2024.02 anaconda/2024.02-py311 anaconda}"
    for mod in ${CONDA_MODULE_CANDIDATES}; do
      if module load "${mod}" >/dev/null 2>&1; then
        if command -v conda >/dev/null 2>&1; then
          echo "Auto-loaded conda module: ${mod}"
          return 0
        fi
      fi
    done
  fi

  if [[ -f "${HOME}/.bashrc" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc" >/dev/null 2>&1 || true
  fi
  command -v conda >/dev/null 2>&1
}

if [[ -n "${ANVIL_MODULES:-}" ]]; then
  if ensure_module_cmd; then
    module purge || true
    for mod in ${ANVIL_MODULES}; do
      module load "${mod}"
    done
  else
    echo "ANVIL_MODULES was set but module command is unavailable." >&2
  fi
fi

CONDA_ENV="${CONDA_ENV:-sticky}"
if ! ensure_conda_cmd; then
  echo "conda command not found. Set ANVIL_MODULES (for example: anaconda/2024.02-py311)." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

activate_conda_env() {
  local env_spec="$1"
  if conda activate "${env_spec}" >/dev/null 2>&1; then
    echo "Activated conda env: ${env_spec}"
    return 0
  fi

  if [[ "${env_spec}" != /* ]]; then
    local candidates=()
    if [[ -n "${SCRATCH:-}" ]]; then
      candidates+=("${SCRATCH}/envs/${env_spec}")
    fi
    candidates+=("/anvil/scratch/${USER}/envs/${env_spec}")

    local env_path
    for env_path in "${candidates[@]}"; do
      if [[ -d "${env_path}" ]] && conda activate "${env_path}" >/dev/null 2>&1; then
        echo "Activated conda env via path fallback: ${env_path}"
        return 0
      fi
    done
  fi

  return 1
}

if ! activate_conda_env "${CONDA_ENV}"; then
  echo "Failed to activate CONDA_ENV=${CONDA_ENV}. If this is a path-based env, pass its full path." >&2
  exit 1
fi

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

EXPERIMENT_CFG="${EXPERIMENT_CFG:-sjd_cifar10}"
EVAL_CFG="${EVAL_CFG:-sjd_cifar10}"
PLATFORM="${PLATFORM:-single}"
SEED="${SEED:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "CHECKPOINT_DIR is required (the periodic checkpoint root)." >&2
  exit 1
fi
CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-periodic}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-null}"
USE_EMA="${USE_EMA:-true}"

ETA="${ETA:-0.8}"
TAU="${TAU:-1.0}"
SAMPLER_STEPS="${SAMPLER_STEPS:-512}"

FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-10000}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-128}"
IS_ENABLED="${IS_ENABLED:-false}"
IS_NUM_SAMPLES="${IS_NUM_SAMPLES:-${FID_NUM_SAMPLES}}"
IS_BATCH_SIZE="${IS_BATCH_SIZE:-${FID_BATCH_SIZE}}"
FORCE_IS="${FORCE_IS:-false}"

SAMPLER_PROBE_BATCHES="${SAMPLER_PROBE_BATCHES:-32}"
SAMPLER_PROBE_BATCH_SIZE="${SAMPLER_PROBE_BATCH_SIZE:-${FID_BATCH_SIZE}}"
SAMPLER_PROBE_SEED_OFFSET="${SAMPLER_PROBE_SEED_OFFSET:-777777}"

if [[ -n "${SCRATCH:-}" ]]; then
  SCRATCH_ROOT="${SCRATCH}"
elif [[ -d "/anvil/scratch/${USER}" ]]; then
  SCRATCH_ROOT="/anvil/scratch/${USER}"
else
  SCRATCH_ROOT="${HOME}/scratch"
fi

DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/sticky-diffusion/data/cifar10}"
FID_CACHE_DIR="${FID_CACHE_DIR:-${SCRATCH_ROOT}/sticky-diffusion/data/fid_stats}"
RUN_DIR="${RUN_DIR:-${SCRATCH_ROOT}/sticky-diffusion/outputs/offline_eval/sjd_eta_tau}"
METRICS_BASENAME="${METRICS_BASENAME:-offline_eval_metrics.json}"
METRICS_PATH="${RUN_DIR}/${METRICS_BASENAME}"

mkdir -p "${RUN_DIR}" "${FID_CACHE_DIR}" "${DATA_DIR}"

echo "[$(date)] Host: $(hostname)"
echo "REPO_ROOT: ${REPO_ROOT}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Checkpoint source: ${CHECKPOINT_SOURCE}"
echo "Checkpoint step: ${CHECKPOINT_STEP}"
echo "Run dir: ${RUN_DIR}"
echo "Metrics path: ${METRICS_PATH}"
echo "eta: ${ETA}"
echo "tau/logit_temperature: ${TAU}"
echo "FID samples: ${FID_NUM_SAMPLES}"
echo "FID batch size: ${FID_BATCH_SIZE}"
echo "Sampler probe batches: ${SAMPLER_PROBE_BATCHES}"
echo

python -m sticky.entrypoints.eval_checkpoint \
  experiment="${EXPERIMENT_CFG}" \
  eval="${EVAL_CFG}" \
  seed="${SEED}" \
  experiment.training.seed="${SEED}" \
  experiment.dataset.data_dir="${DATA_DIR}" \
  experiment.runtime.platform="${PLATFORM}" \
  experiment.sampler.n_steps="${SAMPLER_STEPS}" \
  experiment.sampler.logit_temperature="${TAU}" \
  forward/beta@experiment.forward.beta=vp_linear \
  forward/hazard@experiment.forward.hazard=poly_alpha \
  forward/jump@experiment.forward.jump=vp_matched \
  experiment.forward.jump.eta="${ETA}" \
  eval.enabled=true \
  eval.run_at_end=true \
  eval.fid_enabled=true \
  eval.fid_every=0 \
  eval.fid_num_samples="${FID_NUM_SAMPLES}" \
  eval.fid_batch_size="${FID_BATCH_SIZE}" \
  eval.fid_dataset_name=cifar10 \
  eval.fid_split=train \
  eval.fid_cache_dir="${FID_CACHE_DIR}" \
  eval.is_enabled="${IS_ENABLED}" \
  eval.is_every=0 \
  eval.is_num_samples="${IS_NUM_SAMPLES}" \
  eval.is_batch_size="${IS_BATCH_SIZE}" \
  wandb.enabled="${WANDB_ENABLED}" \
  offline_eval.checkpoint_dir="${CHECKPOINT_DIR}" \
  offline_eval.checkpoint_source="${CHECKPOINT_SOURCE}" \
  offline_eval.checkpoint_step="${CHECKPOINT_STEP}" \
  offline_eval.use_ema="${USE_EMA}" \
  offline_eval.force_fid=true \
  offline_eval.force_is="${FORCE_IS}" \
  offline_eval.sampler_probe_batches="${SAMPLER_PROBE_BATCHES}" \
  offline_eval.sampler_probe_batch_size="${SAMPLER_PROBE_BATCH_SIZE}" \
  offline_eval.sampler_probe_seed_offset="${SAMPLER_PROBE_SEED_OFFSET}" \
  offline_eval.output_path="${METRICS_PATH}" \
  hydra.run.dir="${RUN_DIR}"

SUMMARY_PATH="${RUN_DIR}/summary.json"
python - "${METRICS_PATH}" "${SUMMARY_PATH}" "${ETA}" "${TAU}" <<'PY'
import json
import pathlib
import sys

metrics_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
eta = float(sys.argv[3])
tau = float(sys.argv[4])

payload = json.loads(metrics_path.read_text(encoding="utf-8"))
eval_metrics = payload.get("metrics", {})
probe_metrics = payload.get("sampling_probe", {}).get("metrics", {})

summary = {
    "eta": eta,
    "tau": tau,
    "fid": eval_metrics.get("eval/fid"),
    "is": eval_metrics.get("eval/is"),
    "is_std": eval_metrics.get("eval/is_std"),
    "sampling/lam_mean_active": probe_metrics.get("sampling/lam_mean_active"),
    "sampling/p_jump_mean_active": probe_metrics.get("sampling/p_jump_mean_active"),
    "sampling/jump_frac_total": probe_metrics.get("sampling/jump_frac_total"),
    "sampling/frac_committed_pre_force": probe_metrics.get("sampling/frac_committed_pre_force"),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[offline-eval] Wrote summary to {summary_path}", flush=True)
PY
