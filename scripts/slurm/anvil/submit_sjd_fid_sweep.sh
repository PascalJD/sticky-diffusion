#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "REPO_ROOT not found: ${REPO_ROOT}" >&2
  exit 1
fi
cd "${REPO_ROOT}"

PARTITION="${PARTITION:-ai}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
CONSTRAINT="${CONSTRAINT:-}"
EXCLUDE="${EXCLUDE:-}"
NODELIST="${NODELIST:-}"

TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-24}"
MEMORY="${MEMORY:-240G}"

CONDA_ENV="${CONDA_ENV:-sticky}"
ANVIL_MODULES="${ANVIL_MODULES:-modtree/gpu cuda/12.0.1 anaconda/2024.02-py311}"
EXPERIMENT_CFG="${EXPERIMENT_CFG:-sjd_cifar10}"
EVAL_CFG="${EVAL_CFG:-sjd_cifar10}"
PLATFORM="${PLATFORM:-single}"
SEED="${SEED:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"
CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-periodic}"
USE_EMA="${USE_EMA:-true}"

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_DIR="$(dirname "${CHECKPOINT_PATH}")"
  if [[ -z "${CHECKPOINT_STEP}" ]]; then
    base_name="$(basename "${CHECKPOINT_PATH}")"
    if [[ "${base_name}" =~ ^checkpoint_([0-9]+) ]]; then
      CHECKPOINT_STEP="${BASH_REMATCH[1]}"
    fi
  fi
fi

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "Set CHECKPOINT_PATH (preferred) or CHECKPOINT_DIR." >&2
  exit 1
fi
if [[ -z "${CHECKPOINT_STEP}" ]]; then
  CHECKPOINT_STEP="null"
fi

if [[ -n "${SCRATCH:-}" ]]; then
  SCRATCH_ROOT="${SCRATCH}"
elif [[ -d "/anvil/scratch/${USER}" ]]; then
  SCRATCH_ROOT="/anvil/scratch/${USER}"
else
  SCRATCH_ROOT="${HOME}/scratch"
fi

DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/sticky-diffusion/data/cifar10}"
FID_CACHE_DIR="${FID_CACHE_DIR:-${SCRATCH_ROOT}/sticky-diffusion/data/fid_stats}"

FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-10000}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-128}"
IS_ENABLED="${IS_ENABLED:-false}"
FORCE_IS="${FORCE_IS:-false}"

SAMPLER_STEPS="${SAMPLER_STEPS:-512}"
SAMPLER_PROBE_BATCHES="${SAMPLER_PROBE_BATCHES:-32}"
SAMPLER_PROBE_BATCH_SIZE="${SAMPLER_PROBE_BATCH_SIZE:-${FID_BATCH_SIZE}}"
SAMPLER_PROBE_SEED_OFFSET="${SAMPLER_PROBE_SEED_OFFSET:-777777}"

ETA_VALUES="${ETA_VALUES:-0.6 0.8 1.0}"
TAU_VALUES="${TAU_VALUES:-0.8 1.0 1.2}"

ckpt_parent="$(dirname "${CHECKPOINT_DIR}")"
if [[ "$(basename "${CHECKPOINT_DIR}")" == "checkpoints" ]]; then
  RUN_ROOT="${ckpt_parent}"
else
  RUN_ROOT="${CHECKPOINT_DIR}"
fi

SWEEP_TAG="${SWEEP_TAG:-sjd_fid_eta_tau_probe${FID_NUM_SAMPLES}_$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-${RUN_ROOT}/eval_sweeps/${SWEEP_TAG}}"
LOG_DIR="${LOG_DIR:-${SWEEP_ROOT}/logs}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-sjd_fid}"

mkdir -p "${SWEEP_ROOT}" "${LOG_DIR}" "${DATA_DIR}" "${FID_CACHE_DIR}"

manifest="${SWEEP_ROOT}/submit_manifest.tsv"
echo -e "job_id\teta\ttau\trun_dir\tmetrics_json\tsummary_json" > "${manifest}"

echo "Submitting SJD checkpoint FID sweep"
echo "  repo: ${REPO_ROOT}"
echo "  partition: ${PARTITION}"
echo "  account: ${ACCOUNT:-<none>}"
echo "  checkpoint_dir: ${CHECKPOINT_DIR}"
echo "  checkpoint_step: ${CHECKPOINT_STEP}"
echo "  sweep_root: ${SWEEP_ROOT}"
echo "  eta values: ${ETA_VALUES}"
echo "  tau values: ${TAU_VALUES}"
echo "  fid_num_samples: ${FID_NUM_SAMPLES}"
echo

for eta in ${ETA_VALUES}; do
  for tau in ${TAU_VALUES}; do
    eta_tag="${eta//./p}"
    tau_tag="${tau//./p}"
    run_key="eta_${eta_tag}__tau_${tau_tag}"
    run_dir="${SWEEP_ROOT}/${run_key}"
    job_name="${JOB_NAME_PREFIX}_e${eta_tag}_t${tau_tag}"
    metrics_json="${run_dir}/offline_eval_metrics.json"
    summary_json="${run_dir}/summary.json"

    export_vars=(
      "REPO_ROOT=${REPO_ROOT}"
      "CONDA_ENV=${CONDA_ENV}"
      "ANVIL_MODULES=${ANVIL_MODULES}"
      "EXPERIMENT_CFG=${EXPERIMENT_CFG}"
      "EVAL_CFG=${EVAL_CFG}"
      "PLATFORM=${PLATFORM}"
      "SEED=${SEED}"
      "WANDB_ENABLED=${WANDB_ENABLED}"
      "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
      "CHECKPOINT_SOURCE=${CHECKPOINT_SOURCE}"
      "CHECKPOINT_STEP=${CHECKPOINT_STEP}"
      "USE_EMA=${USE_EMA}"
      "DATA_DIR=${DATA_DIR}"
      "FID_CACHE_DIR=${FID_CACHE_DIR}"
      "ETA=${eta}"
      "TAU=${tau}"
      "SAMPLER_STEPS=${SAMPLER_STEPS}"
      "FID_NUM_SAMPLES=${FID_NUM_SAMPLES}"
      "FID_BATCH_SIZE=${FID_BATCH_SIZE}"
      "IS_ENABLED=${IS_ENABLED}"
      "FORCE_IS=${FORCE_IS}"
      "SAMPLER_PROBE_BATCHES=${SAMPLER_PROBE_BATCHES}"
      "SAMPLER_PROBE_BATCH_SIZE=${SAMPLER_PROBE_BATCH_SIZE}"
      "SAMPLER_PROBE_SEED_OFFSET=${SAMPLER_PROBE_SEED_OFFSET}"
      "RUN_DIR=${run_dir}"
    )

    sbatch_args=(
      --partition="${PARTITION}"
      --time="${TIME_LIMIT}"
      --nodes="${NODES}"
      --ntasks="${NTASKS}"
      --gpus-per-node="${GPUS_PER_NODE}"
      --cpus-per-task="${CPUS_PER_TASK}"
      --mem="${MEMORY}"
      --job-name="${job_name}"
      --output="${LOG_DIR}/${job_name}_%j.out"
      --error="${LOG_DIR}/${job_name}_%j.err"
      --export="ALL,$(IFS=,; echo "${export_vars[*]}")"
      --parsable
    )
    if [[ -n "${ACCOUNT}" ]]; then
      sbatch_args+=(--account="${ACCOUNT}")
    fi
    if [[ -n "${QOS}" ]]; then
      sbatch_args+=(--qos="${QOS}")
    fi
    if [[ -n "${CONSTRAINT}" ]]; then
      sbatch_args+=(--constraint="${CONSTRAINT}")
    fi
    if [[ -n "${EXCLUDE}" ]]; then
      sbatch_args+=(--exclude="${EXCLUDE}")
    fi
    if [[ -n "${NODELIST}" ]]; then
      sbatch_args+=(--nodelist="${NODELIST}")
    fi

    sbatch_out="$(sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/eval_checkpoint.slurm")"
    job_id="$(printf '%s\n' "${sbatch_out}" | grep -Eo '[0-9]+' | tail -n 1 || true)"
    echo "${sbatch_out}"
    echo -e "${job_id}\t${eta}\t${tau}\t${run_dir}\t${metrics_json}\t${summary_json}" >> "${manifest}"
    echo "submitted job ${job_id} for eta=${eta}, tau=${tau}"
    if [[ -n "${job_id}" ]] && command -v scontrol >/dev/null 2>&1; then
      job_info="$(scontrol show job "${job_id}" 2>/dev/null | tr '\n' ' ' || true)"
      if [[ -n "${job_info}" ]]; then
        effective_partition="$(printf '%s\n' "${job_info}" | grep -Eo 'Partition=[^ ]+' | head -n 1 | cut -d= -f2-)"
        effective_qos="$(printf '%s\n' "${job_info}" | grep -Eo 'QOS=[^ ]+' | head -n 1 | cut -d= -f2-)"
        effective_time_limit="$(printf '%s\n' "${job_info}" | grep -Eo 'TimeLimit=[^ ]+' | head -n 1 | cut -d= -f2-)"
        echo "  effective partition: ${effective_partition:-<unknown>}"
        echo "  effective qos:       ${effective_qos:-<unknown>}"
        echo "  effective timelimit: ${effective_time_limit:-<unknown>}"
      fi
    fi
  done
done

echo
echo "Sweep submissions complete."
echo "Manifest: ${manifest}"
