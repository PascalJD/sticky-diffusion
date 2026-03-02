#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_EXTRA_OVERRIDES="forward/beta@experiment.forward.beta=vp_linear forward/hazard@experiment.forward.hazard=poly_alpha forward/jump@experiment.forward.jump=vp_matched experiment.forward.jump.eta=0.8 experiment.training.num_log_images=4"
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  EXTRA_OVERRIDES="${DEFAULT_EXTRA_OVERRIDES} ${EXTRA_OVERRIDES}"
else
  EXTRA_OVERRIDES="${DEFAULT_EXTRA_OVERRIDES}"
fi

export MODEL="${MODEL:-sjd}"
export EXPERIMENT_CFG="${EXPERIMENT_CFG:-sjd_cifar10}"
export EVAL_CFG="${EVAL_CFG:-sjd_cifar10}"

export JOB_NAME="${JOB_NAME:-sticky_sjd}"
export PARTITION="${PARTITION:-ai}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-48}"
export MEMORY="${MEMORY:-480G}"
export TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

export PLATFORM="${PLATFORM:-pmap}"
export REQUIRED_LOCAL_DEVICES="${REQUIRED_LOCAL_DEVICES:-${GPUS_PER_NODE}}"

# Mirror the CADD paper-run training cadence.
export BATCH_SIZE="${BATCH_SIZE:-512}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
export TRAIN_STEPS="${TRAIN_STEPS:-500000}"
export LOG_EVERY="${LOG_EVERY:-1000}"
export METRICS_EVERY="${METRICS_EVERY:-1000}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
export LOG_IMAGES_EVERY="${LOG_IMAGES_EVERY:-10000}"
export SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-true}"
export MODEL_TIMESTEPS="${MODEL_TIMESTEPS:-}"
export SAMPLE_TIMESTEPS="${SAMPLE_TIMESTEPS:-128}"

export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export B2="${B2:-0.99}"

export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export EVAL_ENABLED="${EVAL_ENABLED:-false}"
export BASELINE_ARCH_114M="${BASELINE_ARCH_114M:-true}"
export DISABLE_AUGMENT="${DISABLE_AUGMENT:-true}"
export EXTRA_OVERRIDES

exec "${SCRIPT_DIR}/submit_train.sh"
