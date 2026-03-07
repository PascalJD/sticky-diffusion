#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-sjd}"
export EXPERIMENT_CFG="${EXPERIMENT_CFG:-sjd_cifar10}"
export EVAL_CFG="${EVAL_CFG:-sjd_cifar10}"

export JOB_NAME="${JOB_NAME:-sticky_sjd}"
export PARTITION="${PARTITION:-ai}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-48}"
export MEMORY="${MEMORY:-480G}"
export TIME_LIMIT="${TIME_LIMIT:-48:00:00}"

export PLATFORM="${PLATFORM:-pmap}"
export REQUIRED_LOCAL_DEVICES="${REQUIRED_LOCAL_DEVICES:-${GPUS_PER_NODE}}"

# Use the experiment config defaults unless explicitly overridden at submit time.
export BATCH_SIZE="${BATCH_SIZE:-}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
export TRAIN_STEPS="${TRAIN_STEPS:-}"
export LOG_EVERY="${LOG_EVERY:-}"
export METRICS_EVERY="${METRICS_EVERY:-}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-}"
export LOG_IMAGES_EVERY="${LOG_IMAGES_EVERY:-}"
export SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-}"
export MODEL_TIMESTEPS="${MODEL_TIMESTEPS:-}"
export SAMPLE_TIMESTEPS="${SAMPLE_TIMESTEPS:-}"

export LEARNING_RATE="${LEARNING_RATE:-}"
export WARMUP_STEPS="${WARMUP_STEPS:-}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-}"
export B2="${B2:-}"

export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export EVAL_ENABLED="${EVAL_ENABLED:-false}"
export BASELINE_ARCH_114M="${BASELINE_ARCH_114M:-false}"
export DISABLE_AUGMENT="${DISABLE_AUGMENT:-false}"
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

exec "${SCRIPT_DIR}/submit_train.sh"
