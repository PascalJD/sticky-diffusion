#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_EXTRA_OVERRIDES="experiment.model.corrector_enabled=false experiment.model.corrector_steps=0 experiment.model.corrector_remask_frac=0.0"
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  EXTRA_OVERRIDES="${DEFAULT_EXTRA_OVERRIDES} ${EXTRA_OVERRIDES}"
else
  EXTRA_OVERRIDES="${DEFAULT_EXTRA_OVERRIDES}"
fi

export MODEL="${MODEL:-cadd}"
export EXPERIMENT_CFG="${EXPERIMENT_CFG:-cadd_cifar10}"
export EVAL_CFG="${EVAL_CFG:-cadd_cifar10}"

export JOB_NAME="${JOB_NAME:-sticky_cadd}"
export PARTITION="${PARTITION:-ai}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-48}"
export MEMORY="${MEMORY:-480G}"
export TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

export PLATFORM="${PLATFORM:-pmap}"
export REQUIRED_LOCAL_DEVICES="${REQUIRED_LOCAL_DEVICES:-${GPUS_PER_NODE}}"

export BATCH_SIZE="${BATCH_SIZE:-512}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
export TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
export LOG_EVERY="${LOG_EVERY:-500}"
export METRICS_EVERY="${METRICS_EVERY:-1000}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
export LOG_IMAGES_EVERY="${LOG_IMAGES_EVERY:-0}"
export SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-true}"

export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export EVAL_ENABLED="${EVAL_ENABLED:-false}"
export BASELINE_ARCH_114M="${BASELINE_ARCH_114M:-true}"
export DISABLE_AUGMENT="${DISABLE_AUGMENT:-true}"
export DISABLE_CORRECTOR="${DISABLE_CORRECTOR:-true}"
export EXTRA_OVERRIDES

exec "${SCRIPT_DIR}/submit_train.sh"
