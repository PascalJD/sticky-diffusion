#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PKL="/home/patrick/.cache/discrete_diffusion/params_10000.pkl"
CUDA_DEVICE="${1:-0}"
NUM_SAMPLES=4
BATCH_SIZE=1

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -m sticky.cli.eval_checkpoint \
  experiment=openwebtext/sjd_openwebtext \
  eval=openwebtext \
  offline_eval.checkpoint_dir="$CHECKPOINT_PKL" \
  offline_eval.checkpoint_source=root \
  offline_eval.checkpoint_step=null \
  offline_eval.use_run_config=false \
  eval.text_num_samples="$NUM_SAMPLES" \
  eval.text_batch_size="$BATCH_SIZE" \
  +eval.fid_batch_size=1 \
  +eval.is_batch_size=1 \
  model/anchor@experiment.model.anchor=normal_normalized \
  experiment.model.anchor.dim=768

