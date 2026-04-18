#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_DIR [CHECKPOINT_SOURCE]" >&2
  echo "example: $0 /absolute/path/to/run best" >&2
  exit 1
fi

RUN_DIR="$1"
CHECKPOINT_SOURCE="${2:-best}"
OUT_DIR="${RUN_DIR%/}/sampler_comparison"
mkdir -p "${OUT_DIR}"

PYTHONPATH=src python -m sticky.cli.eval_checkpoint \
  experiment=cifar10/sjd_cifar10 \
  eval=cifar10_sjd_ablation \
  offline_eval.run_dir="${RUN_DIR}" \
  offline_eval.use_run_config=false \
  offline_eval.checkpoint_source="${CHECKPOINT_SOURCE}" \
  offline_eval.output_path="${OUT_DIR}/cifar10_sjd_ablation.json" \
  "$@"

echo "Wrote sampler comparison metrics to ${OUT_DIR}" >&2
