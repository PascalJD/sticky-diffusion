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

run_eval() {
  local name="$1"
  shift
  python -m sticky.cli.eval_checkpoint \
    experiment=sudoku/mdlm_sudoku_tfw_top_prob_margin \
    eval=sudoku_mdlm \
    offline_eval.run_dir="${RUN_DIR}" \
    offline_eval.use_run_config=false \
    offline_eval.checkpoint_source="${CHECKPOINT_SOURCE}" \
    offline_eval.output_path="${OUT_DIR}/${name}.json" \
    "$@"
}

run_eval "uniform" \
  sampler=mdlm_sudoku_uniform

run_eval "top_probability" \
  sampler=mdlm_sudoku_top_probability

run_eval "top_prob_margin" \
  sampler=mdlm_sudoku_top_prob_margin

run_eval "top_prob_margin_gumbel_0p5" \
  sampler=mdlm_sudoku_tfw_top_prob_margin

echo "Wrote sampler comparison metrics to ${OUT_DIR}" >&2
