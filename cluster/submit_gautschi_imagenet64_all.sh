#!/bin/bash
# Submit all ImageNet 64x64 training jobs on Gautschi.
# Usage: bash cluster/submit_gautschi_imagenet64_all.sh [extra hydra overrides...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/gautschi_imagenet64_train.sbatch"

METHODS=(
  "imagenet64/sjd_imagenet64"
  "imagenet64/md4_imagenet64"
  "imagenet64/mdlm_imagenet64"
  "imagenet64/bitdiff_imagenet64"
  "imagenet64/ddpm_imagenet64"
)

SHORT_NAMES=(
  "sjd-imgnet64"
  "md4-imgnet64"
  "mdlm-imgnet64"
  "bitdiff-imgnet64"
  "ddpm-imgnet64"
)

for i in "${!METHODS[@]}"; do
  exp="${METHODS[$i]}"
  job_name="${SHORT_NAMES[$i]}"
  echo "Submitting ${exp} as ${job_name} ..."
  sbatch \
    --job-name="${job_name}" \
    --export="ALL,EXPERIMENT=${exp}" \
    "${SBATCH_SCRIPT}" \
    "$@"
done

echo "All 5 ImageNet64 jobs submitted."
