#!/bin/bash
#SBATCH --job-name=sjd_stage1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

if [[ -z "${MANIFEST:-}" ]]; then
  echo "MANIFEST env var is required." >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "REPO_ROOT not found: ${REPO_ROOT}" >&2
  exit 1
fi
cd "${REPO_ROOT}"

# Optional module list, e.g.:
#   export ANVIL_MODULES="anaconda/2023.09 cuda/12.2"
if [[ -n "${ANVIL_MODULES:-}" ]]; then
  if command -v module >/dev/null 2>&1; then
    module purge || true
    for mod in ${ANVIL_MODULES}; do
      module load "${mod}"
    done
  else
    echo "ANVIL_MODULES was set but 'module' command is unavailable." >&2
  fi
fi

CONDA_ENV="${CONDA_ENV:-sticky}"
if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Load the appropriate Anvil module first." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mapfile -t RUN_LINES < <(awk 'NF && $1 !~ /^#/' "${MANIFEST}")
if [[ "${#RUN_LINES[@]}" -eq 0 ]]; then
  echo "No runnable lines in manifest: ${MANIFEST}" >&2
  exit 1
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#RUN_LINES[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${IDX} out of range [0, $(( ${#RUN_LINES[@]} - 1 ))]" >&2
  exit 1
fi

RUN_OVERRIDES_LINE="${RUN_LINES[$IDX]}"

RUN_NAME="$(awk '
  {
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^hydra\.job\.name=/) {
        sub(/^hydra\.job\.name=/, "", $i)
        print $i
        exit
      }
    }
  }
' <<< "${RUN_OVERRIDES_LINE}")"
if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="run_${IDX}"
fi

TUNING_TAG="${TUNING_TAG:-sjd_stage1_adhoc}"
RUN_DIR_OVERRIDE="hydra.run.dir=outputs/tuning/${TUNING_TAG}/${RUN_NAME}"

# Stage-1 baseline defaults for CIFAR-10 SJD (CADD-like setup).
BASE_OVERRIDES_DEFAULT="\
experiment=sjd_cifar10 \
eval=sjd_cifar10 \
wandb.enabled=false \
experiment.runtime.platform=pmap \
experiment.model.image_backbone=adm_unet5d \
experiment/forward/jump=vp_matched \
experiment/forward/hazard=poly_alpha \
experiment.forward.jump.eta=0.8 \
experiment.forward.hazard.p=1.0 \
experiment.sampler.n_steps=512 \
experiment.sampler.logit_temperature=1.0 \
experiment.model.learnable_anchors=false \
experiment.optim.learning_rate=1e-4 \
experiment.optim.b2=0.999 \
experiment.dataset.augment.enabled=false \
experiment.model.dropout_rate=0.1 \
eval.fid_enabled=true \
eval.fid_every=50000 \
eval.fid_num_samples=10000 \
eval.is_enabled=false \
experiment.training.checkpoint_every_steps=10000"

BASE_OVERRIDES="${BASE_OVERRIDES:-${BASE_OVERRIDES_DEFAULT}}"

read -r -a BASE_OVERRIDES_ARR <<< "${BASE_OVERRIDES}"
read -r -a RUN_OVERRIDES_ARR <<< "${RUN_OVERRIDES_LINE}"

echo "[$(date)] Host: $(hostname)"
echo "Manifest: ${MANIFEST}"
echo "Array index: ${IDX}"
echo "Run name: ${RUN_NAME}"
echo "Hydra run dir: outputs/tuning/${TUNING_TAG}/${RUN_NAME}"
echo "Run overrides: ${RUN_OVERRIDES_LINE}"
echo

python -m sticky.entrypoints.train \
  "${BASE_OVERRIDES_ARR[@]}" \
  "${RUN_OVERRIDES_ARR[@]}" \
  "${RUN_DIR_OVERRIDE}"
