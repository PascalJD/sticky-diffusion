#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CALLER_PWD="${PWD}"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "REPO_ROOT not found: ${REPO_ROOT}" >&2
  exit 1
fi
cd "${REPO_ROOT}"

PRESCREEN_MANIFEST="${PRESCREEN_MANIFEST:-}"
if [[ -z "${PRESCREEN_MANIFEST}" ]]; then
  echo "PRESCREEN_MANIFEST is required and should point to prescreen manifest.jsonl." >&2
  exit 1
fi
if [[ "${PRESCREEN_MANIFEST}" != /* ]]; then
  PRESCREEN_MANIFEST="${CALLER_PWD}/${PRESCREEN_MANIFEST}"
fi
if [[ ! -f "${PRESCREEN_MANIFEST}" ]]; then
  echo "PRESCREEN_MANIFEST not found: ${PRESCREEN_MANIFEST}" >&2
  exit 1
fi

PYTHON_CMD="${PYTHON_CMD:-python3}"
if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
  else
    echo "Need python3 or python on PATH to read the prescreen manifest." >&2
    exit 1
  fi
fi

PARTITION="${PARTITION:-ai}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
CONSTRAINT="${CONSTRAINT:-}"
EXCLUDE="${EXCLUDE:-}"
NODELIST="${NODELIST:-}"

TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-48}"
MEMORY="${MEMORY:-480G}"

CONDA_ENV="${CONDA_ENV:-sticky}"
ANVIL_MODULES="${ANVIL_MODULES:-}"
PLATFORM="${PLATFORM:-pmap}"
REQUIRED_LOCAL_DEVICES="${REQUIRED_LOCAL_DEVICES:-${GPUS_PER_NODE}}"

WANDB_ENABLED="${WANDB_ENABLED:-false}"
EVAL_ENABLED="${EVAL_ENABLED:-false}"
DRY_RUN="${DRY_RUN:-0}"

RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-sjd_anchor_short}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-sticky_sjd_anchor}"
SEEDS="${SEEDS:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
LEARNABLE_OVERRIDE="${LEARNABLE_OVERRIDE:-${ANCHOR_LEARNABLE_OVERRIDE:-}}"

EXPERIMENT_CFG="${EXPERIMENT_CFG:-sjd_anchor_study_cifar10}"
EVAL_CFG="${EVAL_CFG:-sjd_anchor_study_cifar10}"
CHECKPOINT_SUBDIR="${CHECKPOINT_SUBDIR:-checkpoints}"

if [[ -n "${SCRATCH:-}" ]]; then
  SCRATCH_ROOT="${SCRATCH}"
elif [[ -d "/anvil/scratch/${USER}" ]]; then
  SCRATCH_ROOT="/anvil/scratch/${USER}"
else
  SCRATCH_ROOT="${HOME}/scratch"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/sticky-diffusion/outputs}"
DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/sticky-diffusion/data/cifar10}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-${OUTPUT_ROOT}/anchor_short_submissions/${RUN_TAG_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${SUBMISSION_ROOT}/logs}"
OVERRIDE_DIR="${SUBMISSION_ROOT}/override_files"
PREVIEW_COMMANDS_PATH="${SUBMISSION_ROOT}/sbatch_commands.txt"
SUBMISSION_MANIFEST="${SUBMISSION_MANIFEST:-${SUBMISSION_ROOT}/submission_manifest.tsv}"

mkdir -p "${SUBMISSION_ROOT}" "${LOG_DIR}" "${OVERRIDE_DIR}" "${OUTPUT_ROOT}" "${DATA_DIR}"

if [[ -n "${LEARNABLE_OVERRIDE}" ]]; then
  case "${LEARNABLE_OVERRIDE,,}" in
    true|false) ;;
    *)
      echo "LEARNABLE_OVERRIDE must be empty, true, or false. Got: ${LEARNABLE_OVERRIDE}" >&2
      exit 1
      ;;
  esac
fi

CANDIDATE_TSV="${SUBMISSION_ROOT}/candidates.tsv"
"${PYTHON_CMD}" - "${PRESCREEN_MANIFEST}" "${OVERRIDE_DIR}" "${LEARNABLE_OVERRIDE}" "${CANDIDATE_TSV}" <<'PY'
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
override_dir = Path(sys.argv[2])
learnable_override = sys.argv[3].strip().lower()
candidate_tsv = Path(sys.argv[4])

override_dir.mkdir(parents=True, exist_ok=True)
rows = []
seen = set()

for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line:
        continue
    entry = json.loads(line)
    candidate = str(entry["candidate"])
    if candidate in seen:
        raise ValueError(f"Duplicate candidate in manifest: {candidate}")
    seen.add(candidate)

    override_items = list(entry.get("override_items", []))
    filtered = []
    for item in override_items:
        if item.startswith("experiment=") or item.startswith("eval="):
            continue
        if learnable_override and item.startswith("experiment.model.anchor.learnable="):
            continue
        filtered.append(item)

    if learnable_override:
        filtered.append(f"experiment.model.anchor.learnable={learnable_override}")

    override_path = override_dir / f"{candidate}.txt"
    payload = "\n".join(filtered).strip()
    if payload:
        override_path.write_text(payload + "\n", encoding="utf-8")
    else:
        override_path.write_text("", encoding="utf-8")

    rows.append(
        (
            candidate,
            str(override_path),
            str(entry.get("preset", "")),
            str(entry.get("recommended_scale", "")),
        )
    )

with candidate_tsv.open("w", encoding="utf-8") as handle:
    handle.write("candidate\toverride_file\tpreset\trecommended_scale\n")
    for candidate, override_file, preset, recommended_scale in rows:
        handle.write(
            f"{candidate}\t{override_file}\t{preset}\t{recommended_scale}\n"
        )
PY

printf 'candidate\tseed\ttrain_job_id\trun_dir\tcheckpoint_dir\toverride_file\n' > "${SUBMISSION_MANIFEST}"
: > "${PREVIEW_COMMANDS_PATH}"

echo "Submitting SJD anchor short sweep"
echo "  repo:            ${REPO_ROOT}"
echo "  prescreen file:  ${PRESCREEN_MANIFEST}"
echo "  experiment:      ${EXPERIMENT_CFG}"
echo "  eval:            ${EVAL_CFG}"
echo "  seeds:           ${SEEDS}"
echo "  train steps:     ${TRAIN_STEPS}"
echo "  batch size:      ${BATCH_SIZE:-<experiment default>}"
echo "  output root:     ${OUTPUT_ROOT}"
echo "  submission root: ${SUBMISSION_ROOT}"
echo "  learnable ovrd:  ${LEARNABLE_OVERRIDE:-<manifest default>}"
echo "  dry run:         ${DRY_RUN}"
echo

while IFS=$'\t' read -r candidate override_file preset recommended_scale; do
  if [[ -z "${candidate}" ]]; then
    continue
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Override file for ${candidate}: ${override_file}"
    sed 's/^/  /' "${override_file}"
  fi

  for seed in ${SEEDS}; do
    run_tag="${RUN_TAG_PREFIX}__${candidate}__seed${seed}"
    run_dir="${OUTPUT_ROOT}/${run_tag}"
    checkpoint_dir="${run_dir}/${CHECKPOINT_SUBDIR}"
    job_name="${JOB_NAME_PREFIX}_${candidate}_s${seed}"

    if ! submit_out="$(
      env \
        MODEL=sjd \
        EXPERIMENT_CFG="${EXPERIMENT_CFG}" \
        EVAL_CFG="${EVAL_CFG}" \
        PARTITION="${PARTITION}" \
        ACCOUNT="${ACCOUNT}" \
        QOS="${QOS}" \
        CONSTRAINT="${CONSTRAINT}" \
        EXCLUDE="${EXCLUDE}" \
        NODELIST="${NODELIST}" \
        TIME_LIMIT="${TIME_LIMIT}" \
        NODES="${NODES}" \
        NTASKS="${NTASKS}" \
        GPUS_PER_NODE="${GPUS_PER_NODE}" \
        CPUS_PER_TASK="${CPUS_PER_TASK}" \
        MEMORY="${MEMORY}" \
        CONDA_ENV="${CONDA_ENV}" \
        ANVIL_MODULES="${ANVIL_MODULES}" \
        PLATFORM="${PLATFORM}" \
        REQUIRED_LOCAL_DEVICES="${REQUIRED_LOCAL_DEVICES}" \
        WANDB_ENABLED="${WANDB_ENABLED}" \
        EVAL_ENABLED="${EVAL_ENABLED}" \
        RUN_TAG="${run_tag}" \
        JOB_NAME="${job_name}" \
        OUTPUT_ROOT="${OUTPUT_ROOT}" \
        DATA_DIR="${DATA_DIR}" \
        LOG_DIR="${LOG_DIR}" \
        TRAIN_STEPS="${TRAIN_STEPS}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
        SEED="${seed}" \
        SBATCH_PARSABLE=1 \
        DRY_RUN="${DRY_RUN}" \
        EXTRA_OVERRIDES_FILE="${override_file}" \
        bash "${SCRIPT_DIR}/submit_train.sh"
    )"; then
      printf '%s\n' "${submit_out}"
      exit 1
    fi
    printf '%s\n' "${submit_out}"

    printf '%s\n' "${submit_out}" >> "${PREVIEW_COMMANDS_PATH}"

    job_id="DRY_RUN"
    if [[ "${DRY_RUN}" != "1" ]]; then
      job_id="$(
        printf '%s\n' "${submit_out}" \
          | grep -E '^[0-9]+([;][^[:space:]]+)?$' \
          | tail -n 1 \
          | cut -d';' -f1 \
          || true
      )"
      if [[ -z "${job_id}" ]]; then
        echo "Could not parse job id for candidate=${candidate}, seed=${seed}" >&2
        exit 1
      fi
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${candidate}" \
      "${seed}" \
      "${job_id}" \
      "${run_dir}" \
      "${checkpoint_dir}" \
      "${override_file}" >> "${SUBMISSION_MANIFEST}"
  done
done < <(tail -n +2 "${CANDIDATE_TSV}")

echo
echo "Sweep submission manifest: ${SUBMISSION_MANIFEST}"
echo "Sbatch command log:        ${PREVIEW_COMMANDS_PATH}"
