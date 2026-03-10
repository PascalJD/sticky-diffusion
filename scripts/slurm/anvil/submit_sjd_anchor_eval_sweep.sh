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

TRAINING_MANIFEST="${TRAINING_MANIFEST:-}"
RUN_DIR_LIST="${RUN_DIR_LIST:-}"
if [[ -z "${TRAINING_MANIFEST}" ]] && [[ -z "${RUN_DIR_LIST}" ]]; then
  echo "Set TRAINING_MANIFEST or RUN_DIR_LIST." >&2
  exit 1
fi
if [[ -n "${TRAINING_MANIFEST}" ]] && [[ -n "${RUN_DIR_LIST}" ]]; then
  echo "Set only one of TRAINING_MANIFEST or RUN_DIR_LIST." >&2
  exit 1
fi

resolve_input_path() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    return 0
  fi
  if [[ "${value}" == /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${CALLER_PWD}/${value}"
  fi
}

if [[ -n "${TRAINING_MANIFEST}" ]]; then
  TRAINING_MANIFEST="$(resolve_input_path "${TRAINING_MANIFEST}")"
  if [[ ! -f "${TRAINING_MANIFEST}" ]]; then
    echo "TRAINING_MANIFEST not found: ${TRAINING_MANIFEST}" >&2
    exit 1
  fi
fi
if [[ -n "${RUN_DIR_LIST}" ]]; then
  RUN_DIR_LIST="$(resolve_input_path "${RUN_DIR_LIST}")"
  if [[ ! -f "${RUN_DIR_LIST}" ]]; then
    echo "RUN_DIR_LIST not found: ${RUN_DIR_LIST}" >&2
    exit 1
  fi
fi

PYTHON_CMD="${PYTHON_CMD:-python3}"
if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
  else
    echo "Need python3 or python on PATH to parse manifests." >&2
    exit 1
  fi
fi

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

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
ANVIL_MODULES="${ANVIL_MODULES:-}"
PLATFORM="${PLATFORM:-single}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
DRY_RUN="${DRY_RUN:-0}"
DEPEND_ON_TRAIN="${DEPEND_ON_TRAIN:-0}"

EXPERIMENT_CFG="${EXPERIMENT_CFG:-sjd_anchor_study_cifar10}"
EVAL_CFG="${EVAL_CFG:-sjd_anchor_study_cifar10}"
CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-final}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-null}"
USE_EMA="${USE_EMA:-true}"

FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-10000}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-128}"
IS_ENABLED="${IS_ENABLED:-false}"
IS_NUM_SAMPLES="${IS_NUM_SAMPLES:-${FID_NUM_SAMPLES}}"
IS_BATCH_SIZE="${IS_BATCH_SIZE:-${FID_BATCH_SIZE}}"
FORCE_IS="${FORCE_IS:-false}"
NFE_BUDGETS="${NFE_BUDGETS:-64 128 256 512}"
CANDIDATES="${CANDIDATES:-}"
ETA_OVERRIDE="${ETA_OVERRIDE:-}"
TAU_OVERRIDE="${TAU_OVERRIDE:-}"

if [[ -n "${SCRATCH:-}" ]]; then
  SCRATCH_ROOT="${SCRATCH}"
elif [[ -d "/anvil/scratch/${USER}" ]]; then
  SCRATCH_ROOT="/anvil/scratch/${USER}"
else
  SCRATCH_ROOT="${HOME}/scratch"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/sticky-diffusion/outputs}"
DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/sticky-diffusion/data/cifar10}"
FID_CACHE_DIR="${FID_CACHE_DIR:-${SCRATCH_ROOT}/sticky-diffusion/data/fid_stats}"
SWEEP_TAG="${SWEEP_TAG:-sjd_anchor_eval_fid${FID_NUM_SAMPLES}_$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-${OUTPUT_ROOT}/anchor_eval_sweeps/${SWEEP_TAG}}"
LOG_DIR="${LOG_DIR:-${SWEEP_ROOT}/logs}"
AGGREGATE_DIR="${AGGREGATE_DIR:-${SWEEP_ROOT}/aggregate}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-sjd_anchor_eval}"
NORMALIZED_RUNS="${SWEEP_ROOT}/normalized_runs.tsv"
SUBMISSION_MANIFEST="${SUBMISSION_MANIFEST:-${SWEEP_ROOT}/eval_manifest.tsv}"
PREVIEW_COMMANDS_PATH="${SWEEP_ROOT}/sbatch_commands.txt"

mkdir -p "${SWEEP_ROOT}" "${LOG_DIR}" "${AGGREGATE_DIR}" "${DATA_DIR}" "${FID_CACHE_DIR}"

"${PYTHON_CMD}" - "${TRAINING_MANIFEST}" "${RUN_DIR_LIST}" "${NORMALIZED_RUNS}" "${CANDIDATES}" <<'PY'
import csv
import json
from pathlib import Path
import sys

training_manifest = sys.argv[1].strip()
run_dir_list = sys.argv[2].strip()
output_path = Path(sys.argv[3])
candidate_filter = set(sys.argv[4].split()) if sys.argv[4].strip() else None


def read_run_context(run_dir: Path) -> dict:
    run_context_path = run_dir / "run_context.json"
    if not run_context_path.exists():
        return {}
    payload = json.loads(run_context_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def baseline_from_run_context(payload: dict) -> tuple[str, str, str]:
    exp_cfg = payload.get("config", {}).get("experiment", {})
    if not isinstance(exp_cfg, dict):
        return "", "", ""
    training_cfg = exp_cfg.get("training", {})
    forward_cfg = exp_cfg.get("forward", {})
    sampler_cfg = exp_cfg.get("sampler", {})
    seed = training_cfg.get("seed", "")
    jump_cfg = forward_cfg.get("jump", {}) if isinstance(forward_cfg, dict) else {}
    eta = jump_cfg.get("eta", "") if isinstance(jump_cfg, dict) else ""
    tau = ""
    if isinstance(sampler_cfg, dict):
        tau = sampler_cfg.get("logit_temperature", sampler_cfg.get("temperature", ""))
    return str(seed), str(eta), str(tau)


def normalize_row(row: dict) -> dict:
    run_dir = Path(str(row.get("run_dir", ""))).expanduser().resolve()
    run_context = read_run_context(run_dir)
    resolved_seed, baseline_eta, baseline_tau = baseline_from_run_context(run_context)
    checkpoint_dir = row.get("checkpoint_dir", "") or run_context.get("checkpoint_dir", "")
    if not checkpoint_dir:
        checkpoint_dir = str(run_dir / "checkpoints")
    candidate = str(row.get("candidate", "") or run_dir.name)
    seed = str(row.get("seed", "") or resolved_seed or "0")
    return {
        "candidate": candidate,
        "seed": seed,
        "train_job_id": str(row.get("train_job_id", "")),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(Path(checkpoint_dir)),
        "baseline_eta": baseline_eta,
        "baseline_tau": baseline_tau,
    }


def rows_from_training_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [normalize_row(row) for row in reader]


def rows_from_run_dir_list(path: Path) -> list[dict]:
    raw_lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in raw_lines if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        return []

    first = lines[0]
    rows = []
    if "\t" in first and "run_dir" in first.split("\t"):
        reader = csv.DictReader(lines, delimiter="\t")
        for row in reader:
            rows.append(normalize_row(row))
        return rows

    for line in lines:
        parts = line.split("\t")
        if len(parts) == 1:
            rows.append(normalize_row({"run_dir": parts[0]}))
            continue
        if len(parts) >= 3:
            row = {
                "candidate": parts[0],
                "seed": parts[1],
                "run_dir": parts[2],
            }
            if len(parts) >= 4:
                row["train_job_id"] = parts[3]
            if len(parts) >= 5:
                row["checkpoint_dir"] = parts[4]
            rows.append(normalize_row(row))
            continue
        raise ValueError(
            "RUN_DIR_LIST lines must be either `run_dir` or "
            "`candidate<TAB>seed<TAB>run_dir[<TAB>train_job_id[<TAB>checkpoint_dir]]`."
        )
    return rows


if training_manifest:
    rows = rows_from_training_manifest(Path(training_manifest))
else:
    rows = rows_from_run_dir_list(Path(run_dir_list))

if candidate_filter is not None:
    rows = [row for row in rows if row["candidate"] in candidate_filter]

def sort_key(row: dict) -> tuple[str, int | str]:
    try:
        seed_key: int | str = int(row["seed"])
    except Exception:
        seed_key = str(row["seed"])
    return str(row["candidate"]), seed_key


rows.sort(key=sort_key)

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "candidate",
            "seed",
            "train_job_id",
            "run_dir",
            "checkpoint_dir",
            "baseline_eta",
            "baseline_tau",
        ],
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
PY

printf 'candidate\tseed\tnfe_budget\ttrain_job_id\teval_job_id\tdependency\ttrain_run_dir\tcheckpoint_dir\teval_run_dir\tmetrics_json\tsummary_json\tcheckpoint_source\tcheckpoint_step\tuse_ema\tfid_num_samples\tfid_batch_size\tis_enabled\tis_num_samples\tis_batch_size\tbaseline_eta\tbaseline_tau\trequested_eta\trequested_tau\teffective_eta\teffective_tau\n' > "${SUBMISSION_MANIFEST}"
: > "${PREVIEW_COMMANDS_PATH}"

echo "Submitting SJD anchor eval sweep"
echo "  repo:              ${REPO_ROOT}"
echo "  training manifest: ${TRAINING_MANIFEST:-<none>}"
echo "  run dir list:      ${RUN_DIR_LIST:-<none>}"
echo "  candidates:        ${CANDIDATES:-<all>}"
echo "  nfe budgets:       ${NFE_BUDGETS}"
echo "  fid_num_samples:   ${FID_NUM_SAMPLES}"
echo "  is_enabled:        ${IS_ENABLED}"
echo "  depend_on_train:   ${DEPEND_ON_TRAIN}"
echo "  sweep root:        ${SWEEP_ROOT}"
echo "  aggregate dir:     ${AGGREGATE_DIR}"
echo "  dry run:           ${DRY_RUN}"
echo

while IFS=$'\t' read -r candidate seed train_job_id train_run_dir checkpoint_dir baseline_eta baseline_tau; do
  if [[ -z "${candidate}" ]]; then
    continue
  fi

  requested_eta="inherit"
  requested_tau="inherit"
  effective_eta="${baseline_eta}"
  effective_tau="${baseline_tau}"
  if [[ -n "${ETA_OVERRIDE}" ]]; then
    requested_eta="${ETA_OVERRIDE}"
    effective_eta="${ETA_OVERRIDE}"
  fi
  if [[ -n "${TAU_OVERRIDE}" ]]; then
    requested_tau="${TAU_OVERRIDE}"
    effective_tau="${TAU_OVERRIDE}"
  fi

  dependency=""
  if is_truthy "${DEPEND_ON_TRAIN}" && [[ "${train_job_id}" =~ ^[0-9]+$ ]]; then
    dependency="afterok:${train_job_id}"
  fi

  for nfe_budget in ${NFE_BUDGETS}; do
    eval_run_dir="${SWEEP_ROOT}/${candidate}/seed_${seed}/nfe_${nfe_budget}"
    metrics_json="${eval_run_dir}/offline_eval_metrics.json"
    summary_json="${eval_run_dir}/summary.json"
    job_name="${JOB_NAME_PREFIX}_${candidate}_s${seed}_n${nfe_budget}"

    export_vars=(
      "REPO_ROOT=${REPO_ROOT}"
      "CONDA_ENV=${CONDA_ENV}"
      "ANVIL_MODULES=${ANVIL_MODULES}"
      "EXPERIMENT_CFG=${EXPERIMENT_CFG}"
      "EVAL_CFG=${EVAL_CFG}"
      "PLATFORM=${PLATFORM}"
      "SEED=${seed}"
      "WANDB_ENABLED=${WANDB_ENABLED}"
      "TRAIN_RUN_DIR=${train_run_dir}"
      "CHECKPOINT_DIR=${checkpoint_dir}"
      "CHECKPOINT_SOURCE=${CHECKPOINT_SOURCE}"
      "CHECKPOINT_STEP=${CHECKPOINT_STEP}"
      "USE_EMA=${USE_EMA}"
      "NFE_BUDGET=${nfe_budget}"
      "FID_NUM_SAMPLES=${FID_NUM_SAMPLES}"
      "FID_BATCH_SIZE=${FID_BATCH_SIZE}"
      "IS_ENABLED=${IS_ENABLED}"
      "IS_NUM_SAMPLES=${IS_NUM_SAMPLES}"
      "IS_BATCH_SIZE=${IS_BATCH_SIZE}"
      "FORCE_IS=${FORCE_IS}"
      "ETA_OVERRIDE=${ETA_OVERRIDE}"
      "TAU_OVERRIDE=${TAU_OVERRIDE}"
      "DATA_DIR=${DATA_DIR}"
      "FID_CACHE_DIR=${FID_CACHE_DIR}"
      "RUN_DIR=${eval_run_dir}"
      "METRICS_PATH=${metrics_json}"
      "SUMMARY_PATH=${summary_json}"
      "CANDIDATE_NAME=${candidate}"
      "SWEEP_MANIFEST=${SUBMISSION_MANIFEST}"
      "AGGREGATE_DIR=${AGGREGATE_DIR}"
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
    if [[ -n "${dependency}" ]]; then
      sbatch_args+=(--dependency="${dependency}")
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf 'DRY_RUN sbatch'
      for arg in "${sbatch_args[@]}"; do
        printf ' %q' "${arg}"
      done
      printf ' %q\n' "${SCRIPT_DIR}/eval_anchor_checkpoint.slurm"
      {
        printf 'DRY_RUN sbatch'
        for arg in "${sbatch_args[@]}"; do
          printf ' %q' "${arg}"
        done
        printf ' %q\n' "${SCRIPT_DIR}/eval_anchor_checkpoint.slurm"
      } >> "${PREVIEW_COMMANDS_PATH}"
      eval_job_id="DRY_RUN"
    else
      sbatch_out="$(sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/eval_anchor_checkpoint.slurm")"
      printf '%s\n' "${sbatch_out}"
      printf '%s\n' "${sbatch_out}" >> "${PREVIEW_COMMANDS_PATH}"
      eval_job_id="$(
        printf '%s\n' "${sbatch_out}" \
          | grep -E '^[0-9]+([;][^[:space:]]+)?$' \
          | tail -n 1 \
          | cut -d';' -f1 \
          || true
      )"
      if [[ -z "${eval_job_id}" ]]; then
        echo "Could not parse eval job id for candidate=${candidate}, seed=${seed}, nfe=${nfe_budget}" >&2
        exit 1
      fi
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${candidate}" \
      "${seed}" \
      "${nfe_budget}" \
      "${train_job_id}" \
      "${eval_job_id}" \
      "${dependency}" \
      "${train_run_dir}" \
      "${checkpoint_dir}" \
      "${eval_run_dir}" \
      "${metrics_json}" \
      "${summary_json}" \
      "${CHECKPOINT_SOURCE}" \
      "${CHECKPOINT_STEP}" \
      "${USE_EMA}" \
      "${FID_NUM_SAMPLES}" \
      "${FID_BATCH_SIZE}" \
      "${IS_ENABLED}" \
      "${IS_NUM_SAMPLES}" \
      "${IS_BATCH_SIZE}" \
      "${baseline_eta}" \
      "${baseline_tau}" \
      "${requested_eta}" \
      "${requested_tau}" \
      "${effective_eta}" \
      "${effective_tau}" >> "${SUBMISSION_MANIFEST}"
  done
done < <(tail -n +2 "${NORMALIZED_RUNS}")

echo
echo "Eval sweep submissions complete."
echo "Manifest: ${SUBMISSION_MANIFEST}"
echo "Aggregate outputs will be refreshed under: ${AGGREGATE_DIR}"
