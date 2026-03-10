from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _int_or_none(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mu = _mean(values)
    assert mu is not None
    return float(math.sqrt(sum((value - mu) ** 2 for value in values) / len(values)))


def _format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_scalar(value) for key, value in row.items()})


def _collect_rows(
    manifest_rows: list[dict[str, str]],
    *,
    strict: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for row in manifest_rows:
        summary_path = Path(str(row.get("summary_json", "")))
        metrics_path = Path(str(row.get("metrics_json", "")))

        payload = None
        source = None
        if summary_path.exists():
            payload = _read_json(summary_path)
            source = summary_path
        elif metrics_path.exists():
            payload = _read_json(metrics_path)
            source = metrics_path
        else:
            skipped.append(
                {
                    "candidate": row.get("candidate", ""),
                    "seed": row.get("seed", ""),
                    "nfe_budget": row.get("nfe_budget", ""),
                    "reason": "missing_summary_and_metrics",
                }
            )
            if strict:
                raise FileNotFoundError(
                    f"Neither summary nor metrics exists for candidate={row.get('candidate')} "
                    f"seed={row.get('seed')} nfe={row.get('nfe_budget')}"
                )
            continue

        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), dict) else {}
        experiment_config = (
            payload.get("experiment_config", {})
            if isinstance(payload.get("experiment_config", {}), dict)
            else {}
        )
        checkpoint_cfg = payload.get("checkpoint", {}) if isinstance(payload.get("checkpoint", {}), dict) else {}

        fid = _float_or_none(payload.get("fid", metrics.get("eval/fid", None)))
        is_mean = _float_or_none(payload.get("is", metrics.get("eval/is", None)))
        is_std = _float_or_none(payload.get("is_std", metrics.get("eval/is_std", None)))

        raw_rows.append(
            {
                "candidate": row.get("candidate", ""),
                "seed": _int_or_none(row.get("seed", "")),
                "nfe_budget": _int_or_none(row.get("nfe_budget", "")),
                "fid": fid,
                "is": is_mean,
                "is_std": is_std,
                "baseline_eta": _float_or_none(row.get("baseline_eta", "")),
                "baseline_tau": _float_or_none(row.get("baseline_tau", "")),
                "requested_eta": _float_or_none(row.get("requested_eta", "")),
                "requested_tau": _float_or_none(row.get("requested_tau", "")),
                "effective_eta": _float_or_none(
                    payload.get("effective_eta", experiment_config.get("jump_eta", None))
                ),
                "effective_tau": _float_or_none(
                    payload.get(
                        "effective_logit_temperature",
                        experiment_config.get("sampler_logit_temperature", None),
                    )
                ),
                "sample_timesteps": _int_or_none(
                    payload.get(
                        "sample_timesteps",
                        payload.get("evaluation", {}).get("sample_timesteps", None),
                    )
                ),
                "fid_num_samples": _int_or_none(
                    payload.get(
                        "fid_num_samples",
                        payload.get("evaluation", {}).get("fid_num_samples", None),
                    )
                ),
                "train_job_id": row.get("train_job_id", ""),
                "eval_job_id": row.get("eval_job_id", ""),
                "dependency": row.get("dependency", ""),
                "train_run_dir": row.get("train_run_dir", ""),
                "checkpoint_dir": row.get("checkpoint_dir", checkpoint_cfg.get("root_dir", "")),
                "eval_run_dir": row.get("eval_run_dir", ""),
                "checkpoint_path": payload.get(
                    "checkpoint_path",
                    checkpoint_cfg.get("checkpoint_path", ""),
                ),
                "summary_json": row.get("summary_json", ""),
                "metrics_json": row.get("metrics_json", ""),
                "source_json": str(source),
            }
        )

    return raw_rows, skipped


def _candidate_budget_summary(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        candidate = str(row["candidate"])
        nfe_budget = int(row["nfe_budget"])
        groups.setdefault((candidate, nfe_budget), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for candidate, nfe_budget in sorted(groups):
        group = groups[(candidate, nfe_budget)]
        fid_values = [float(row["fid"]) for row in group if row.get("fid") is not None]
        is_values = [float(row["is"]) for row in group if row.get("is") is not None]
        summary_rows.append(
            {
                "candidate": candidate,
                "nfe_budget": nfe_budget,
                "num_results": len(group),
                "fid_mean": _mean(fid_values),
                "fid_std": _std(fid_values),
                "fid_min": min(fid_values) if fid_values else None,
                "fid_max": max(fid_values) if fid_values else None,
                "is_mean": _mean(is_values),
                "is_std_across_seeds": _std(is_values),
                "baseline_eta": group[0].get("baseline_eta", None),
                "baseline_tau": group[0].get("baseline_tau", None),
                "effective_eta": group[0].get("effective_eta", None),
                "effective_tau": group[0].get("effective_tau", None),
            }
        )
    return summary_rows


def _candidate_best(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_rows: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        candidate = str(row["candidate"])
        fid_mean = row.get("fid_mean", None)
        if fid_mean is None:
            continue
        current = best_rows.get(candidate)
        if current is None:
            best_rows[candidate] = dict(row)
            continue
        if float(fid_mean) < float(current["fid_mean"]):
            best_rows[candidate] = dict(row)
            continue
        if (
            float(fid_mean) == float(current["fid_mean"])
            and int(row["nfe_budget"]) < int(current["nfe_budget"])
        ):
            best_rows[candidate] = dict(row)

    out = []
    for candidate in sorted(best_rows):
        row = dict(best_rows[candidate])
        row["best_nfe_budget"] = row.pop("nfe_budget")
        row["best_fid_mean"] = row.pop("fid_mean")
        row["best_fid_std"] = row.pop("fid_std")
        out.append(row)
    return out


def run(manifest_path: Path, output_dir: Path, *, strict: bool) -> dict[str, Any]:
    manifest_rows = _read_manifest(manifest_path)
    raw_rows, skipped = _collect_rows(manifest_rows, strict=strict)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.tsv"
    budget_path = output_dir / "candidate_budget_summary.tsv"
    best_path = output_dir / "candidate_best.tsv"
    index_path = output_dir / "aggregate_index.json"

    budget_rows = _candidate_budget_summary(raw_rows)
    best_rows = _candidate_best(budget_rows)

    if raw_rows:
        _write_tsv(raw_path, raw_rows)
    if budget_rows:
        _write_tsv(budget_path, budget_rows)
    if best_rows:
        _write_tsv(best_path, best_rows)

    index_payload = {
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "raw_results_path": str(raw_path),
        "candidate_budget_summary_path": str(budget_path),
        "candidate_best_path": str(best_path),
        "num_manifest_rows": len(manifest_rows),
        "num_raw_results": len(raw_rows),
        "num_skipped": len(skipped),
        "skipped": skipped,
    }
    index_path.write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate anchor-study offline-eval results across seeds and NFE budgets."
    )
    parser.add_argument("--manifest", required=True, help="Path to eval_manifest.tsv")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for aggregate TSV outputs. Defaults to <manifest_dir>/aggregate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any manifest row is missing both summary and metrics JSON.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else manifest_path.parent / "aggregate"
    )
    result = run(manifest_path, output_dir, strict=bool(args.strict))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
