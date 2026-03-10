from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from sticky.entrypoints.aggregate_anchor_eval import run as run_eval_aggregate


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}")
        rows.append(payload)
    return rows


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_scalar(value) for key, value in row.items()})


def _format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, (list, tuple)):
        return "|".join(_format_scalar(item) for item in value)
    return str(value)


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "null", "inherit"):
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


def _bool_or_none(value: Any) -> bool | None:
    if value in (None, "", "null"):
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _discover_single_path(
    *,
    candidates: list[Path],
    label: str,
) -> Path:
    if not candidates:
        raise FileNotFoundError(f"Could not discover {label}.")
    candidates = sorted({path.resolve() for path in candidates})
    if len(candidates) > 1:
        listing = "\n".join(f"  - {path}" for path in candidates)
        raise ValueError(f"Found multiple {label} candidates:\n{listing}")
    return candidates[0]


def _resolve_prescreen_root(
    *,
    study_root: Path,
    prescreen_root: Path | None,
    prescreen_manifest: Path | None,
) -> tuple[Path, Path]:
    if prescreen_manifest is not None:
        manifest_path = prescreen_manifest.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Prescreen manifest not found: {manifest_path}")
        root = prescreen_root.resolve() if prescreen_root is not None else manifest_path.parent
        return root, manifest_path

    if prescreen_root is not None:
        root = prescreen_root.resolve()
        manifest_path = root / "manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Expected prescreen manifest at {manifest_path}")
        return root, manifest_path

    manifest_path = _discover_single_path(
        candidates=[
            path
            for path in study_root.rglob("manifest.jsonl")
            if (path.parent / "candidates").is_dir()
        ],
        label="prescreen manifest",
    )
    return manifest_path.parent, manifest_path


def _resolve_eval_manifest(
    *,
    study_root: Path,
    eval_manifest: Path | None,
) -> Path:
    if eval_manifest is not None:
        manifest_path = eval_manifest.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Eval manifest not found: {manifest_path}")
        return manifest_path

    return _discover_single_path(
        candidates=list(study_root.rglob("eval_manifest.tsv")),
        label="eval manifest",
    )


def _summarize_unique(values: list[Any]) -> Any:
    filtered = [value for value in values if value not in (None, "", [], ())]
    if not filtered:
        return None
    unique = []
    seen = set()
    for value in filtered:
        marker = json.dumps(value, sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(value)
    if len(unique) == 1:
        return unique[0]
    return "|".join(_format_scalar(value) for value in unique)


def _load_prescreen_metadata(
    *,
    prescreen_root: Path,
    prescreen_manifest: Path,
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for entry in _read_jsonl(prescreen_manifest):
        candidate = str(entry.get("candidate", ""))
        if not candidate:
            continue
        metadata[candidate] = dict(entry)

    candidate_dir = prescreen_root / "candidates"
    for candidate, entry in list(metadata.items()):
        candidate_json = candidate_dir / f"{candidate}.json"
        if not candidate_json.is_file():
            continue
        payload = _read_json(candidate_json)
        metadata[candidate] = {
            **entry,
            "candidate_payload": payload,
        }
    return metadata


def _load_train_run_context(train_run_dir: str) -> dict[str, Any]:
    if not train_run_dir:
        return {}
    path = Path(train_run_dir).resolve() / "run_context.json"
    if not path.is_file():
        return {}
    return _read_json(path)


def _extract_train_config_fields(payload: dict[str, Any]) -> dict[str, Any]:
    experiment_cfg = payload.get("config", {}).get("experiment", {})
    if not isinstance(experiment_cfg, dict):
        return {}
    task_cfg = experiment_cfg.get("task", {})
    dataset_cfg = experiment_cfg.get("dataset", {})
    training_cfg = experiment_cfg.get("training", {})
    optim_cfg = experiment_cfg.get("optim", {})
    return {
        "short_task_name": task_cfg.get("name", None) if isinstance(task_cfg, dict) else None,
        "short_train_name": training_cfg.get("name", None) if isinstance(training_cfg, dict) else None,
        "short_train_steps": training_cfg.get("num_train_steps", None) if isinstance(training_cfg, dict) else None,
        "short_sample_timesteps": training_cfg.get("sample_timesteps", None) if isinstance(training_cfg, dict) else None,
        "short_batch_size": dataset_cfg.get("batch_size", None) if isinstance(dataset_cfg, dict) else None,
        "short_eval_batch_size": dataset_cfg.get("eval_batch_size", None) if isinstance(dataset_cfg, dict) else None,
        "short_learning_rate": optim_cfg.get("learning_rate", None) if isinstance(optim_cfg, dict) else None,
        "short_warmup_steps": optim_cfg.get("warmup_steps", None) if isinstance(optim_cfg, dict) else None,
        "short_weight_decay": optim_cfg.get("weight_decay", None) if isinstance(optim_cfg, dict) else None,
        "short_b2": optim_cfg.get("b2", None) if isinstance(optim_cfg, dict) else None,
    }


def _rank_rows(
    rows: list[dict[str, Any]],
    *,
    key_fn,
) -> None:
    ranked = sorted(rows, key=key_fn)
    for idx, row in enumerate(ranked, start=1):
        row.setdefault("_ranks", {})
        row["_ranks"][key_fn.__name__] = idx


def _key_best_mean(row: dict[str, Any]) -> tuple[Any, ...]:
    value = row.get("best_fid_mean", None)
    std = row.get("best_fid_std", None)
    high = row.get("highest_nfe_fid_mean", None)
    return (
        value is None,
        float("inf") if value is None else float(value),
        float("inf") if std is None else float(std),
        float("inf") if high is None else float(high),
        str(row.get("candidate", "")),
    )


def _key_highest_nfe(row: dict[str, Any]) -> tuple[Any, ...]:
    value = row.get("highest_nfe_fid_mean", None)
    std = row.get("highest_nfe_fid_std", None)
    best = row.get("best_fid_mean", None)
    return (
        value is None,
        float("inf") if value is None else float(value),
        float("inf") if std is None else float(std),
        float("inf") if best is None else float(best),
        str(row.get("candidate", "")),
    )


def _key_stability(row: dict[str, Any]) -> tuple[Any, ...]:
    value = row.get("stability_score", None)
    best = row.get("best_fid_mean", None)
    return (
        value is None,
        float("inf") if value is None else float(value),
        float("inf") if best is None else float(best),
        str(row.get("candidate", "")),
    )


def _format_metric_pair(mean_value: Any, std_value: Any) -> str:
    if mean_value is None:
        return "-"
    if std_value is None:
        return format(float(mean_value), ".3f")
    return f"{float(mean_value):.3f} +/- {float(std_value):.3f}"


def _format_transform(meta: dict[str, Any]) -> str:
    parts = []
    center = _bool_or_none(meta.get("transform_center_columns", None))
    whiten = _bool_or_none(meta.get("transform_whiten", None))
    equalize = _bool_or_none(meta.get("transform_equalize_row_norms", None))
    scale = _float_or_none(meta.get("transform_scale", None))
    if center is not None:
        parts.append(f"center={center}")
    if whiten is not None:
        parts.append(f"whiten={whiten}")
    if equalize is not None:
        parts.append(f"equalize={equalize}")
    if scale is not None:
        parts.append(f"scale={scale:.4g}")
    return ", ".join(parts) if parts else "-"


def _build_markdown(
    *,
    study_root: Path,
    output_dir: Path,
    prescreen_manifest: Path,
    eval_manifest: Path,
    budgets: list[int],
    rows: list[dict[str, Any]],
) -> str:
    by_best = sorted(rows, key=_key_best_mean)
    by_high = sorted(rows, key=_key_highest_nfe)

    lines = [
        "# Anchor Study Report",
        "",
        f"- Study root: `{study_root}`",
        f"- Prescreen manifest: `{prescreen_manifest}`",
        f"- Eval manifest: `{eval_manifest}`",
        f"- Report dir: `{output_dir}`",
        f"- Tested NFE budgets: `{', '.join(str(budget) for budget in budgets) if budgets else 'none'}`",
        f"- Candidates summarized: `{len(rows)}`",
        "",
        "## Ranking: Best Mean FID Over Budgets",
        "",
        "| Rank | Candidate | Family | Transform | Matched Scale | Best NFE | Best Mean FID | Highest-NFE Mean FID | Stability |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: |",
    ]
    for rank, row in enumerate(by_best, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(row.get("candidate", "")),
                    str(row.get("anchor_family", "")),
                    _format_transform(row),
                    _format_scalar(row.get("matched_scale", None)),
                    _format_scalar(row.get("best_nfe_budget", None)),
                    _format_metric_pair(row.get("best_fid_mean", None), row.get("best_fid_std", None)),
                    _format_metric_pair(
                        row.get("highest_nfe_fid_mean", None),
                        row.get("highest_nfe_fid_std", None),
                    ),
                    _format_scalar(row.get("stability_score", None)),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Ranking: Mean FID At Highest NFE Budget",
            "",
            "| Rank | Candidate | Highest NFE | Highest-NFE Mean FID | Best Mean FID | Stability |",
            "| --- | --- | ---: | --- | --- | ---: |",
        ]
    )
    for rank, row in enumerate(by_high, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(row.get("candidate", "")),
                    _format_scalar(row.get("highest_nfe_budget", None)),
                    _format_metric_pair(
                        row.get("highest_nfe_fid_mean", None),
                        row.get("highest_nfe_fid_std", None),
                    ),
                    _format_metric_pair(row.get("best_fid_mean", None), row.get("best_fid_std", None)),
                    _format_scalar(row.get("stability_score", None)),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Top Candidates Reproduction Notes", ""])
    for row in by_best[: min(5, len(by_best))]:
        lines.extend(
            [
                f"### {row.get('candidate', '')}",
                f"- Preset/family: `{row.get('preset', '')}` / `{row.get('anchor_family', '')}`",
                f"- Learnable: `{row.get('learnable', '')}`",
                f"- Matched scale: `{_format_scalar(row.get('matched_scale', None))}`",
                f"- Transform: `{_format_transform(row)}`",
                f"- Short run: `steps={_format_scalar(row.get('short_train_steps', None))}`, `batch={_format_scalar(row.get('short_batch_size', None))}`, `eval_batch={_format_scalar(row.get('short_eval_batch_size', None))}`, `lr={_format_scalar(row.get('short_learning_rate', None))}`",
                f"- Seeds: `{row.get('train_seeds', '')}`",
                f"- Best budget/result: `nfe={_format_scalar(row.get('best_nfe_budget', None))}`, `{_format_metric_pair(row.get('best_fid_mean', None), row.get('best_fid_std', None))}`",
                f"- Highest-NFE result: `nfe={_format_scalar(row.get('highest_nfe_budget', None))}`, `{_format_metric_pair(row.get('highest_nfe_fid_mean', None), row.get('highest_nfe_fid_std', None))}`",
                f"- Prescreen override string: `{row.get('prescreen_override_string', '')}`",
                f"- Train run dirs: `{row.get('train_run_dirs', '')}`",
                f"- Metrics JSONs: `{row.get('metrics_jsons', '')}`",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def run(
    *,
    study_root: Path,
    output_dir: Path,
    prescreen_root: Path | None,
    prescreen_manifest: Path | None,
    eval_manifest: Path | None,
    strict: bool,
) -> dict[str, str]:
    study_root = study_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_prescreen_root, resolved_prescreen_manifest = _resolve_prescreen_root(
        study_root=study_root,
        prescreen_root=prescreen_root,
        prescreen_manifest=prescreen_manifest,
    )
    resolved_eval_manifest = _resolve_eval_manifest(
        study_root=study_root,
        eval_manifest=eval_manifest,
    )

    aggregate_dir = output_dir / "eval_aggregate"
    aggregate_index = run_eval_aggregate(
        resolved_eval_manifest,
        aggregate_dir,
        strict=bool(strict),
    )
    raw_results_path = Path(aggregate_index["raw_results_path"])
    budget_results_path = Path(aggregate_index["candidate_budget_summary_path"])
    best_results_path = Path(aggregate_index["candidate_best_path"])
    raw_rows = _read_tsv(raw_results_path) if raw_results_path.is_file() else []
    budget_rows = _read_tsv(budget_results_path) if budget_results_path.is_file() else []
    best_rows = _read_tsv(best_results_path) if best_results_path.is_file() else []

    prescreen_meta = _load_prescreen_metadata(
        prescreen_root=resolved_prescreen_root,
        prescreen_manifest=resolved_prescreen_manifest,
    )

    candidate_budget_map: dict[tuple[str, int], dict[str, str]] = {}
    candidate_budgets: dict[str, list[int]] = {}
    for row in budget_rows:
        candidate = str(row.get("candidate", ""))
        nfe_budget = _int_or_none(row.get("nfe_budget", ""))
        if candidate == "" or nfe_budget is None:
            continue
        candidate_budget_map[(candidate, nfe_budget)] = row
        candidate_budgets.setdefault(candidate, []).append(nfe_budget)

    candidate_best_map = {
        str(row.get("candidate", "")): row
        for row in best_rows
        if str(row.get("candidate", "")) != ""
    }

    raw_by_candidate: dict[str, list[dict[str, str]]] = {}
    all_budgets = sorted(
        {
            int(value)
            for values in candidate_budgets.values()
            for value in values
        }
    )
    for row in raw_rows:
        candidate = str(row.get("candidate", ""))
        if candidate:
            raw_by_candidate.setdefault(candidate, []).append(row)

    report_rows: list[dict[str, Any]] = []
    all_candidates = sorted(set(prescreen_meta) | set(raw_by_candidate))
    for candidate in all_candidates:
        meta_entry = prescreen_meta.get(candidate, {})
        candidate_payload = meta_entry.get("candidate_payload", {})
        anchor_payload = candidate_payload.get("anchor", {}) if isinstance(candidate_payload, dict) else {}
        transform_payload = (
            anchor_payload.get("transform", {})
            if isinstance(anchor_payload, dict)
            else {}
        )
        candidate_rows = raw_by_candidate.get(candidate, [])
        budgets = sorted(set(candidate_budgets.get(candidate, [])))
        best_row = candidate_best_map.get(candidate, {})
        highest_budget = max(budgets) if budgets else None
        highest_row = (
            candidate_budget_map.get((candidate, highest_budget), {})
            if highest_budget is not None
            else {}
        )

        train_run_dirs = sorted(
            {
                str(row.get("train_run_dir", ""))
                for row in candidate_rows
                if str(row.get("train_run_dir", "")) != ""
            }
        )
        train_contexts = [
            _load_train_run_context(train_run_dir)
            for train_run_dir in train_run_dirs
        ]
        train_cfg_rows = [_extract_train_config_fields(payload) for payload in train_contexts]
        aggregated_train_cfg: dict[str, Any] = {}
        for key in (
            "short_task_name",
            "short_train_name",
            "short_train_steps",
            "short_sample_timesteps",
            "short_batch_size",
            "short_eval_batch_size",
            "short_learning_rate",
            "short_warmup_steps",
            "short_weight_decay",
            "short_b2",
        ):
            aggregated_train_cfg[key] = _summarize_unique([row.get(key, None) for row in train_cfg_rows])

        stability_values = [
            _float_or_none(row.get("fid_std", None))
            for row in (
                candidate_budget_map[(candidate, budget)]
                for budget in budgets
                if (candidate, budget) in candidate_budget_map
            )
        ]
        stability_values = [value for value in stability_values if value is not None]
        stability_score = _mean(stability_values)

        report_row: dict[str, Any] = {
            "candidate": candidate,
            "preset": meta_entry.get("preset", candidate_payload.get("candidate", {}).get("preset", None)),
            "learnable": _bool_or_none(
                meta_entry.get("learnable", candidate_payload.get("candidate", {}).get("learnable", None))
            ),
            "anchor_family": anchor_payload.get("family", meta_entry.get("family", None)),
            "anchor_dim": _int_or_none(anchor_payload.get("dim", meta_entry.get("dim", None))),
            "anchor_seed": _int_or_none(anchor_payload.get("seed", meta_entry.get("anchor_seed", None))),
            "anchor_projection_seed": _int_or_none(
                anchor_payload.get("projection_seed", meta_entry.get("anchor_projection_seed", None))
            ),
            "anchor_order_weight": _float_or_none(
                anchor_payload.get("order_weight", meta_entry.get("anchor_order_weight", None))
            ),
            "anchor_residual_weight": _float_or_none(
                anchor_payload.get("residual_weight", meta_entry.get("anchor_residual_weight", None))
            ),
            "matched_scale": _float_or_none(
                meta_entry.get(
                    "recommended_scale",
                    candidate_payload.get("matched_scale", {}).get("recommended_scale", None),
                )
            ),
            "configured_scale": _float_or_none(anchor_payload.get("configured_scale", None)),
            "transform_center_columns": _bool_or_none(
                transform_payload.get("center_columns", meta_entry.get("transform", {}).get("center_columns", None))
            ),
            "transform_whiten": _bool_or_none(
                transform_payload.get("whiten", meta_entry.get("transform", {}).get("whiten", None))
            ),
            "transform_whiten_eps": _float_or_none(
                transform_payload.get("whiten_eps", meta_entry.get("transform", {}).get("whiten_eps", None))
            ),
            "transform_equalize_row_norms": _bool_or_none(
                transform_payload.get(
                    "equalize_row_norms",
                    meta_entry.get("transform", {}).get("equalize_row_norms", None),
                )
            ),
            "transform_target_row_norm": _float_or_none(
                transform_payload.get(
                    "target_row_norm",
                    meta_entry.get("transform", {}).get("target_row_norm", None),
                )
            ),
            "transform_scale": _float_or_none(
                transform_payload.get("scale", meta_entry.get("transform", {}).get("scale", None))
            ),
            "selected_reference_mode": meta_entry.get("selected_reference_mode", None),
            "selected_reference_t": _float_or_none(meta_entry.get("selected_reference_t", None)),
            "target_identifiability": _float_or_none(meta_entry.get("target_identifiability", None)),
            "prescreen_override_string": meta_entry.get("override_string", None),
            "short_task_name": aggregated_train_cfg.get("short_task_name", None),
            "short_train_name": aggregated_train_cfg.get("short_train_name", None),
            "short_train_steps": aggregated_train_cfg.get("short_train_steps", None),
            "short_sample_timesteps": aggregated_train_cfg.get("short_sample_timesteps", None),
            "short_batch_size": aggregated_train_cfg.get("short_batch_size", None),
            "short_eval_batch_size": aggregated_train_cfg.get("short_eval_batch_size", None),
            "short_learning_rate": aggregated_train_cfg.get("short_learning_rate", None),
            "short_warmup_steps": aggregated_train_cfg.get("short_warmup_steps", None),
            "short_weight_decay": aggregated_train_cfg.get("short_weight_decay", None),
            "short_b2": aggregated_train_cfg.get("short_b2", None),
            "train_num_seeds": len(
                {
                    str(row.get("seed", ""))
                    for row in candidate_rows
                    if str(row.get("seed", "")) != ""
                }
            ),
            "train_seeds": "|".join(
                sorted(
                    {
                        str(row.get("seed", ""))
                        for row in candidate_rows
                        if str(row.get("seed", "")) != ""
                    }
                )
            ),
            "tested_nfe_budgets": "|".join(str(budget) for budget in budgets),
            "highest_nfe_budget": highest_budget,
            "highest_nfe_fid_mean": _float_or_none(highest_row.get("fid_mean", None)),
            "highest_nfe_fid_std": _float_or_none(highest_row.get("fid_std", None)),
            "best_nfe_budget": _int_or_none(best_row.get("best_nfe_budget", None)),
            "best_fid_mean": _float_or_none(best_row.get("best_fid_mean", None)),
            "best_fid_std": _float_or_none(best_row.get("best_fid_std", None)),
            "stability_score": stability_score,
            "train_run_dirs": "|".join(train_run_dirs),
            "eval_run_dirs": "|".join(
                sorted(
                    {
                        str(row.get("eval_run_dir", ""))
                        for row in candidate_rows
                        if str(row.get("eval_run_dir", "")) != ""
                    }
                )
            ),
            "metrics_jsons": "|".join(
                sorted(
                    {
                        str(row.get("metrics_json", ""))
                        for row in candidate_rows
                        if str(row.get("metrics_json", "")) != ""
                    }
                )
            ),
            "summary_jsons": "|".join(
                sorted(
                    {
                        str(row.get("summary_json", ""))
                        for row in candidate_rows
                        if str(row.get("summary_json", "")) != ""
                    }
                )
            ),
        }

        for budget in all_budgets:
            budget_row = candidate_budget_map.get((candidate, budget), {})
            report_row[f"fid_mean_nfe_{budget}"] = _float_or_none(budget_row.get("fid_mean", None))
            report_row[f"fid_std_nfe_{budget}"] = _float_or_none(budget_row.get("fid_std", None))
            report_row[f"num_seeds_nfe_{budget}"] = _int_or_none(budget_row.get("num_results", None))

        report_rows.append(report_row)

    _rank_rows(report_rows, key_fn=_key_best_mean)
    _rank_rows(report_rows, key_fn=_key_highest_nfe)
    _rank_rows(report_rows, key_fn=_key_stability)

    for row in report_rows:
        ranks = row.pop("_ranks", {})
        row["rank_best_mean_fid"] = ranks.get("_key_best_mean", None)
        row["rank_highest_nfe_fid"] = ranks.get("_key_highest_nfe", None)
        row["rank_stability"] = ranks.get("_key_stability", None)

    report_rows.sort(key=_key_best_mean)

    csv_path = output_dir / "anchor_study_report.csv"
    md_path = output_dir / "anchor_study_report.md"
    index_path = output_dir / "anchor_study_report_index.json"

    _write_csv(csv_path, report_rows)
    md_path.write_text(
        _build_markdown(
            study_root=study_root,
            output_dir=output_dir,
            prescreen_manifest=resolved_prescreen_manifest,
            eval_manifest=resolved_eval_manifest,
            budgets=all_budgets,
            rows=report_rows,
        ),
        encoding="utf-8",
    )
    index_path.write_text(
        json.dumps(
            {
                "study_root": str(study_root),
                "output_dir": str(output_dir),
                "prescreen_root": str(resolved_prescreen_root),
                "prescreen_manifest": str(resolved_prescreen_manifest),
                "eval_manifest": str(resolved_eval_manifest),
                "aggregate_dir": str(aggregate_dir),
                "csv_path": str(csv_path),
                "markdown_path": str(md_path),
                "num_candidates": len(report_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "csv_path": str(csv_path),
        "markdown_path": str(md_path),
        "index_path": str(index_path),
        "aggregate_dir": str(aggregate_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join prescreen and offline-eval outputs into an anchor-study report."
    )
    parser.add_argument("--study-root", required=True, help="Study root directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Report output directory. Defaults to <study-root>/report.",
    )
    parser.add_argument(
        "--prescreen-root",
        default=None,
        help="Optional prescreen output directory containing manifest.jsonl and candidates/.",
    )
    parser.add_argument(
        "--prescreen-manifest",
        default=None,
        help="Optional explicit path to prescreen manifest.jsonl.",
    )
    parser.add_argument(
        "--eval-manifest",
        default=None,
        help="Optional explicit path to eval_manifest.tsv.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if eval summaries are missing instead of skipping them.",
    )
    args = parser.parse_args()

    study_root = Path(args.study_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else study_root / "report"
    )
    prescreen_root = (
        None if args.prescreen_root is None else Path(args.prescreen_root).expanduser().resolve()
    )
    prescreen_manifest = (
        None
        if args.prescreen_manifest is None
        else Path(args.prescreen_manifest).expanduser().resolve()
    )
    eval_manifest = (
        None if args.eval_manifest is None else Path(args.eval_manifest).expanduser().resolve()
    )

    result = run(
        study_root=study_root,
        output_dir=output_dir,
        prescreen_root=prescreen_root,
        prescreen_manifest=prescreen_manifest,
        eval_manifest=eval_manifest,
        strict=bool(args.strict),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
