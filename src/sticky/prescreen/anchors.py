from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from sticky.models.sjd.anchors import (
    AnchorTableConfig,
    anchor_learnable_from_mapping,
    anchor_table_config_from_mapping,
    build_anchor_table_views,
)
from sticky.models.sjd.sdes import alpha_sigma
from sticky.training.eval import resolve_from_original_cwd
from sticky.training.persistence import get_hydra_output_dir, now_utc_iso, resolve_run_path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrescreenCandidate:
    name: str
    preset: str
    learnable: bool = False
    anchor_overrides: dict[str, Any] = field(default_factory=dict)


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _load_anchor_preset(preset_name: str) -> DictConfig:
    repo_root = Path(hydra.utils.get_original_cwd())
    preset_path = repo_root / "config" / "model" / "anchor" / f"{preset_name}.yaml"
    if not preset_path.exists():
        raise FileNotFoundError(f"Anchor preset does not exist: {preset_path}")
    return OmegaConf.load(preset_path)


def _candidate_from_cfg(cfg: Any) -> PrescreenCandidate:
    name = str(cfg.get("name"))
    preset = str(cfg.get("preset"))
    learnable = bool(cfg.get("learnable", False))
    raw_overrides = cfg.get("anchor_overrides", None)
    if raw_overrides is None:
        anchor_overrides = {}
    elif OmegaConf.is_config(raw_overrides):
        anchor_overrides = OmegaConf.to_container(raw_overrides, resolve=True)
    elif isinstance(raw_overrides, dict):
        anchor_overrides = dict(raw_overrides)
    else:
        anchor_overrides = raw_overrides
    if anchor_overrides is None:
        anchor_overrides = {}
    if not isinstance(anchor_overrides, dict):
        raise ValueError(
            f"Candidate {name!r} anchor_overrides must resolve to a mapping, "
            f"got {type(anchor_overrides).__name__}."
        )
    return PrescreenCandidate(
        name=name,
        preset=preset,
        learnable=learnable,
        anchor_overrides=anchor_overrides,
    )


def _load_candidates(cfg: DictConfig) -> list[PrescreenCandidate]:
    candidates_cfg = cfg.prescreen.get("candidates", [])
    candidates = [_candidate_from_cfg(item) for item in candidates_cfg]
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError(f"Candidate names must be unique, got {names}.")
    if not candidates:
        raise ValueError("prescreen.candidates must contain at least one candidate.")
    return candidates


def _prepare_candidate_experiment_cfg(
    experiment_cfg: DictConfig,
    candidate: PrescreenCandidate,
) -> DictConfig:
    cfg_local = _clone_cfg(experiment_cfg)
    cfg_local.model.anchor = _load_anchor_preset(candidate.preset)
    cfg_local.model.anchor.learnable = bool(candidate.learnable)
    if candidate.anchor_overrides:
        cfg_local.model.anchor = OmegaConf.merge(
            cfg_local.model.anchor,
            OmegaConf.create(candidate.anchor_overrides),
        )
    return cfg_local


def _flatten_overrides(prefix: str, payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in sorted(payload.items()):
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.extend(_flatten_overrides(dotted, value))
        else:
            out.append(f"{dotted}={_format_override_value(value)}")
    return out


def _format_override_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (list, tuple)):
        inner = ",".join(_format_override_value(item) for item in value)
        return f"[{inner}]"
    return str(value)


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _average_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty_like(sorted_vals, dtype=np.float64)

    start = 0
    while start < sorted_vals.size:
        stop = start + 1
        while (stop < sorted_vals.size) and (sorted_vals[stop] == sorted_vals[start]):
            stop += 1
        avg_rank = 0.5 * (start + stop - 1) + 1.0
        ranks[start:stop] = avg_rank
        start = stop

    out = np.empty_like(ranks)
    out[order] = ranks
    return out


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    xr = _average_rank(x)
    yr = _average_rank(y)
    xc = xr - np.mean(xr)
    yc = yr - np.mean(yr)
    denom = np.linalg.norm(xc) * np.linalg.norm(yc)
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(xc, yc) / denom)


def _alpha_sigma_arrays(beta, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha, sigma = alpha_sigma(beta, np.asarray(times, dtype=np.float32))
    return np.asarray(alpha, dtype=np.float64), np.asarray(sigma, dtype=np.float64)


def _alpha_eq_sigma_time(beta, *, steps: int = 80) -> float | None:
    T = float(getattr(beta, "T", 1.0))

    def f(t: float) -> float:
        alpha, sigma = _alpha_sigma_arrays(beta, np.asarray([t], dtype=np.float64))
        return float(alpha[0] - sigma[0])

    lo = 0.0
    hi = T
    flo = f(lo)
    fhi = f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        return None

    for _ in range(int(steps)):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid == 0.0:
            return mid
        if flo * fmid > 0.0:
            lo = mid
            flo = fmid
        else:
            hi = mid
            fhi = fmid
    return 0.5 * (lo + hi)


def _hazard_survival_50_time(hazard) -> float | None:
    if hazard is None:
        return None
    T = float(getattr(hazard, "T", 1.0))
    final_cdf = float(np.asarray(hazard.cdf(np.asarray(T, dtype=np.float32))))
    if final_cdf < 0.5:
        return None
    t50 = hazard.inv_cdf(np.asarray(0.5, dtype=np.float32))
    return float(np.asarray(t50))


def _manual_reference_time(cfg: DictConfig, *, T: float) -> float | None:
    manual_t = cfg.prescreen.reference_time.get("manual_t", None)
    if manual_t in (None, "", "null"):
        return None
    manual_t = float(manual_t)
    if (manual_t < 0.0) or (manual_t > T):
        raise ValueError(
            f"prescreen.reference_time.manual_t must lie in [0, {T}], got {manual_t}."
        )
    return manual_t


def _reference_times(cfg: DictConfig, *, beta, hazard) -> dict[str, Any]:
    T = float(getattr(beta, "T", cfg.experiment.sampler.get("T", 1.0)))
    available = {
        "alpha_eq_sigma": _alpha_eq_sigma_time(beta),
        "hazard_survival_50": _hazard_survival_50_time(hazard),
        "manual": _manual_reference_time(cfg, T=T),
    }
    mode = str(cfg.prescreen.reference_time.get("mode", "alpha_eq_sigma"))
    if mode not in available:
        raise ValueError(
            f"Unknown prescreen.reference_time.mode={mode!r}. "
            "Expected one of: alpha_eq_sigma, hazard_survival_50, manual."
        )
    selected_t = available[mode]
    if selected_t is None:
        raise ValueError(
            f"Reference-time mode {mode!r} is unavailable for this configuration."
        )
    return {
        "selected_mode": mode,
        "selected_t": float(selected_t),
        "available": {
            key: (None if value is None else float(value))
            for key, value in available.items()
        },
    }


def _pairwise_geometry(table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diffs = table[:, None, :] - table[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    upper = np.triu_indices(int(table.shape[0]), k=1)
    pairwise = distances[upper]
    token_gap = np.abs(upper[0] - upper[1]).astype(np.float64)
    return distances, pairwise, token_gap


def _identifiability_curves(
    *,
    beta,
    nn_distances: np.ndarray,
    times: np.ndarray,
    quantiles: list[float],
) -> dict[str, Any]:
    alpha, sigma = _alpha_sigma_arrays(beta, times)
    ratio = alpha / np.maximum(sigma, 1e-12)
    values = ratio[:, None] * nn_distances[None, :]

    out = {
        "times": [float(value) for value in times],
        "median": [float(value) for value in np.median(values, axis=1)],
        "quantiles": {},
    }
    for q in quantiles:
        key = f"q{int(round(100.0 * q)):02d}"
        out["quantiles"][key] = [float(value) for value in np.quantile(values, q, axis=1)]
    return out


def _identifiability_at_time(
    *,
    beta,
    nn_distances: np.ndarray,
    t_ref: float,
    quantiles: list[float],
) -> dict[str, Any]:
    alpha, sigma = _alpha_sigma_arrays(beta, np.asarray([t_ref], dtype=np.float64))
    factor = float(alpha[0] / max(sigma[0], 1e-12))
    values = factor * nn_distances
    out = {
        "t_ref": float(t_ref),
        "median": float(np.median(values)),
        "quantiles": {},
    }
    for q in quantiles:
        key = f"q{int(round(100.0 * q)):02d}"
        out["quantiles"][key] = float(np.quantile(values, q))
    return out


def _stage_report(
    *,
    table: np.ndarray,
    beta,
    times: np.ndarray,
    t_ref: float,
    ident_quantiles: list[float],
) -> dict[str, Any]:
    distances, pairwise, token_gap = _pairwise_geometry(table)
    row_norms = np.linalg.norm(table, axis=1)

    masked = np.array(distances, copy=True)
    np.fill_diagonal(masked, np.inf)
    nn_distances = np.min(masked, axis=1)
    row_norm_mean = float(np.mean(row_norms))
    row_norm_std = float(np.std(row_norms))
    row_norm_cv = float(row_norm_std / row_norm_mean) if row_norm_mean > 0.0 else float("nan")

    return {
        "shape": [int(table.shape[0]), int(table.shape[1])],
        "nearest_neighbor_distance": {
            **_summary_stats(nn_distances),
            "histogram_values": [float(value) for value in nn_distances],
        },
        "pairwise_distance": _summary_stats(pairwise),
        "row_norms": {
            **_summary_stats(row_norms),
            "mean": row_norm_mean,
            "std": row_norm_std,
            "cv": row_norm_cv,
            "values": [float(value) for value in row_norms],
        },
        "spearman_pairwise_distance_vs_token_gap": _spearman_corr(pairwise, token_gap),
        "identifiability_curve": _identifiability_curves(
            beta=beta,
            nn_distances=nn_distances,
            times=times,
            quantiles=ident_quantiles,
        ),
        "identifiability_at_t_ref": _identifiability_at_time(
            beta=beta,
            nn_distances=nn_distances,
            t_ref=t_ref,
            quantiles=ident_quantiles,
        ),
    }


def _stage_tables(
    anchor_config: AnchorTableConfig,
) -> dict[str, np.ndarray]:
    views = build_anchor_table_views(anchor_config)
    transformed = np.asarray(views.transformed, dtype=np.float64)
    return {
        "raw": np.asarray(views.raw, dtype=np.float64),
        "transformed": transformed,
        "configured": np.asarray(views.final, dtype=np.float64),
    }


def _matched_scale(
    *,
    beta,
    transformed_table: np.ndarray,
    t_ref: float,
    target_identifiability: float,
) -> dict[str, float]:
    stage = _stage_report(
        table=transformed_table,
        beta=beta,
        times=np.asarray([t_ref], dtype=np.float64),
        t_ref=t_ref,
        ident_quantiles=[0.5],
    )
    base_ident = float(stage["identifiability_at_t_ref"]["median"])
    if not math.isfinite(base_ident) or (base_ident <= 0.0):
        raise ValueError(
            "Cannot compute matched scale because transformed identifiability at "
            f"t_ref={t_ref} is {base_ident}."
        )
    scale = float(target_identifiability / base_ident)
    return {
        "recommended_scale": scale,
        "transformed_identifiability_at_t_ref": base_ident,
        "matched_identifiability_at_t_ref": float(target_identifiability),
    }


def _json_default(value: Any):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(manifest_path: Path, entries: list[dict[str, Any]]) -> None:
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True, default=_json_default) + "\n")


def _write_slurm_snippets(path: Path, entries: list[dict[str, Any]]) -> None:
    lines = ["# candidate\toverrides"]
    for entry in entries:
        lines.append(f"{entry['candidate']}\t{entry['override_string']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_candidate(
    *,
    output_dir: Path,
    candidate_name: str,
    stage_reports: dict[str, dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        LOGGER.warning("Skipping plots because matplotlib could not be imported: %s", exc)
        return []

    plot_paths: list[str] = []

    ident_path = output_dir / f"{candidate_name}_identifiability.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for stage_name in ("raw", "transformed", "matched_scaled"):
        curve = stage_reports[stage_name]["identifiability_curve"]
        ax.plot(curve["times"], curve["median"], label=stage_name)
    ax.set_xlabel("t")
    ax.set_ylabel("Median identifiability")
    ax.set_title(f"{candidate_name}: identifiability vs time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ident_path, dpi=160)
    plt.close(fig)
    plot_paths.append(str(ident_path))

    norms_path = output_dir / f"{candidate_name}_row_norms.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for stage_name in ("raw", "transformed", "matched_scaled"):
        values = stage_reports[stage_name]["row_norms"]["values"]
        ax.plot(values, label=stage_name)
    ax.set_xlabel("token id")
    ax.set_ylabel("row norm")
    ax.set_title(f"{candidate_name}: row norms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(norms_path, dpi=160)
    plt.close(fig)
    plot_paths.append(str(norms_path))

    hist_path = output_dir / f"{candidate_name}_nn_hist.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for stage_name in ("raw", "transformed", "matched_scaled"):
        values = stage_reports[stage_name]["nearest_neighbor_distance"]["histogram_values"]
        ax.hist(values, bins=24, alpha=0.45, label=stage_name)
    ax.set_xlabel("nearest-neighbor distance")
    ax.set_ylabel("count")
    ax.set_title(f"{candidate_name}: nearest-neighbor distances")
    ax.legend()
    fig.tight_layout()
    fig.savefig(hist_path, dpi=160)
    plt.close(fig)
    plot_paths.append(str(hist_path))

    return plot_paths


def _summary_row(
    *,
    candidate: PrescreenCandidate,
    actual_learnable: bool,
    anchor_config: AnchorTableConfig,
    reference: dict[str, Any],
    matched: dict[str, float],
    stage_reports: dict[str, dict[str, Any]],
    configured_scale: float,
    manifest_entry: dict[str, Any],
) -> dict[str, Any]:
    transformed = stage_reports["transformed"]
    matched_scaled = stage_reports["matched_scaled"]
    return {
        "candidate": candidate.name,
        "preset": candidate.preset,
        "learnable": actual_learnable,
        "family": anchor_config.family,
        "dim": anchor_config.anchor_dim,
        "selected_reference_mode": reference["selected_mode"],
        "selected_reference_t": reference["selected_t"],
        "alpha_eq_sigma_t": reference["available"].get("alpha_eq_sigma"),
        "hazard_survival_50_t": reference["available"].get("hazard_survival_50"),
        "manual_t": reference["available"].get("manual"),
        "configured_scale": configured_scale,
        "recommended_scale": matched["recommended_scale"],
        "transformed_identifiability_t_ref": transformed["identifiability_at_t_ref"]["median"],
        "matched_identifiability_t_ref": matched_scaled["identifiability_at_t_ref"]["median"],
        "transformed_nn_median": transformed["nearest_neighbor_distance"]["median"],
        "matched_nn_median": matched_scaled["nearest_neighbor_distance"]["median"],
        "transformed_pairwise_median": transformed["pairwise_distance"]["median"],
        "matched_pairwise_median": matched_scaled["pairwise_distance"]["median"],
        "transformed_row_norm_mean": transformed["row_norms"]["mean"],
        "transformed_row_norm_cv": transformed["row_norms"]["cv"],
        "matched_row_norm_mean": matched_scaled["row_norms"]["mean"],
        "matched_row_norm_cv": matched_scaled["row_norms"]["cv"],
        "transformed_spearman_token_gap": transformed["spearman_pairwise_distance_vs_token_gap"],
        "matched_spearman_token_gap": matched_scaled["spearman_pairwise_distance_vs_token_gap"],
        "override_string": manifest_entry["override_string"],
    }


def _resolved_output_dir(cfg: DictConfig) -> Path:
    default_rel = str(cfg.prescreen.get("output_dir", "prescreen_anchors"))
    return resolve_run_path(
        path_like=cfg.prescreen.get("output_dir", None),
        default_rel=default_rel,
        base_dir=get_hydra_output_dir(),
    )


def _maybe_resolve_plot_dir(cfg: DictConfig, output_dir: Path) -> Path:
    plot_dir_cfg = cfg.prescreen.plots.get("dir", "plots")
    if Path(str(plot_dir_cfg)).is_absolute():
        return Path(str(plot_dir_cfg))
    return output_dir / str(plot_dir_cfg)


def _resolved_root_config_path() -> str:
    return str(Path(resolve_from_original_cwd("config/prescreen_anchors.yaml")))


def run_anchor_prescreen(cfg: DictConfig) -> dict[str, str]:
    experiment_cfg = cfg.experiment
    beta = hydra.utils.instantiate(experiment_cfg.forward.beta)
    hazard_cfg = experiment_cfg.forward.get("hazard", None)
    hazard = hydra.utils.instantiate(hazard_cfg, beta=beta) if hazard_cfg is not None else None

    candidates = _load_candidates(cfg)
    output_dir = _resolved_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = _maybe_resolve_plot_dir(cfg, output_dir)
    if bool(cfg.prescreen.plots.get("enabled", False)):
        plot_dir.mkdir(parents=True, exist_ok=True)

    quantiles = [float(value) for value in cfg.prescreen.get("ident_quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])]
    t_min = float(cfg.prescreen.time_grid.get("t_min", 1e-4))
    t_max = float(cfg.prescreen.time_grid.get("t_max", getattr(beta, "T", 1.0)))
    num_points = int(cfg.prescreen.time_grid.get("num_points", 129))
    if (t_min < 0.0) or (t_max <= t_min):
        raise ValueError(
            f"Invalid prescreen time grid bounds: t_min={t_min}, t_max={t_max}."
        )
    if num_points < 2:
        raise ValueError(
            f"prescreen.time_grid.num_points must be >= 2, got {num_points}."
        )
    times = np.linspace(t_min, t_max, num_points, dtype=np.float64)
    target_identifiability = float(cfg.prescreen.get("target_identifiability", 1.0))
    if target_identifiability <= 0.0:
        raise ValueError(
            f"prescreen.target_identifiability must be positive, got {target_identifiability}."
        )
    reference = _reference_times(cfg, beta=beta, hazard=hazard)

    summary_rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    results: dict[str, Any] = {
        "timestamp_utc": now_utc_iso(),
        "root_config": _resolved_root_config_path(),
        "output_dir": str(output_dir),
        "candidates": [],
    }

    for candidate in candidates:
        candidate_experiment = _prepare_candidate_experiment_cfg(experiment_cfg, candidate)
        anchor_config = anchor_table_config_from_mapping(
            candidate_experiment.model,
            vocab_size=int(candidate_experiment.dataset.vocab_size),
        )
        actual_learnable = anchor_learnable_from_mapping(
            candidate_experiment.model,
            default=bool(candidate.learnable),
        )
        configured_scale = float(candidate_experiment.model.anchor.transform.scale)
        stage_tables = _stage_tables(anchor_config)
        matched = _matched_scale(
            beta=beta,
            transformed_table=stage_tables["transformed"],
            t_ref=float(reference["selected_t"]),
            target_identifiability=target_identifiability,
        )
        stage_tables["matched_scaled"] = (
            stage_tables["transformed"] * float(matched["recommended_scale"])
        )

        stage_reports = {
            stage_name: _stage_report(
                table=table,
                beta=beta,
                times=times,
                t_ref=float(reference["selected_t"]),
                ident_quantiles=quantiles,
            )
            for stage_name, table in stage_tables.items()
        }

        override_items = [
            f"experiment={cfg.prescreen.base_experiment}",
            f"eval={cfg.prescreen.base_eval}",
            f"model/anchor@experiment.model.anchor={candidate.preset}",
            f"experiment.model.anchor.learnable={_format_override_value(actual_learnable)}",
        ]
        manifest_overrides = dict(candidate.anchor_overrides)
        if "transform" in manifest_overrides:
            transform_override = manifest_overrides["transform"]
            if isinstance(transform_override, dict):
                transform_override = dict(transform_override)
                transform_override.pop("scale", None)
                manifest_overrides["transform"] = transform_override
        override_items.extend(
            _flatten_overrides("experiment.model.anchor", manifest_overrides)
        )
        override_items.append(
            "experiment.model.anchor.transform.scale="
            f"{_format_override_value(matched['recommended_scale'])}"
        )
        manifest_entry = {
            "candidate": candidate.name,
            "preset": candidate.preset,
            "learnable": actual_learnable,
            "family": anchor_config.family,
            "dim": int(anchor_config.anchor_dim),
            "anchor_seed": anchor_config.seed,
            "anchor_projection_seed": anchor_config.projection_seed,
            "anchor_order_weight": float(anchor_config.order_weight),
            "anchor_residual_weight": float(anchor_config.residual_weight),
            "anchor_teacher_checkpoint_path": anchor_config.teacher_checkpoint_path,
            "transform": {
                "center_columns": bool(anchor_config.transform.center_columns),
                "whiten": bool(anchor_config.transform.whiten),
                "whiten_eps": float(anchor_config.transform.whiten_eps),
                "equalize_row_norms": bool(anchor_config.transform.equalize_row_norms),
                "target_row_norm": (
                    None
                    if anchor_config.transform.target_row_norm is None
                    else float(anchor_config.transform.target_row_norm)
                ),
                "scale": float(anchor_config.transform.scale),
            },
            "recommended_scale": float(matched["recommended_scale"]),
            "selected_reference_mode": reference["selected_mode"],
            "selected_reference_t": float(reference["selected_t"]),
            "target_identifiability": target_identifiability,
            "override_items": override_items,
            "override_string": " ".join(override_items),
        }
        manifest_entries.append(manifest_entry)

        plot_paths: list[str] = []
        if bool(cfg.prescreen.plots.get("enabled", False)):
            plot_paths = _plot_candidate(
                output_dir=plot_dir,
                candidate_name=candidate.name,
                stage_reports=stage_reports,
            )

        candidate_payload = {
            "candidate": {
                "name": candidate.name,
                "preset": candidate.preset,
                "learnable": actual_learnable,
                "anchor_overrides": candidate.anchor_overrides,
            },
            "anchor": {
                "family": anchor_config.family,
                "dim": anchor_config.anchor_dim,
                "init_std": float(anchor_config.init_std),
                "seed": anchor_config.seed,
                "order_weight": anchor_config.order_weight,
                "residual_weight": anchor_config.residual_weight,
                "projection_seed": anchor_config.projection_seed,
                "teacher_checkpoint_path": anchor_config.teacher_checkpoint_path,
                "configured_scale": configured_scale,
                "transform": {
                    "center_columns": bool(anchor_config.transform.center_columns),
                    "whiten": bool(anchor_config.transform.whiten),
                    "whiten_eps": float(anchor_config.transform.whiten_eps),
                    "equalize_row_norms": bool(anchor_config.transform.equalize_row_norms),
                    "target_row_norm": (
                        None
                        if anchor_config.transform.target_row_norm is None
                        else float(anchor_config.transform.target_row_norm)
                    ),
                    "scale": float(anchor_config.transform.scale),
                },
            },
            "reference_times": reference,
            "matched_scale": matched,
            "stages": stage_reports,
            "manifest": manifest_entry,
            "plots": plot_paths,
        }
        candidate_path = candidate_dir / f"{candidate.name}.json"
        _write_json(candidate_path, candidate_payload)
        results["candidates"].append(
            {
                "name": candidate.name,
                "json_path": str(candidate_path),
            }
        )
        summary_rows.append(
            _summary_row(
                candidate=candidate,
                actual_learnable=actual_learnable,
                anchor_config=anchor_config,
                reference=reference,
                matched=matched,
                stage_reports=stage_reports,
                configured_scale=configured_scale,
                manifest_entry=manifest_entry,
            )
        )

    summary_path = output_dir / "summary.tsv"
    manifest_path = output_dir / "manifest.jsonl"
    slurm_path = output_dir / "slurm_overrides.txt"
    resolved_cfg_path = output_dir / "resolved_config.yaml"

    _write_summary(summary_path, summary_rows)
    _write_manifest(manifest_path, manifest_entries)
    if bool(cfg.prescreen.manifest.get("write_slurm_snippets", True)):
        _write_slurm_snippets(slurm_path, manifest_entries)
    resolved_cfg_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    _write_json(output_dir / "index.json", results)

    return {
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "slurm_path": str(slurm_path),
        "resolved_config_path": str(resolved_cfg_path),
    }
