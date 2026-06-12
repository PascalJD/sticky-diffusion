from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from sticky.core.config_paths import config_root
from sticky.data.sudoku import make_sudoku_board_iterator
from sticky.models.sjd.anchors import AnchorTable
from sticky.rng import make_rng
from sticky.training.csv_report import write_sudoku_sjd_progress_artifacts
from sticky.training.persistence import get_hydra_output_dir


def _should_run_eval(*, step_i: int, every: int, log_at_step_zero: bool) -> bool:
    if every <= 0:
        return False
    if step_i == 0:
        return bool(log_at_step_zero)
    return (step_i % every) == 0


def _sorted_unit_target() -> np.ndarray:
    return np.arange(1, 10, dtype=np.int32)


def _unit_valid_mask(units: np.ndarray) -> np.ndarray:
    units = np.asarray(units, dtype=np.int32)
    if units.ndim != 3 or units.shape[-1] != 9:
        raise ValueError(f"Expected units with shape [B, N, 9], got {units.shape}.")
    return np.all(np.sort(units, axis=-1) == _sorted_unit_target()[None, None, :], axis=-1)


def _row_valid_mask(board: np.ndarray) -> np.ndarray:
    board = np.asarray(board, dtype=np.int32).reshape(-1, 9, 9)
    return _unit_valid_mask(board)


def _col_valid_mask(board: np.ndarray) -> np.ndarray:
    board = np.asarray(board, dtype=np.int32).reshape(-1, 9, 9)
    return _unit_valid_mask(np.transpose(board, (0, 2, 1)))


def _box_valid_mask(board: np.ndarray) -> np.ndarray:
    board = np.asarray(board, dtype=np.int32).reshape(-1, 9, 9)
    boxes = board.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    return _unit_valid_mask(boxes)


def _safe_ratio(num: int | float, den: int | float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _normalize_sampler_method(method: str | None) -> str:
    key = str("uniform" if method is None else method).strip().lower()
    if key == "vanilla":
        return "uniform"
    return key


def _is_interpolation(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith("${")


def _resolve_sampler_default_entry(entry: Any) -> str | None:
    """Return the referenced sampler-group name for a defaults-list entry, or None to skip.

    Supports the two forms used by sampler configs:
      - bare string (relative within `sampler/` group), e.g. ``sjd_sudoku``
      - single-key mapping ``{group_path: name}`` where the group resolves to ``sampler``,
        e.g. ``{/sampler: sjd_sudoku}`` or ``{sampler: sjd_sudoku}``.
    ``_self_`` entries return ``None`` (handled by the caller).
    """
    if isinstance(entry, str):
        token = entry.strip()
        if not token or token == "_self_":
            return None
        return token
    if isinstance(entry, dict) and len(entry) == 1:
        key, value = next(iter(entry.items()))
        group = str(key).strip().lstrip("/")
        if group != "sampler":
            raise ValueError(
                f"Sampler defaults entry references unsupported group {key!r}; "
                "only the 'sampler' group is supported here."
            )
        if value is None:
            return None
        return str(value).strip()
    raise ValueError(f"Unsupported sampler defaults entry: {entry!r}")


def _compose_sampler_group(name: str, _seen: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Hydra-style composition of a sampler config group.

    Recursively resolves each entry in ``defaults``, merging in left-to-right
    order, with the file's own contents merged at the position of ``_self_``
    (or appended last if ``_self_`` is omitted, matching Hydra 1.1+ behavior).
    """
    if name in _seen:
        cycle = " -> ".join(list(_seen) + [name])
        raise ValueError(f"Circular sampler config defaults: {cycle}")
    # Resolve the sampler-group YAML against the per-task subfolders introduced
    # in the config cleanup (sampler/{cifar10,openwebtext,sudoku}/<name>.yaml),
    # falling back to a flat sampler/<name>.yaml for shared bases like base.yaml.
    sampler_root = config_root() / "sampler"
    candidates = [
        sampler_root / "sudoku" / f"{name}.yaml",
        sampler_root / "cifar10" / f"{name}.yaml",
        sampler_root / "openwebtext" / f"{name}.yaml",
        sampler_root / f"{name}.yaml",
    ]
    path = next((p for p in candidates if p.exists()), candidates[-1])
    if not path.exists():
        raise FileNotFoundError(f"Unknown sampler config group {name!r}: {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Sampler config {name!r} did not load as a mapping.")
    defaults_list = raw.pop("defaults", None) or []
    if not isinstance(defaults_list, list):
        raise ValueError(f"Sampler config {name!r} has non-list 'defaults'.")

    next_seen = _seen | {name}
    layers: list[dict[str, Any]] = []
    self_inserted = False
    for entry in defaults_list:
        if isinstance(entry, str) and entry.strip() == "_self_":
            layers.append(raw)
            self_inserted = True
            continue
        ref = _resolve_sampler_default_entry(entry)
        if ref is None:
            continue
        layers.append(_compose_sampler_group(ref, next_seen))
    if not self_inserted:
        layers.append(raw)

    merged = OmegaConf.to_container(
        OmegaConf.merge(*[OmegaConf.create(layer) for layer in layers]),
        resolve=False,
    )
    if not isinstance(merged, dict):
        raise ValueError(f"Composed sampler config {name!r} is not a mapping.")
    return merged


def _load_sampler_group_overrides(name: str) -> dict[str, Any]:
    return _compose_sampler_group(name)


def _iter_named_entries(entries_cfg: Any, field_name: str) -> list[tuple[str | None, dict[str, Any]]]:
    if entries_cfg is None:
        return []

    container = OmegaConf.to_container(entries_cfg, resolve=False)
    if isinstance(container, dict):
        out: list[tuple[str | None, dict[str, Any]]] = []
        for label, entry in container.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Each {field_name} entry must be a mapping of options.")
            out.append((str(label), dict(entry)))
        return out

    if isinstance(container, list):
        out = []
        for entry in container:
            if not isinstance(entry, dict):
                raise ValueError(f"Each {field_name} entry must be a mapping.")
            label = entry.get("label", None)
            out.append((None if label is None else str(label), dict(entry)))
        return out

    raise ValueError(f"{field_name} must be either a mapping or a list of mappings.")


def _overlay_sampler_fields(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in (
        "n_steps",
        "method",
        "sampling_grid",
        "categorical_sampling_policy",
        "decoding_style",
        "revealed_token_sample_mode",
        "cache_predictions",
        "oracle_noise_type",
        "oracle_noise_scale",
    ):
        if key not in src:
            continue
        value = src[key]
        if value is None or _is_interpolation(value):
            continue
        dst[key] = value


def _overlay_sjd_policy_fields(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in (
        "n_steps",
        "policy",
        "sampling_grid",
        "stochastic_k",
        "eta",
        "logit_temperature",
        "log_ratio_clip",
        "init_std",
        "blur_score",
    ):
        if key not in src:
            continue
        value = src[key]
        if value is None or _is_interpolation(value):
            continue
        dst[key] = value


def _overlay_sjd_sampler_fields(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in (
        "T",
        "n_steps",
        "sampling_grid",
        "score_scale",
        "logit_temperature",
        "categorical_sampling_policy",
        "hazard_mode",
        "alloc_mode",
        "log_ratio_clip",
        "init_std",
        "force_classify_at_end",
        "refresh_logits_after_em_step",
        "metrics_count_nfe",
    ):
        if key not in src:
            continue
        value = src[key]
        if value is None or _is_interpolation(value):
            continue
        dst[key] = value


def _resolve_sudoku_sampler_specs(
    *,
    cfg: DictConfig,
    eval_cfg: DictConfig,
    prefix: str,
) -> tuple[list[dict[str, Any]], str, bool]:
    base = {
        "label": "default",
        "metrics_prefix": str(prefix),
        "n_steps": int(cfg.sampler.get("n_steps", cfg.model.get("timesteps", 50))),
        "method": str(cfg.sampler.get("method", "uniform")),
        "sampling_grid": str(cfg.sampler.get("sampling_grid", "loglinear")),
        "categorical_sampling_policy": str(
            cfg.sampler.get("categorical_sampling_policy", "exact")
        ),
        "decoding_style": str(
            cfg.sampler.get(
                "decoding_style",
                cfg.model.get("decoding_style", "monotone_reveal"),
            )
        ),
        "revealed_token_sample_mode": str(
            cfg.sampler.get("revealed_token_sample_mode", "sample")
        ),
        "cache_predictions": bool(cfg.sampler.get("cache_predictions", False)),
        "oracle_noise_type": str(cfg.sampler.get("oracle_noise_type", "none")),
        "oracle_noise_scale": float(cfg.sampler.get("oracle_noise_scale", 0.0)),
    }
    base["effective_method"] = _normalize_sampler_method(base["method"])

    primary_label = str(eval_cfg.get("sudoku_primary_sampler_label", base["label"]))
    run_all = bool(eval_cfg.get("sudoku_run_all_sampler_modes", False))
    if not run_all:
        return [base], primary_label, False

    resolved_specs: list[dict[str, Any]] = []
    for entry_label, entry in _iter_named_entries(
        eval_cfg.get("sudoku_eval_samplers", None),
        "sudoku_eval_samplers",
    ):
        spec = dict(base)
        sampler_group = entry.get("sampler")
        if sampler_group:
            _overlay_sampler_fields(spec, _load_sampler_group_overrides(str(sampler_group)))
        _overlay_sampler_fields(spec, entry)
        label = str(
            entry_label if entry_label is not None else entry.get("label", spec["method"])
        ).strip()
        if not label:
            raise ValueError("Each sudoku_eval_samplers entry requires a non-empty label.")
        spec["label"] = label
        spec["metrics_prefix"] = f"{prefix}/{label}"
        spec["n_steps"] = int(spec["n_steps"])
        spec["oracle_noise_scale"] = float(spec["oracle_noise_scale"])
        spec["effective_method"] = _normalize_sampler_method(spec["method"])
        resolved_specs.append(spec)

    if not resolved_specs:
        return [base], primary_label, False

    labels = {spec["label"] for spec in resolved_specs}
    if primary_label not in labels:
        primary_label = resolved_specs[0]["label"]
    return resolved_specs, primary_label, True


def _resolve_sudoku_sjd_run_specs(
    *,
    cfg: DictConfig,
    eval_cfg: DictConfig,
    prefix: str,
) -> tuple[list[dict[str, Any]], str, bool]:
    base_policy = {
        "kind": "policy",
        "label": "plugin_hazard_eta_0p97",
        "metrics_prefix": str(prefix),
        "policy": str(eval_cfg.get("sudoku_default_policy", "plugin_hazard")),
        "n_steps": int(cfg.sampler.get("n_steps", 50)),
        "sampling_grid": str(cfg.sampler.get("sampling_grid", "uniform")),
        "stochastic_k": bool(eval_cfg.get("sudoku_stochastic_k", False)),
        "eta": float(cfg.get("forward", {}).get("jump", {}).get("eta", 0.97)),
        "logit_temperature": float(cfg.sampler.get("logit_temperature", 1.0)),
        "log_ratio_clip": float(cfg.sampler.get("log_ratio_clip", 10.0)),
        "init_std": float(cfg.sampler.get("init_std", 1.0)),
        "blur_score": False,
    }
    base_sampler = {
        "kind": "sampler",
        "label": "predictor_only",
        "metrics_prefix": str(prefix),
        "T": float(cfg.sampler.get("T", 1.0)),
        "n_steps": int(cfg.sampler.get("n_steps", 50)),
        "sampling_grid": str(cfg.sampler.get("sampling_grid", "uniform")),
        "score_scale": float(cfg.sampler.get("score_scale", 1.0)),
        "logit_temperature": float(cfg.sampler.get("logit_temperature", 1.0)),
        "categorical_sampling_policy": str(
            cfg.sampler.get("categorical_sampling_policy", "legacy_low")
        ),
        "hazard_mode": str(cfg.sampler.get("hazard_mode", "plugin")),
        "alloc_mode": str(cfg.sampler.get("alloc_mode", "sample")),
        "log_ratio_clip": float(cfg.sampler.get("log_ratio_clip", 10.0)),
        "init_std": float(cfg.sampler.get("init_std", 1.0)),
        "force_classify_at_end": bool(cfg.sampler.get("force_classify_at_end", True)),
        "refresh_logits_after_em_step": bool(
            cfg.sampler.get("refresh_logits_after_em_step", False)
        ),
        "metrics_count_nfe": bool(cfg.sampler.get("metrics_count_nfe", True)),
    }

    primary_label = str(
        eval_cfg.get(
            "sudoku_primary_sampler_label",
            base_sampler["label"],
        )
    )
    resolved_specs: list[dict[str, Any]] = []
    for entry_label, entry in _iter_named_entries(
        eval_cfg.get("sudoku_eval_sjd_runs", None),
        "sudoku_eval_sjd_runs",
    ):
        kind = str(entry.get("kind", "sampler")).strip().lower()
        if kind == "policy":
            spec = dict(base_policy)
            _overlay_sjd_policy_fields(spec, entry)
            spec["n_steps"] = int(spec["n_steps"])
            spec["eta"] = float(spec["eta"])
            spec["logit_temperature"] = float(spec["logit_temperature"])
            spec["log_ratio_clip"] = float(spec["log_ratio_clip"])
            spec["init_std"] = float(spec["init_std"])
            spec["blur_score"] = bool(spec["blur_score"])
        elif kind == "sampler":
            spec = dict(base_sampler)
            sampler_group = entry.get("sampler")
            if sampler_group:
                _overlay_sjd_sampler_fields(spec, _load_sampler_group_overrides(str(sampler_group)))
            _overlay_sjd_sampler_fields(spec, entry)
        else:
            raise ValueError(
                "Each sudoku_eval_sjd_runs entry must set kind to 'policy' or 'sampler', "
                f"got {kind!r}."
            )
        label = str(
            entry_label
            if entry_label is not None
            else entry.get(
                "label",
                spec.get("policy", spec["label"]),
            )
        ).strip()
        if not label:
            raise ValueError("Each sudoku_eval_sjd_runs entry requires a non-empty label.")
        spec["label"] = label
        spec["metrics_prefix"] = f"{prefix}/{label}"
        if spec["kind"] == "policy" and bool(spec.get("blur_score", False)):
            if abs(float(spec["eta"]) - 1.0) > 1e-9:
                raise ValueError(
                    f"sudoku_eval_sjd_runs entry {label!r} sets blur_score=true with "
                    f"eta={spec['eta']}: the W-aware plug-in score requires eta == 1.0."
                )
        resolved_specs.append(spec)

    if not resolved_specs:
        return [base_sampler], primary_label, False

    labels = {spec["label"] for spec in resolved_specs}
    if primary_label not in labels:
        primary_label = resolved_specs[0]["label"]
    return resolved_specs, primary_label, True


def _clone_model_for_sampler(model, *, sampler_spec: dict[str, Any]):
    replace_kwargs = {}
    for field, spec_key in (
        ("sampler", "effective_method"),
        ("sampling_grid", "sampling_grid"),
        ("categorical_sampling_policy", "categorical_sampling_policy"),
        ("decoding_style", "decoding_style"),
        ("revealed_token_sample_mode", "revealed_token_sample_mode"),
        ("cache_predictions", "cache_predictions"),
        ("oracle_noise_type", "oracle_noise_type"),
        ("oracle_noise_scale", "oracle_noise_scale"),
    ):
        if hasattr(model, field):
            replace_kwargs[field] = sampler_spec[spec_key]
    if not replace_kwargs:
        return model
    return replace(model, **replace_kwargs)


def _accumulate_board_sudoku_diagnostics(
    totals: dict[str, float],
    diag: dict[str, Any],
) -> None:
    if "selected_margin_sum_total" in diag and "selected_top_prob_margin_sum_total" not in diag:
        totals["selected_top_prob_margin_sum_total"] += float(diag["selected_margin_sum_total"])
    if "selected_margin_count_total" in diag and "selected_top_prob_margin_count_total" not in diag:
        totals["selected_top_prob_margin_count_total"] += float(diag["selected_margin_count_total"])
    for key in (
        "example_step_count",
        "masked_unknown_total_across_steps",
        "selected_count_total_across_steps",
        "selected_top_probability_sum_total",
        "selected_top_probability_count_total",
        "selected_top_prob_margin_sum_total",
        "selected_top_prob_margin_count_total",
        "unknown_token_total",
        "final_masked_unknown_total",
    ):
        if key in diag:
            totals[key] += float(diag[key])


def _accumulate_sjd_sampler_diagnostics(
    totals: dict[str, float],
    diag: dict[str, Any],
    *,
    batch_examples: int,
) -> None:
    for key in (
        "sampling/nfe_total",
        "sampling/continuous_to_anchor_commits_total",
        "sampling/anchor_to_continuous_unstick_attempts_total",
        "sampling/anchor_to_continuous_unstick_accepts_total",
        "sampling/langevin_updates_total",
        "sampling/gate_mean_continuous",
        "sampling/gate_mean_committed",
        "sampling/frac_committed_pre_force",
        "sampling/frac_committed_final",
    ):
        if key not in diag:
            continue
        totals[f"{key}__weighted_sum"] += float(diag[key]) * float(batch_examples)


def _evaluate_board_batch_counts(
    *,
    pred_board: np.ndarray,
    solution_board: np.ndarray,
    clue_board: np.ndarray,
    clue_mask: np.ndarray,
) -> Dict[str, int]:
    pred_board = np.asarray(pred_board, dtype=np.int32)
    solution_board = np.asarray(solution_board, dtype=np.int32)
    clue_board = np.asarray(clue_board, dtype=np.int32)
    clue_mask = np.asarray(clue_mask, dtype=np.bool_)

    if pred_board.shape != solution_board.shape:
        raise ValueError(
            "pred_board and solution_board must have matching shapes, got "
            f"{pred_board.shape} vs {solution_board.shape}."
        )
    if clue_board.shape != solution_board.shape or clue_mask.shape != solution_board.shape:
        raise ValueError(
            "clue_board and clue_mask must match solution_board shape, got "
            f"{clue_board.shape} and {clue_mask.shape} vs {solution_board.shape}."
        )

    unknown_mask = ~clue_mask
    row_valid = _row_valid_mask(pred_board)
    col_valid = _col_valid_mask(pred_board)
    box_valid = _box_valid_mask(pred_board)
    clue_consistent = np.all(np.where(clue_mask, pred_board == clue_board, True), axis=1)
    board_exact = np.all(pred_board == solution_board, axis=1)
    board_valid = np.all(row_valid, axis=1) & np.all(col_valid, axis=1) & np.all(box_valid, axis=1)
    solve = board_exact & clue_consistent & board_valid

    return {
        "num_examples": int(pred_board.shape[0]),
        "unknown_cell_correct": int(np.sum((pred_board == solution_board) & unknown_mask)),
        "unknown_cell_total": int(np.sum(unknown_mask)),
        "full_cell_correct": int(np.sum(pred_board == solution_board)),
        "full_cell_total": int(pred_board.size),
        "board_exact": int(np.sum(board_exact)),
        "solve_count": int(np.sum(solve)),
        "row_valid_total": int(np.sum(row_valid)),
        "row_total": int(row_valid.size),
        "col_valid_total": int(np.sum(col_valid)),
        "col_total": int(col_valid.size),
        "box_valid_total": int(np.sum(box_valid)),
        "box_total": int(box_valid.size),
        "clue_consistent_total": int(np.sum(clue_consistent)),
    }


def _base_totals() -> dict[str, float]:
    return {
        "num_examples": 0,
        "unknown_cell_correct": 0,
        "unknown_cell_total": 0,
        "full_cell_correct": 0,
        "full_cell_total": 0,
        "board_exact": 0,
        "solve_count": 0,
        "row_valid_total": 0,
        "row_total": 0,
        "col_valid_total": 0,
        "col_total": 0,
        "box_valid_total": 0,
        "box_total": 0,
        "clue_consistent_total": 0,
        "num_batches": 0,
        "example_step_count": 0.0,
        "masked_unknown_total_across_steps": 0.0,
        "selected_count_total_across_steps": 0.0,
        "selected_top_probability_sum_total": 0.0,
        "selected_top_probability_count_total": 0.0,
        "selected_top_prob_margin_sum_total": 0.0,
        "selected_top_prob_margin_count_total": 0.0,
        "unknown_token_total": 0.0,
        "final_masked_unknown_total": 0.0,
        "sampling/nfe_total__weighted_sum": 0.0,
        "sampling/continuous_to_anchor_commits_total__weighted_sum": 0.0,
        "sampling/anchor_to_continuous_unstick_attempts_total__weighted_sum": 0.0,
        "sampling/anchor_to_continuous_unstick_accepts_total__weighted_sum": 0.0,
        "sampling/langevin_updates_total__weighted_sum": 0.0,
        "sampling/gate_mean_continuous__weighted_sum": 0.0,
        "sampling/gate_mean_committed__weighted_sum": 0.0,
        "sampling/frac_committed_pre_force__weighted_sum": 0.0,
        "sampling/frac_committed_final__weighted_sum": 0.0,
        "sampling/wallclock_sec_per_board__weighted_sum": 0.0,
    }


def _finalize_metrics(
    *,
    totals: dict[str, float],
    metric_prefix: str,
    n_steps: int,
    checkpoint_source: str,
    include_sjd_sampler_metrics: bool = False,
) -> Dict[str, float]:
    step_den = max(float(totals["example_step_count"]), 1.0)
    top_prob_den = max(float(totals["selected_top_probability_count_total"]), 1.0)
    top_margin_den = max(float(totals["selected_top_prob_margin_count_total"]), 1.0)
    unknown_den = max(float(totals["unknown_token_total"]), 1.0)
    cell_acc_unknown = float(_safe_ratio(totals["unknown_cell_correct"], totals["unknown_cell_total"]))
    board_acc_exact = float(_safe_ratio(totals["board_exact"], totals["num_examples"]))
    full_cell_acc = float(_safe_ratio(totals["full_cell_correct"], totals["full_cell_total"]))
    avg_commits_per_step = float(float(totals["selected_count_total_across_steps"]) / step_den)
    metrics = {
        f"{metric_prefix}/solve_rate": float(_safe_ratio(totals["solve_count"], totals["num_examples"])),
        f"{metric_prefix}/cell_acc_unknown": cell_acc_unknown,
        f"{metric_prefix}/masked_cell_acc": cell_acc_unknown,
        f"{metric_prefix}/full_cell_acc": full_cell_acc,
        f"{metric_prefix}/board_acc_exact": board_acc_exact,
        f"{metric_prefix}/board_acc": board_acc_exact,
        f"{metric_prefix}/row_valid_fraction": float(_safe_ratio(totals["row_valid_total"], totals["row_total"])),
        f"{metric_prefix}/col_valid_fraction": float(_safe_ratio(totals["col_valid_total"], totals["col_total"])),
        f"{metric_prefix}/box_valid_fraction": float(_safe_ratio(totals["box_valid_total"], totals["box_total"])),
        f"{metric_prefix}/clue_consistency_fraction": float(_safe_ratio(totals["clue_consistent_total"], totals["num_examples"])),
        f"{metric_prefix}/mean_masked_unknown_positions_per_step": float(float(totals["masked_unknown_total_across_steps"]) / step_den),
        f"{metric_prefix}/mean_k_selected_per_step": avg_commits_per_step,
        f"{metric_prefix}/avg_commits_per_step": avg_commits_per_step,
        f"{metric_prefix}/mean_selected_top_probability": float(float(totals["selected_top_probability_sum_total"]) / top_prob_den),
        f"{metric_prefix}/mean_selected_top_prob_margin": float(float(totals["selected_top_prob_margin_sum_total"]) / top_margin_den),
        f"{metric_prefix}/final_masked_unknown_fraction": float(float(totals["final_masked_unknown_total"]) / unknown_den),
        f"{metric_prefix}/checkpoint_is_live": float(checkpoint_source == "live"),
        f"{metric_prefix}/checkpoint_is_latest": float(checkpoint_source in {"latest", "periodic", "root"}),
        f"{metric_prefix}/checkpoint_is_best": float(checkpoint_source == "best"),
        f"{metric_prefix}/checkpoint_is_final": float(checkpoint_source == "final"),
        f"{metric_prefix}/num_examples": float(totals["num_examples"]),
        f"{metric_prefix}/num_batches": float(totals["num_batches"]),
        f"{metric_prefix}/sampler_steps": float(n_steps),
        f"{metric_prefix}/n_steps": float(n_steps),
    }
    if include_sjd_sampler_metrics:
        num_examples = max(float(totals["num_examples"]), 1.0)
        metrics.update(
            {
                f"{metric_prefix}/sampling/nfe_total": float(
                    totals["sampling/nfe_total__weighted_sum"] / num_examples
                ),
                f"{metric_prefix}/sampling/continuous_to_anchor_commits_total": float(
                    totals["sampling/continuous_to_anchor_commits_total__weighted_sum"] / num_examples
                ),
                f"{metric_prefix}/sampling/continuous_to_anchor_commits_per_step": float(
                    (totals["sampling/continuous_to_anchor_commits_total__weighted_sum"] / num_examples)
                    / max(float(n_steps), 1.0)
                ),
                f"{metric_prefix}/sampling/anchor_to_continuous_unstick_attempts_total": float(
                    totals["sampling/anchor_to_continuous_unstick_attempts_total__weighted_sum"] / num_examples
                ),
                f"{metric_prefix}/sampling/anchor_to_continuous_unstick_accepts_total": float(
                    totals["sampling/anchor_to_continuous_unstick_accepts_total__weighted_sum"] / num_examples
                ),
                f"{metric_prefix}/sampling/anchor_to_continuous_accept_rate": float(
                    _safe_ratio(
                        totals["sampling/anchor_to_continuous_unstick_accepts_total__weighted_sum"],
                        totals["sampling/anchor_to_continuous_unstick_attempts_total__weighted_sum"],
                    )
                ),
                f"{metric_prefix}/sampling/langevin_updates_total": float(
                    totals["sampling/langevin_updates_total__weighted_sum"] / num_examples
                ),
                f"{metric_prefix}/sampling/langevin_updates_per_step": float(
                    (totals["sampling/langevin_updates_total__weighted_sum"] / num_examples)
                    / max(float(n_steps), 1.0)
                ),
                f"{metric_prefix}/sampling/gate_mean_continuous": float(
                    totals["sampling/gate_mean_continuous__weighted_sum"] / num_examples
                ),
                f"{metric_prefix}/sampling/gate_mean_committed": float(
                    totals["sampling/gate_mean_committed__weighted_sum"] / num_examples
                ),
                f"{metric_prefix}/sampling/frac_committed_pre_force": float(
                    totals["sampling/frac_committed_pre_force__weighted_sum"] / num_examples
                ),
                f"{metric_prefix}/sampling/frac_committed_final": float(
                    totals["sampling/frac_committed_final__weighted_sum"] / num_examples
                ),
                f"{metric_prefix}/sampling/wallclock_sec_per_board": float(
                    totals["sampling/wallclock_sec_per_board__weighted_sum"] / num_examples
                ),
            }
        )
    return metrics


def _make_policy_table(wandb_mod, rows: list[dict[str, Any]]):
    if wandb_mod is None or not rows or not hasattr(wandb_mod, "Table"):
        return None
    columns = [
        "policy",
        "eta",
        "uses_state",
        "model_implied",
        "masked_cell_acc",
        "full_cell_acc",
        "board_acc",
        "avg_commits_per_step",
        "n_steps",
    ]
    data = [[row.get(column) for column in columns] for row in rows]
    return wandb_mod.Table(columns=columns, data=data)


def _make_sjd_runs_table(wandb_mod, rows: list[dict[str, Any]]):
    if wandb_mod is None or not rows or not hasattr(wandb_mod, "Table"):
        return None
    columns = [
        "label",
        "kind",
        "policy",
        "eta",
        "n_steps",
        "nfe_total",
        "solve_rate",
        "cell_acc_unknown",
        "board_acc_exact",
        "clue_consistency_fraction",
        "wallclock_sec_per_board",
    ]
    data = [[row.get(column) for column in columns] for row in rows]
    return wandb_mod.Table(columns=columns, data=data)


def _sjd_run_row(spec: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    prefix = str(spec["metrics_prefix"])
    sampling_nfe_total = metrics.get(f"{prefix}/sampling/nfe_total")
    sampling_wallclock = metrics.get(f"{prefix}/sampling/wallclock_sec_per_board")
    row = {
        "label": spec["label"],
        "kind": spec.get("kind", "sampler"),
        "policy": spec.get("policy"),
        "eta": spec.get("eta"),
        "n_steps": metrics.get(f"{prefix}/n_steps"),
        "nfe_total": sampling_nfe_total,
        "solve_rate": metrics.get(f"{prefix}/solve_rate"),
        "cell_acc_unknown": metrics.get(f"{prefix}/cell_acc_unknown"),
        "row_valid_fraction": metrics.get(f"{prefix}/row_valid_fraction"),
        "col_valid_fraction": metrics.get(f"{prefix}/col_valid_fraction"),
        "box_valid_fraction": metrics.get(f"{prefix}/box_valid_fraction"),
        "board_acc_exact": metrics.get(f"{prefix}/board_acc_exact"),
        "clue_consistency_fraction": metrics.get(f"{prefix}/clue_consistency_fraction"),
        "full_cell_acc": metrics.get(f"{prefix}/full_cell_acc"),
        "avg_commits_per_step": metrics.get(f"{prefix}/avg_commits_per_step"),
        "mean_masked_unknown_positions_per_step": metrics.get(
            f"{prefix}/mean_masked_unknown_positions_per_step"
        ),
        "mean_selected_top_probability": metrics.get(
            f"{prefix}/mean_selected_top_probability"
        ),
        "mean_selected_top_prob_margin": metrics.get(
            f"{prefix}/mean_selected_top_prob_margin"
        ),
        "final_masked_unknown_fraction": metrics.get(
            f"{prefix}/final_masked_unknown_fraction"
        ),
        "sampling_nfe_total": sampling_nfe_total,
        "sampling_langevin_updates_total": metrics.get(
            f"{prefix}/sampling/langevin_updates_total"
        ),
        "sampling_langevin_updates_per_step": metrics.get(
            f"{prefix}/sampling/langevin_updates_per_step"
        ),
        "sampling_gate_mean_continuous": metrics.get(
            f"{prefix}/sampling/gate_mean_continuous"
        ),
        "sampling_gate_mean_committed": metrics.get(
            f"{prefix}/sampling/gate_mean_committed"
        ),
        "sampling_frac_committed_pre_force": metrics.get(
            f"{prefix}/sampling/frac_committed_pre_force"
        ),
        "sampling_frac_committed_final": metrics.get(
            f"{prefix}/sampling/frac_committed_final"
        ),
        "sampling_wallclock_sec_per_board": sampling_wallclock,
        "wallclock_sec_per_board": sampling_wallclock,
    }
    if spec.get("kind") == "policy":
        row["nfe_total"] = None
        row["sampling_nfe_total"] = None
        row["sampling_langevin_updates_total"] = None
        row["sampling_langevin_updates_per_step"] = None
        row["sampling_gate_mean_continuous"] = None
        row["sampling_gate_mean_committed"] = None
        row["sampling_frac_committed_pre_force"] = None
        row["sampling_frac_committed_final"] = None
        row["sampling_wallclock_sec_per_board"] = None
        row["wallclock_sec_per_board"] = None
    return row


def build_sudoku_eval_logger(
    *,
    cfg: DictConfig,
    eval_cfg: DictConfig,
    task,
    model,
    wandb_mod,
    eval_every: int,
    log_at_step_zero: bool,
    sample_timesteps_override: Optional[int] = None,
):
    model_name = str(cfg.model.name).strip().lower()
    task_name = str(getattr(getattr(task, "spec", None), "name", "")).strip().lower()
    if task is None or model is None:
        raise ValueError("Sudoku evaluator requires `task` and `model`.")
    if (model_name, task_name) not in {
        ("mdlm", "mdlm_sudoku"),
        ("mdm", "mdm_sudoku_inpaint"),
        ("sjd", "sjd_sudoku_inpaint"),
    }:
        raise ValueError(
            "Sudoku evaluator only supports board-level Sudoku inpaint tasks: "
            "('mdlm', 'mdlm_sudoku'), ('mdm', 'mdm_sudoku_inpaint'), "
            "or ('sjd', 'sjd_sudoku_inpaint')."
        )

    prefix = str(eval_cfg.get("prefix", "eval"))
    verbose = bool(eval_cfg.get("verbose", False))
    progress_every_batches = int(eval_cfg.get("sudoku_progress_every_batches", 20))
    eval_seed_offset = int(eval_cfg.get("sudoku_eval_seed_offset", 1776))
    sample_seed_offset = int(eval_cfg.get("sudoku_sample_seed_offset", 314159))
    fold_in_step = bool(eval_cfg.get("sudoku_eval_fold_in_step", False))
    num_batches_default = int(eval_cfg.get("sudoku_num_batches", 64))
    num_batches_force = int(eval_cfg.get("sudoku_num_batches_force", -1))
    num_batches_per_mode = int(eval_cfg.get("sudoku_num_batches_per_sampler", num_batches_default))
    checkpoint_source = str(eval_cfg.get("checkpoint_source", "live")).strip().lower()
    log_policy_table = bool(eval_cfg.get("sudoku_log_policy_table", False))
    # Per-sampler `sampling/*` diagnostic metrics (NFE, commit/unstick counts,
    # gate means, wallclock) are useful when debugging samplers but add noise
    # to routine training dashboards. The CSV mirror is unaffected — diagnostics
    # are always written to sudoku_sjd_progress.csv regardless of this flag.
    log_diagnostics_to_wandb = bool(eval_cfg.get("sudoku_log_diagnostics_to_wandb", False))
    write_progress_csv = bool(eval_cfg.get("sudoku_write_progress_csv", False))
    progress_csv_path = eval_cfg.get("sudoku_progress_csv_path", None)
    write_latest_csv = bool(eval_cfg.get("sudoku_write_latest_csv", False))
    latest_csv_path = eval_cfg.get("sudoku_latest_csv_path", None)

    use_sjd_sampler_profiles = False
    table_builder = _make_policy_table
    if model_name in {"mdm", "mdlm"}:
        sampler_specs, primary_label, run_all_modes = _resolve_sudoku_sampler_specs(
            cfg=cfg,
            eval_cfg=eval_cfg,
            prefix=prefix,
        )
        if sample_timesteps_override is not None:
            for spec in sampler_specs:
                spec["n_steps"] = int(sample_timesteps_override)

        if model_name == "mdm":
            from sticky.models.baselines.mdm.board_sampling import conditional_generate
        else:
            from sticky.models.baselines.mdlm.sudoku_sampling import conditional_generate

        sampler_eval_fns = {}
        for sampler_spec in sampler_specs:
            sampler_model = _clone_model_for_sampler(model, sampler_spec=sampler_spec)

            @jax.jit
            def _sample_conditional(params, rng, known_tokens, known_mask, *, _model=sampler_model, _steps=int(sampler_spec["n_steps"])):
                return conditional_generate(
                    rng,
                    {"params": params, "ema_params": None},
                    model=_model,
                    known_tokens=known_tokens,
                    known_token_mask=known_mask,
                    timesteps=_steps,
                    conditioning=None,
                    use_ema=False,
                    return_diagnostics=True,
                )

            sampler_eval_fns[sampler_spec["label"]] = _sample_conditional
        policy_row_fn = None
    else:
        from sticky.models.sjd.sampler import SamplerConfig
        from sticky.models.sjd import board_sampling as sjd_board_sampling
        from sticky.models.sjd import sampling as sjd_sampling

        sampler_specs, primary_label, run_all_modes = _resolve_sudoku_sjd_run_specs(
            cfg=cfg,
            eval_cfg=eval_cfg,
            prefix=prefix,
        )
        use_sjd_sampler_profiles = any(spec["kind"] == "sampler" for spec in sampler_specs)
        if sample_timesteps_override is not None:
            for spec in sampler_specs:
                spec["n_steps"] = int(sample_timesteps_override)
        policy_row_fn = _sjd_run_row
        table_builder = _make_sjd_runs_table
        sampler_eval_fns = {}

        def _build_sjd_sampler_eval_fn(sampler_spec: dict[str, Any]):
            sampler_cfg = SamplerConfig(
                T=float(sampler_spec["T"]),
                n_steps=int(sampler_spec["n_steps"]),
                sampling_grid=str(sampler_spec["sampling_grid"]),
                score_scale=float(sampler_spec["score_scale"]),
                logit_temperature=float(sampler_spec["logit_temperature"]),
                categorical_sampling_policy=str(sampler_spec["categorical_sampling_policy"]),
                hazard_mode=str(sampler_spec["hazard_mode"]),
                alloc_mode=str(sampler_spec["alloc_mode"]),
                log_ratio_clip=float(sampler_spec["log_ratio_clip"]),
                init_std=float(sampler_spec["init_std"]),
                force_classify_at_end=bool(sampler_spec["force_classify_at_end"]),
                refresh_logits_after_em_step=bool(sampler_spec["refresh_logits_after_em_step"]),
                metrics_count_nfe=bool(sampler_spec["metrics_count_nfe"]),
                tau_grid_size=int(sampler_spec.get("tau_grid_size", 32)),
            )

            @jax.jit
            def _sample_conditional(params, rng, known_tokens, known_mask):
                a_table = model.apply({"params": params}, method=model.anchor_table)
                anchors = AnchorTable(table_float=a_table)
                return sjd_sampling.conditional_generate_board(
                    rng=rng,
                    params=params,
                    model=model,
                    anchors=anchors,
                    beta=task.forward.beta,
                    hazard=task.forward.hazard,
                    jump=task.forward.jump,
                    known_tokens=known_tokens,
                    known_token_mask=known_mask,
                    cfg=sampler_cfg,
                )

            return _sample_conditional

        def _build_sjd_policy_eval_fn(policy_spec: dict[str, Any]):
            @jax.jit
            def _sample_conditional(params, rng, known_tokens, known_mask):
                a_table = model.apply({"params": params}, method=model.anchor_table)
                anchors = AnchorTable(table_float=a_table)
                return sjd_board_sampling.conditional_generate(
                    rng,
                    params=params,
                    model=model,
                    anchors=anchors,
                    beta=task.forward.beta,
                    hazard=task.forward.hazard,
                    jump=task.forward.jump,
                    known_tokens=known_tokens,
                    known_token_mask=known_mask,
                    n_steps=int(policy_spec["n_steps"]),
                    policy=str(policy_spec["policy"]),
                    sampling_grid=str(policy_spec["sampling_grid"]),
                    logit_temperature=float(policy_spec["logit_temperature"]),
                    log_ratio_clip=float(policy_spec["log_ratio_clip"]),
                    init_std=float(policy_spec["init_std"]),
                    stochastic_k=bool(policy_spec.get("stochastic_k", False)),
                    eta=float(policy_spec["eta"]),
                    blur_score=bool(policy_spec.get("blur_score", False)),
                    return_diagnostics=True,
                )

            return _sample_conditional

        for spec in sampler_specs:
            if spec["kind"] == "sampler":
                sampler_eval_fns[spec["label"]] = _build_sjd_sampler_eval_fn(spec)
            else:
                if (
                    bool(spec.get("blur_score", False))
                    and getattr(task.forward.jump, "blur_kernel", None) is None
                ):
                    print(
                        f"[eval:sudoku] WARNING: run {spec['label']!r} requests "
                        "blur_score but no blur kernel is attached to the forward "
                        "jump; the standard (W=I) score is used, so this column "
                        "will duplicate the unblurred one.",
                        flush=True,
                    )
                sampler_eval_fns[spec["label"]] = _build_sjd_policy_eval_fn(spec)

    def _run_eval(
        step_i: int,
        params_for_sampling,
        *,
        num_batches: int,
        sampler_spec: dict[str, Any],
    ) -> tuple[Dict[str, float], dict[str, Any] | None]:
        eval_seed = int(cfg.training.seed) + eval_seed_offset
        if fold_in_step:
            eval_seed += int(step_i)

        eval_iter = make_sudoku_board_iterator(
            split="test",
            batch_size=int(task.eval_batch_size),
            seed=int(eval_seed) + 1,
            data_dir=task.data_dir,
            train_file=task.train_file,
            test_file=task.test_file,
            shuffle=False,
            repeat=False,
            drop_remainder=False,
            mmap=bool(task.mmap),
            max_examples=int(task.max_test_examples),
            auto_download=bool(task.auto_download),
            download_timeout_sec=int(task.download_timeout_sec),
            download_retries=int(task.download_retries),
        )

        totals = _base_totals()
        base_rng = jax.random.fold_in(make_rng(int(cfg.training.seed) + sample_seed_offset), int(step_i))

        while True:
            if num_batches > 0 and totals["num_batches"] >= num_batches:
                break
            try:
                batch = next(eval_iter)
            except StopIteration:
                break

            solution_board = np.asarray(batch["solution_board"], dtype=np.int32)
            clue_board = np.asarray(batch["clue_board"], dtype=np.int32)
            clue_mask = np.asarray(batch["clue_mask"], dtype=np.bool_)

            rng_batch = jax.random.fold_in(base_rng, totals["num_batches"])
            start_time = time.perf_counter()
            pred_board, diag = sampler_eval_fns[sampler_spec["label"]](
                params_for_sampling,
                rng_batch,
                clue_board,
                clue_mask,
            )
            pred_board = jax.block_until_ready(pred_board)
            elapsed = time.perf_counter() - start_time
            diag = jax.device_get(diag)
            _accumulate_board_sudoku_diagnostics(totals, diag)
            batch_examples = int(solution_board.shape[0])
            if sampler_spec.get("kind") == "sampler":
                _accumulate_sjd_sampler_diagnostics(
                    totals,
                    diag,
                    batch_examples=batch_examples,
                )
                totals["sampling/wallclock_sec_per_board__weighted_sum"] += elapsed
            pred_board = np.asarray(pred_board, dtype=np.int32)
            counts = _evaluate_board_batch_counts(
                pred_board=pred_board,
                solution_board=solution_board,
                clue_board=clue_board,
                clue_mask=clue_mask,
            )
            for key, value in counts.items():
                totals[key] += int(value)
            totals["num_batches"] += 1

            if verbose and (totals["num_batches"] % max(1, progress_every_batches) == 0):
                print(
                    f"[eval:sudoku] step={step_i} batches={totals['num_batches']} "
                    f"examples={totals['num_examples']}",
                    flush=True,
                )

        if totals["num_examples"] <= 0:
            raise RuntimeError("Sudoku evaluation consumed 0 examples.")

        metrics = _finalize_metrics(
            totals=totals,
            metric_prefix=str(sampler_spec["metrics_prefix"]),
            n_steps=int(sampler_spec["n_steps"]),
            checkpoint_source=checkpoint_source,
            include_sjd_sampler_metrics=(
                bool(sampler_spec.get("kind") == "sampler")
                and log_diagnostics_to_wandb
            ),
        )
        row = policy_row_fn(sampler_spec, metrics) if policy_row_fn is not None else None
        return metrics, row

    def maybe_eval(
        step_i: int,
        params_for_sampling,
        *,
        force_fid: bool = False,
        force_is: bool = False,
    ) -> Dict[str, float]:
        force_eval = bool(force_fid) or bool(force_is)
        if not force_eval and not _should_run_eval(
            step_i=step_i,
            every=int(eval_every),
            log_at_step_zero=bool(log_at_step_zero),
        ):
            return {}

        num_batches = num_batches_default
        if force_eval and num_batches_force != 0:
            num_batches = num_batches_force
        if verbose:
            scope = "forced" if force_eval else "scheduled"
            batch_scope = "all" if num_batches <= 0 else str(num_batches)
            print(
                f"[eval:sudoku] Running {scope} eval at step {step_i} "
                f"for {batch_scope} batch(es).",
                flush=True,
            )

        per_mode_batches = int(num_batches_per_mode) if (run_all_modes and not force_eval) else int(num_batches)
        if force_eval and num_batches_force != 0:
            per_mode_batches = int(num_batches)

        metrics: Dict[str, float] = {}
        policy_rows: list[dict[str, Any]] = []
        for sampler_spec in sampler_specs:
            sampler_metrics, row = _run_eval(
                step_i,
                params_for_sampling,
                num_batches=int(per_mode_batches),
                sampler_spec=sampler_spec,
            )
            metrics.update(sampler_metrics)
            if row is not None:
                policy_rows.append(row)

        maybe_eval.sudoku_policy_table_rows = policy_rows

        payload: dict[str, Any] = {}
        if metrics:
            payload.update(metrics)

        if wandb_mod is not None and payload:
            if log_policy_table and policy_rows:
                table = table_builder(wandb_mod, policy_rows)
                if table is not None:
                    payload[f"{prefix}/policy_table"] = table
            wandb_mod.log(payload, step=step_i)

        if model_name == "sjd" and policy_rows:
            write_sudoku_sjd_progress_artifacts(
                rows=policy_rows,
                step_i=step_i,
                progress_path_like=None if progress_csv_path is None else str(progress_csv_path),
                latest_path_like=None if latest_csv_path is None else str(latest_csv_path),
                metrics_dir_rel=str(cfg.training.get("metrics_dir", "metrics")),
                base_dir=get_hydra_output_dir(),
                write_progress=write_progress_csv,
                write_latest=write_latest_csv,
            )

        if metrics and (run_all_modes or verbose):
            primary_spec = next(
                (spec for spec in sampler_specs if spec["label"] == primary_label),
                sampler_specs[0],
            )
            primary_prefix = str(primary_spec["metrics_prefix"])
            summary_solve = metrics.get(f"{primary_prefix}/solve_rate")
            summary_board = metrics.get(f"{primary_prefix}/board_acc_exact")
            if summary_solve is not None and summary_board is not None:
                print(
                    f"[eval:sudoku] step={step_i} primary={primary_label} "
                    f"solve_rate={float(summary_solve):.4f} "
                    f"board_acc_exact={float(summary_board):.4f}",
                    flush=True,
                )

        return metrics

    maybe_eval.sudoku_policy_table_rows = []
    return maybe_eval
