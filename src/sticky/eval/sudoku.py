from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from sticky.core.config_paths import config_root
from sticky.data.sudoku import make_sudoku_board_iterator
from sticky.rng import make_rng


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


def _load_sampler_group_overrides(name: str) -> dict[str, Any]:
    path = config_root() / "sampler" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown sampler config group {name!r}: {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Sampler config {name!r} did not load as a mapping.")
    raw.pop("defaults", None)
    return raw


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


def _iter_named_sudoku_sampler_entries(
    samplers_cfg: Any,
) -> list[tuple[str | None, dict[str, Any]]]:
    if samplers_cfg is None:
        return []

    container = OmegaConf.to_container(samplers_cfg, resolve=False)
    if isinstance(container, dict):
        entries: list[tuple[str | None, dict[str, Any]]] = []
        for label, entry in container.items():
            if not isinstance(entry, dict):
                raise ValueError(
                    "Each sudoku_eval_samplers entry must be a mapping of sampler options."
                )
            entries.append((str(label), dict(entry)))
        return entries

    if isinstance(container, list):
        entries = []
        for entry in container:
            if not isinstance(entry, dict):
                raise ValueError("Each sudoku_eval_samplers entry must be a mapping.")
            label = entry.get("label", None)
            entries.append((None if label is None else str(label), dict(entry)))
        return entries

    raise ValueError(
        "sudoku_eval_samplers must be either a mapping of named samplers or a list of mappings."
    )


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
    for entry_label, entry in _iter_named_sudoku_sampler_entries(
        eval_cfg.get("sudoku_eval_samplers", None)
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
    if model_name != "mdm" or task_name != "mdm_sudoku_inpaint":
        raise ValueError(
            "Sudoku evaluator only supports the board-level inpaint benchmark: "
            "task.name='mdm_sudoku_inpaint' and model.name='mdm'."
        )
    if task is None or model is None:
        raise ValueError("Sudoku evaluator requires `task` and `model`.")

    prefix = str(eval_cfg.get("prefix", "eval"))
    verbose = bool(eval_cfg.get("verbose", False))
    progress_every_batches = int(eval_cfg.get("sudoku_progress_every_batches", 20))
    eval_seed_offset = int(eval_cfg.get("sudoku_eval_seed_offset", 1776))
    sample_seed_offset = int(eval_cfg.get("sudoku_sample_seed_offset", 314159))
    fold_in_step = bool(eval_cfg.get("sudoku_eval_fold_in_step", False))
    num_batches_default = int(eval_cfg.get("sudoku_num_batches", 64))
    num_batches_force = int(eval_cfg.get("sudoku_num_batches_force", -1))
    num_batches_per_sampler = int(
        eval_cfg.get("sudoku_num_batches_per_sampler", num_batches_default)
    )
    checkpoint_source = str(eval_cfg.get("checkpoint_source", "live")).strip().lower()

    sampler_specs, primary_sampler_label, run_all_sampler_modes = _resolve_sudoku_sampler_specs(
        cfg=cfg,
        eval_cfg=eval_cfg,
        prefix=prefix,
    )
    if sample_timesteps_override is not None:
        for sampler_spec in sampler_specs:
            sampler_spec["n_steps"] = int(sample_timesteps_override)

    from sticky.models.baselines.mdm.board_sampling import conditional_generate

    def _make_sample_conditional_fn(
        *,
        model_for_eval,
        sampler_steps: int,
    ):
        @jax.jit
        def _sample_conditional(params, rng, known_tokens, known_mask):
            return conditional_generate(
                rng,
                {"params": params, "ema_params": None},
                model=model_for_eval,
                known_tokens=known_tokens,
                known_token_mask=known_mask,
                timesteps=int(sampler_steps),
                conditioning=None,
                use_ema=False,
                return_diagnostics=True,
            )

        return _sample_conditional

    sampler_eval_fns: dict[str, Any] = {}
    for sampler_spec in sampler_specs:
        sampler_model = _clone_model_for_sampler(model, sampler_spec=sampler_spec)
        sampler_eval_fns[sampler_spec["label"]] = _make_sample_conditional_fn(
            model_for_eval=sampler_model,
            sampler_steps=int(sampler_spec["n_steps"]),
        )

    def _run_eval(
        step_i: int,
        params_for_sampling,
        *,
        num_batches: int,
        sampler_spec: dict[str, Any],
    ) -> Dict[str, float]:
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

        totals = {
            "num_examples": 0,
            "unknown_cell_correct": 0,
            "unknown_cell_total": 0,
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
        }

        base_rng = jax.random.fold_in(
            make_rng(int(cfg.training.seed) + sample_seed_offset),
            int(step_i),
        )

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

            if solution_board.ndim != 2 or solution_board.shape[1] != 81:
                raise ValueError(
                    "Board Sudoku evaluator expects batch['solution_board'] with "
                    f"shape [B, 81], got {solution_board.shape}."
                )
            if clue_board.shape != solution_board.shape or clue_mask.shape != solution_board.shape:
                raise ValueError(
                    "Board Sudoku evaluator expects clue_board and clue_mask to "
                    f"match solution_board shape, got {clue_board.shape} and "
                    f"{clue_mask.shape} vs {solution_board.shape}."
                )

            rng_batch = jax.random.fold_in(base_rng, totals["num_batches"])
            pred_board, diag = sampler_eval_fns[sampler_spec["label"]](
                params_for_sampling,
                rng_batch,
                clue_board,
                clue_mask,
            )
            _accumulate_board_sudoku_diagnostics(totals, jax.device_get(diag))
            pred_board = np.asarray(jax.device_get(pred_board), dtype=np.int32)

            counts = _evaluate_board_batch_counts(
                pred_board=pred_board,
                solution_board=solution_board,
                clue_board=clue_board,
                clue_mask=clue_mask,
            )
            for key, value in counts.items():
                totals[key] += int(value)
            totals["num_batches"] += 1

            if verbose and (
                totals["num_batches"] % max(1, progress_every_batches) == 0
            ):
                print(
                    f"[eval:sudoku] step={step_i} batches={totals['num_batches']} "
                    f"examples={totals['num_examples']}",
                    flush=True,
                )

        if totals["num_examples"] <= 0:
            raise RuntimeError("Sudoku evaluation consumed 0 examples.")

        step_den = max(float(totals["example_step_count"]), 1.0)
        top_prob_den = max(float(totals["selected_top_probability_count_total"]), 1.0)
        top_margin_den = max(
            float(totals["selected_top_prob_margin_count_total"]),
            1.0,
        )
        unknown_den = max(float(totals["unknown_token_total"]), 1.0)
        metric_prefix = str(sampler_spec["metrics_prefix"])

        return {
            f"{metric_prefix}/solve_rate": float(
                _safe_ratio(totals["solve_count"], totals["num_examples"])
            ),
            f"{metric_prefix}/cell_acc_unknown": float(
                _safe_ratio(totals["unknown_cell_correct"], totals["unknown_cell_total"])
            ),
            f"{metric_prefix}/board_acc_exact": float(
                _safe_ratio(totals["board_exact"], totals["num_examples"])
            ),
            f"{metric_prefix}/row_valid_fraction": float(
                _safe_ratio(totals["row_valid_total"], totals["row_total"])
            ),
            f"{metric_prefix}/col_valid_fraction": float(
                _safe_ratio(totals["col_valid_total"], totals["col_total"])
            ),
            f"{metric_prefix}/box_valid_fraction": float(
                _safe_ratio(totals["box_valid_total"], totals["box_total"])
            ),
            f"{metric_prefix}/clue_consistency_fraction": float(
                _safe_ratio(totals["clue_consistent_total"], totals["num_examples"])
            ),
            f"{metric_prefix}/mean_masked_unknown_positions_per_step": float(
                float(totals["masked_unknown_total_across_steps"]) / step_den
            ),
            f"{metric_prefix}/mean_k_selected_per_step": float(
                float(totals["selected_count_total_across_steps"]) / step_den
            ),
            f"{metric_prefix}/mean_selected_top_probability": float(
                float(totals["selected_top_probability_sum_total"]) / top_prob_den
            ),
            f"{metric_prefix}/mean_selected_top_prob_margin": float(
                float(totals["selected_top_prob_margin_sum_total"]) / top_margin_den
            ),
            f"{metric_prefix}/final_masked_unknown_fraction": float(
                float(totals["final_masked_unknown_total"]) / unknown_den
            ),
            f"{metric_prefix}/checkpoint_is_live": float(checkpoint_source == "live"),
            f"{metric_prefix}/checkpoint_is_latest": float(
                checkpoint_source in {"latest", "periodic", "root"}
            ),
            f"{metric_prefix}/checkpoint_is_best": float(checkpoint_source == "best"),
            f"{metric_prefix}/checkpoint_is_final": float(checkpoint_source == "final"),
            f"{metric_prefix}/num_examples": float(totals["num_examples"]),
            f"{metric_prefix}/num_batches": float(totals["num_batches"]),
            f"{metric_prefix}/sampler_steps": float(sampler_spec["n_steps"]),
        }

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

        per_sampler_batches = (
            int(num_batches_per_sampler)
            if (run_all_sampler_modes and not force_eval)
            else int(num_batches)
        )
        if force_eval and num_batches_force != 0:
            per_sampler_batches = int(num_batches)

        metrics: Dict[str, float] = {}
        for sampler_spec in sampler_specs:
            metrics.update(
                _run_eval(
                    step_i,
                    params_for_sampling,
                    num_batches=int(per_sampler_batches),
                    sampler_spec=sampler_spec,
                )
            )

        if wandb_mod is not None and metrics:
            wandb_mod.log(metrics, step=step_i)

        if metrics and (run_all_sampler_modes or verbose):
            primary_spec = next(
                (
                    spec
                    for spec in sampler_specs
                    if spec["label"] == primary_sampler_label
                ),
                sampler_specs[0],
            )
            primary_prefix = str(primary_spec["metrics_prefix"])
            summary_solve = metrics.get(f"{primary_prefix}/solve_rate")
            summary_board = metrics.get(f"{primary_prefix}/board_acc_exact")
            if summary_solve is not None and summary_board is not None:
                print(
                    f"[eval:sudoku] step={step_i} primary={primary_sampler_label} "
                    f"solve_rate={float(summary_solve):.4f} "
                    f"board_acc_exact={float(summary_board):.4f}",
                    flush=True,
                )

        return metrics

    return maybe_eval
