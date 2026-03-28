from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from sticky.core.config_paths import config_root
from sticky.data.sudoku import (
    SUDOKU_PACKED_SEQ_LEN,
    SUDOKU_TRIPLET_SEQ_LEN,
    make_sudoku_iterator,
    pack_sudoku_seq2seq,
    packed_sudoku_positions,
)
from sticky.models.sjd.anchors import AnchorTable
from sticky.models.sjd.sampler import SamplerConfig
from sticky.models.sjd import sampling as sjd_sampling
from sticky.rng import make_rng


def _should_run_eval(*, step_i: int, every: int, log_at_step_zero: bool) -> bool:
    if every <= 0:
        return False
    if step_i == 0:
        return bool(log_at_step_zero)
    return (step_i % every) == 0


def valid_solution(output_seq: np.ndarray) -> bool:
    """Check if a `(243,)` Sudoku triplet sequence is a valid solved board."""
    seq = np.asarray(output_seq, dtype=np.int32).reshape(81, 3)

    rows = np.zeros((9, 9), dtype=np.int32)
    cols = np.zeros((9, 9), dtype=np.int32)
    boxes = np.zeros((9, 9), dtype=np.int32)

    for j in range(81):
        row_num = int(seq[j, 0])
        col_num = int(seq[j, 1])
        val = int(seq[j, 2])
        if row_num < 0 or row_num >= 9:
            return False
        if col_num < 0 or col_num >= 9:
            return False
        if val < 1 or val > 9:
            return False

        rows[row_num, val - 1] += 1
        cols[col_num, val - 1] += 1
        boxes[3 * (row_num // 3) + (col_num // 3), val - 1] += 1

    return bool(np.all(rows == 1) and np.all(cols == 1) and np.all(boxes == 1))


def _board_from_sequence(seq: np.ndarray) -> np.ndarray:
    triples = np.asarray(seq, dtype=np.int32).reshape(81, 3)
    board = np.zeros((81,), dtype=np.int32)

    for j in range(81):
        r = int(triples[j, 0])
        c = int(triples[j, 1])
        v = int(triples[j, 2])
        if (0 <= r < 9) and (0 <= c < 9) and (1 <= v <= 9):
            board[r * 9 + c] = v
    return board


def _safe_ratio(num: int, den: int) -> float:
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
        label = str(entry_label if entry_label is not None else entry.get("label", spec["method"])).strip()
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


def _prepare_conditional_inputs(
    *,
    model_name: str,
    input_seq: np.ndarray,
    start_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "mdm":
        packed = pack_sudoku_seq2seq(
            triplet_seq=input_seq,
            start_index=start_index,
        )
        known_mask = np.asarray(packed["prompt_mask"], dtype=np.bool_)
        known_tokens = np.where(
            known_mask,
            np.asarray(packed["packed_seq"], dtype=np.int32),
            0,
        ).astype(np.int32)
        return known_tokens, known_mask

    start_index = np.asarray(start_index, dtype=np.int32).reshape(-1)
    known_token_len = (3 * start_index)[:, None]
    token_pos = np.arange(SUDOKU_TRIPLET_SEQ_LEN, dtype=np.int32)[None, :]
    known_mask = token_pos < known_token_len
    known_tokens = np.where(known_mask, input_seq, 0).astype(np.int32)
    return known_tokens, known_mask


def _decode_mdm_prediction_to_triplets(
    *,
    packed_seq: np.ndarray,
    start_index: np.ndarray,
) -> np.ndarray:
    packed_seq = np.asarray(packed_seq, dtype=np.int32)
    if packed_seq.ndim != 2 or packed_seq.shape[1] != SUDOKU_PACKED_SEQ_LEN:
        raise ValueError(
            "MDM Sudoku evaluator expects packed decoded sequences with shape "
            f"[B, {SUDOKU_PACKED_SEQ_LEN}], got {packed_seq.shape}."
        )

    positions = packed_sudoku_positions(start_index=np.asarray(start_index, dtype=np.int32))
    sep_index = np.asarray(positions["sep_index"], dtype=np.int32).reshape(-1)
    response_start = np.asarray(
        positions["response_start_index"], dtype=np.int32
    ).reshape(-1)
    eos_index = np.asarray(positions["eos_index"], dtype=np.int32).reshape(-1)

    triplet_seq = np.empty((packed_seq.shape[0], SUDOKU_TRIPLET_SEQ_LEN), dtype=np.int32)
    for i in range(packed_seq.shape[0]):
        prompt_len = int(sep_index[i])
        response = packed_seq[i, int(response_start[i]) : int(eos_index[i])]
        if response.shape[0] != SUDOKU_TRIPLET_SEQ_LEN - prompt_len:
            raise ValueError(
                "Invalid packed MDM decode length after removing [SEP]/[EOS]: "
                f"expected {SUDOKU_TRIPLET_SEQ_LEN - prompt_len}, got {response.shape[0]}."
            )
        triplet_seq[i, :prompt_len] = packed_seq[i, :prompt_len]
        triplet_seq[i, prompt_len:] = response
    return triplet_seq


def _accumulate_sudoku_diagnostics(
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
        "selected_margin_sum_total",
        "selected_margin_count_total",
        "selected_row_total_across_steps",
        "selected_col_total_across_steps",
        "selected_value_total_across_steps",
        "selected_eos_total_across_steps",
        "unknown_token_total",
        "final_masked_unknown_total",
    ):
        if key in diag:
            totals[key] += float(diag[key])


def _evaluate_batch_counts(
    *,
    pred_seq: np.ndarray,
    puzzle_sol: np.ndarray,
    start_index: np.ndarray,
    input_seq: np.ndarray,
) -> Dict[str, int]:
    pred_seq = np.asarray(pred_seq, dtype=np.int32)
    puzzle_sol = np.asarray(puzzle_sol, dtype=np.int32)
    start_index = np.asarray(start_index, dtype=np.int32).reshape(-1)
    input_seq = np.asarray(input_seq, dtype=np.int32)

    batch_size = int(pred_seq.shape[0])
    triples = pred_seq.reshape(batch_size, 81, 3)
    target = input_seq.reshape(batch_size, 81, 3)

    row = triples[:, :, 0]
    col = triples[:, :, 1]
    val = triples[:, :, 2]
    target_row = target[:, :, 0]
    target_col = target[:, :, 1]
    target_val = target[:, :, 2]

    valid_rcv = (
        (row >= 0)
        & (row < 9)
        & (col >= 0)
        & (col < 9)
        & (val >= 1)
        & (val <= 9)
    )
    flat_idx = np.clip(row, 0, 8) * 9 + np.clip(col, 0, 8)
    gt_val = np.take_along_axis(puzzle_sol, flat_idx, axis=1)

    pred_cell_mask = np.arange(81, dtype=np.int32)[None, :] >= start_index[:, None]
    success_pred = int(np.sum(valid_rcv & pred_cell_mask & (gt_val == val)))
    total_pred = int(np.sum(pred_cell_mask))
    row_token_correct = int(np.sum(pred_cell_mask & (row == target_row)))
    col_token_correct = int(np.sum(pred_cell_mask & (col == target_col)))
    value_token_correct = int(np.sum(pred_cell_mask & (val == target_val)))
    rowcol_correct_mask = pred_cell_mask & (row == target_row) & (col == target_col)
    value_given_correct_rowcol = int(np.sum(rowcol_correct_mask & (val == target_val)))
    rowcol_correct_total = int(np.sum(rowcol_correct_mask))

    valid_rc = (
        (row >= 0)
        & (row < 9)
        & (col >= 0)
        & (col < 9)
    )
    duplicate_coordinate_total = 0
    for i in range(batch_size):
        valid_pred_coords = pred_cell_mask[i] & valid_rc[i]
        coord_ids = (row[i, valid_pred_coords] * 9) + col[i, valid_pred_coords]
        duplicate_coordinate_total += int(coord_ids.size - np.unique(coord_ids).size)

    known_token_len = (3 * start_index)[:, None]
    known_token_mask = np.arange(243, dtype=np.int32)[None, :] < known_token_len
    given_token_correct = int(np.sum((pred_seq == input_seq) & known_token_mask))
    given_token_total = int(np.sum(known_token_mask))

    valid_complete = 0
    board_exact = 0
    strict_complete = 0
    for i in range(batch_size):
        seq_i = pred_seq[i]
        is_valid = valid_solution(seq_i)
        if is_valid:
            valid_complete += 1
        board_pred = _board_from_sequence(seq_i)
        is_board_exact = np.array_equal(board_pred, puzzle_sol[i])
        if is_board_exact:
            board_exact += 1
        if is_valid and is_board_exact:
            strict_complete += 1

    return {
        "num_examples": batch_size,
        "success_pred": success_pred,
        "total_pred": total_pred,
        "row_token_correct": row_token_correct,
        "col_token_correct": col_token_correct,
        "value_token_correct": value_token_correct,
        "value_given_correct_rowcol": value_given_correct_rowcol,
        "rowcol_correct_total": rowcol_correct_total,
        "duplicate_coordinate_total": duplicate_coordinate_total,
        "valid_complete": valid_complete,
        "board_exact": board_exact,
        "strict_complete": strict_complete,
        "given_token_correct": given_token_correct,
        "given_token_total": given_token_total,
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
    model_name = str(cfg.model.name)
    if model_name not in {"sjd", "mdlm", "mdm"}:
        raise ValueError(
            "Sudoku evaluator currently supports model.name in {'sjd', 'mdlm', 'mdm'}."
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

    if model_name == "sjd":
        beta = getattr(task, "beta", None)
        if beta is None:
            beta = hydra.utils.instantiate(cfg.forward.beta)

        hazard = getattr(task, "hazard", None)
        if hazard is None:
            hazard_cfg = cfg.forward.get("hazard", None)
            hazard = hydra.utils.instantiate(hazard_cfg, beta=beta) if hazard_cfg is not None else None

        jump = getattr(task, "jump", None)
        if jump is None:
            jump_cfg = cfg.forward.get("jump", None)
            jump = hydra.utils.instantiate(jump_cfg, beta=beta) if jump_cfg is not None else None

        if hazard is None or jump is None:
            raise ValueError("Sudoku evaluator requires both hazard and jump schedules.")

        n_steps = (
            int(sample_timesteps_override)
            if sample_timesteps_override is not None
            else int(cfg.sampler.get("n_steps", 250))
        )
        sampler_cfg = SamplerConfig(
            T=float(cfg.sampler.get("T", 1.0)),
            n_steps=n_steps,
            sampling_grid=str(cfg.sampler.get("sampling_grid", "uniform")),
            score_scale=float(cfg.sampler.get("score_scale", 1.0)),
            logit_temperature=float(
                cfg.sampler.get(
                    "logit_temperature",
                    cfg.sampler.get("temperature", 1.0),
                )
            ),
            categorical_sampling_policy=str(
                cfg.sampler.get("categorical_sampling_policy", "legacy_low")
            ),
            hazard_mode=str(cfg.sampler.get("hazard_mode", "plugin")),
            alloc_mode=str(cfg.sampler.get("alloc_mode", "sample")),
            intensity_mode=str(cfg.sampler.get("intensity_mode", "full")),
            log_ratio_clip=float(cfg.sampler.get("log_ratio_clip", 10.0)),
            intensity_chunk_size=int(cfg.sampler.get("intensity_chunk_size", 256)),
            init_std=float(cfg.sampler.get("init_std", 1.0)),
            force_classify_at_end=bool(cfg.sampler.get("force_classify_at_end", True)),
            refresh_logits_after_em_step=bool(
                cfg.sampler.get("refresh_logits_after_em_step", False)
            ),
        )
        sampler_specs = [
            {
                "label": "default",
                "metrics_prefix": prefix,
                "n_steps": n_steps,
            }
        ]
        primary_sampler_label = "default"
        run_all_sampler_modes = False
    else:
        default_n_steps = (
            int(sample_timesteps_override)
            if sample_timesteps_override is not None
            else int(cfg.sampler.get("n_steps", cfg.model.get("timesteps", 50)))
        )
        sampler_specs, primary_sampler_label, run_all_sampler_modes = _resolve_sudoku_sampler_specs(
            cfg=cfg,
            eval_cfg=eval_cfg,
            prefix=prefix,
        )
        if sample_timesteps_override is not None:
            for sampler_spec in sampler_specs:
                sampler_spec["n_steps"] = int(sample_timesteps_override)

    shape = tuple(task.spec.data_shape)

    def _make_sample_conditional_fn(
        *,
        model_for_eval,
        sampler_steps: int,
    ):
        if model_name == "sjd":
            @jax.jit
            def _sample_conditional(params, rng, known_idx, known_mask):
                a_table = model_for_eval.apply({"params": params}, method=model_for_eval.anchor_table)
                anchors = AnchorTable(table_float=a_table)
                out = sjd_sampling.simple_generate(
                    rng=rng,
                    params=params,
                    model=model_for_eval,
                    anchors=anchors,
                    beta=beta,
                    hazard=hazard,
                    jump=jump,
                    cfg=sampler_cfg,
                    batch_size=int(known_idx.shape[0]),
                    shape=shape,
                    known_idx=known_idx,
                    known_mask=known_mask,
                )
                return out.k_filled

            return _sample_conditional

        if model_name == "mdm":
            from sticky.models.baselines.mdm import conditional_generate
        else:
            from sticky.models.baselines.mdlm.sudoku_sampling import conditional_generate

        @jax.jit
        def _sample_conditional(params, rng, known_idx, known_mask):
            return conditional_generate(
                rng,
                {"params": params, "ema_params": None},
                model=model_for_eval,
                known_tokens=known_idx,
                known_token_mask=known_mask,
                timesteps=int(sampler_steps),
                conditioning=None,
                use_ema=False,
                return_diagnostics=True,
            )

        return _sample_conditional

    sampler_eval_fns: dict[str, Any] = {}
    for sampler_spec in sampler_specs:
        sampler_model = (
            model
            if model_name == "sjd"
            else _clone_model_for_sampler(model, sampler_spec=sampler_spec)
        )
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

        eval_iter = make_sudoku_iterator(
            split="test",
            batch_size=int(task.eval_batch_size),
            seed=int(eval_seed) + 1,
            data_dir=task.data_dir,
            train_file=task.train_file,
            test_file=task.test_file,
            shuffle=False,
            repeat=False,
            drop_remainder=False,
            seq_order=str(task.seq_order),
            mmap=bool(task.mmap),
            max_examples=int(task.max_test_examples),
            auto_download=bool(task.auto_download),
            download_timeout_sec=int(task.download_timeout_sec),
            download_retries=int(task.download_retries),
        )

        totals = {
            "num_examples": 0,
            "success_pred": 0,
            "total_pred": 0,
            "row_token_correct": 0,
            "col_token_correct": 0,
            "value_token_correct": 0,
            "value_given_correct_rowcol": 0,
            "rowcol_correct_total": 0,
            "duplicate_coordinate_total": 0,
            "valid_complete": 0,
            "board_exact": 0,
            "strict_complete": 0,
            "given_token_correct": 0,
            "given_token_total": 0,
            "num_batches": 0,
            "example_step_count": 0.0,
            "masked_unknown_total_across_steps": 0.0,
            "selected_count_total_across_steps": 0.0,
            "selected_top_probability_sum_total": 0.0,
            "selected_top_probability_count_total": 0.0,
            "selected_top_prob_margin_sum_total": 0.0,
            "selected_top_prob_margin_count_total": 0.0,
            "selected_margin_sum_total": 0.0,
            "selected_margin_count_total": 0.0,
            "selected_row_total_across_steps": 0.0,
            "selected_col_total_across_steps": 0.0,
            "selected_value_total_across_steps": 0.0,
            "selected_eos_total_across_steps": 0.0,
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

            input_seq = np.asarray(batch["image"], dtype=np.int32)
            puzzle_sol = np.asarray(batch["puzzle"], dtype=np.int32)
            start_index = np.asarray(batch["start_index"], dtype=np.int32).reshape(-1)

            if input_seq.ndim != 2 or input_seq.shape[1] != 243:
                raise ValueError(
                    "Sudoku evaluator expects batch['image'] with shape [B, 243], "
                    f"got {input_seq.shape}."
                )
            if puzzle_sol.ndim != 2 or puzzle_sol.shape[1] != 81:
                raise ValueError(
                    "Sudoku evaluator expects batch['puzzle'] with shape [B, 81], "
                    f"got {puzzle_sol.shape}."
                )

            known_idx, known_mask = _prepare_conditional_inputs(
                model_name=model_name,
                input_seq=input_seq,
                start_index=start_index,
            )

            rng_batch = jax.random.fold_in(base_rng, totals["num_batches"])
            sample_out = sampler_eval_fns[sampler_spec["label"]](
                params_for_sampling,
                rng_batch,
                known_idx,
                known_mask,
            )
            if model_name in {"mdlm", "mdm"}:
                pred_tokens, diag = sample_out
                _accumulate_sudoku_diagnostics(totals, jax.device_get(diag))
                pred_tokens = np.asarray(jax.device_get(pred_tokens), dtype=np.int32)
                pred_seq = (
                    _decode_mdm_prediction_to_triplets(
                        packed_seq=pred_tokens,
                        start_index=start_index,
                    )
                    if model_name == "mdm"
                    else pred_tokens
                )
            else:
                pred_seq = sample_out
            pred_seq = np.asarray(jax.device_get(pred_seq), dtype=np.int32)

            counts = _evaluate_batch_counts(
                pred_seq=pred_seq,
                puzzle_sol=puzzle_sol,
                start_index=start_index,
                input_seq=input_seq,
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

        acc = _safe_ratio(totals["success_pred"], totals["total_pred"])
        complete = _safe_ratio(totals["valid_complete"], totals["num_examples"])
        board_exact = _safe_ratio(totals["board_exact"], totals["num_examples"])
        strict_solve = _safe_ratio(totals["strict_complete"], totals["num_examples"])
        given_token_acc = _safe_ratio(
            totals["given_token_correct"],
            totals["given_token_total"],
        )
        row_token_acc = _safe_ratio(totals["row_token_correct"], totals["total_pred"])
        col_token_acc = _safe_ratio(totals["col_token_correct"], totals["total_pred"])
        value_token_acc = _safe_ratio(totals["value_token_correct"], totals["total_pred"])
        value_acc_given_correct_rowcol = _safe_ratio(
            totals["value_given_correct_rowcol"],
            totals["rowcol_correct_total"],
        )
        duplicate_coordinate_rate = _safe_ratio(
            totals["duplicate_coordinate_total"],
            totals["total_pred"],
        )
        step_den = max(float(totals["example_step_count"]), 1.0)
        top_prob_den = max(float(totals["selected_top_probability_count_total"]), 1.0)
        top_margin_den = max(float(totals["selected_top_prob_margin_count_total"]), 1.0)
        margin_den = max(float(totals["selected_margin_count_total"]), 1.0)
        unknown_den = max(float(totals["unknown_token_total"]), 1.0)
        selected_den = max(float(totals["selected_count_total_across_steps"]), 1.0)
        mean_masked_unknown_per_step = (
            float(totals["masked_unknown_total_across_steps"]) / step_den
        )
        mean_k_selected_per_step = (
            float(totals["selected_count_total_across_steps"]) / step_den
        )
        mean_selected_top_probability = (
            float(totals["selected_top_probability_sum_total"]) / top_prob_den
        )
        mean_selected_top_prob_margin_v2 = (
            float(totals["selected_top_prob_margin_sum_total"]) / top_margin_den
        )
        mean_selected_top_prob_margin = (
            float(totals["selected_margin_sum_total"]) / margin_den
        )
        final_masked_unknown_fraction = (
            float(totals["final_masked_unknown_total"]) / unknown_den
        )
        selected_row_fraction = (
            float(totals["selected_row_total_across_steps"]) / selected_den
        )
        selected_col_fraction = (
            float(totals["selected_col_total_across_steps"]) / selected_den
        )
        selected_value_fraction = (
            float(totals["selected_value_total_across_steps"]) / selected_den
        )
        selected_eos_fraction = (
            float(totals["selected_eos_total_across_steps"]) / selected_den
        )
        metric_prefix = str(sampler_spec["metrics_prefix"])

        metrics = {
            f"{metric_prefix}/acc": float(acc),
            f"{metric_prefix}/acc_complete_puzzle": float(complete),
            f"{metric_prefix}/acc_board_exact": float(board_exact),
            f"{metric_prefix}/acc_solve_strict": float(strict_solve),
            f"{metric_prefix}/given_token_acc": float(given_token_acc),
            f"{metric_prefix}/row_token_acc": float(row_token_acc),
            f"{metric_prefix}/col_token_acc": float(col_token_acc),
            f"{metric_prefix}/value_token_acc": float(value_token_acc),
            f"{metric_prefix}/value_acc_given_correct_rowcol": float(
                value_acc_given_correct_rowcol
            ),
            f"{metric_prefix}/duplicate_coordinate_rate": float(duplicate_coordinate_rate),
            f"{metric_prefix}/solve_rate": float(strict_solve),
            f"{metric_prefix}/mean_masked_unknown_positions_per_step": float(
                mean_masked_unknown_per_step
            ),
            f"{metric_prefix}/mean_k_selected_per_step": float(mean_k_selected_per_step),
            f"{metric_prefix}/mean_selected_top_probability": float(
                mean_selected_top_probability
            ),
            f"{metric_prefix}/mean_selected_top_prob_margin": float(
                mean_selected_top_prob_margin_v2
                if float(totals["selected_top_prob_margin_count_total"]) > 0.0
                else mean_selected_top_prob_margin
            ),
            f"{metric_prefix}/mean_selected_top_prob_margin_legacy": float(
                mean_selected_top_prob_margin
            ),
            f"{metric_prefix}/selected_row_fraction": float(selected_row_fraction),
            f"{metric_prefix}/selected_col_fraction": float(selected_col_fraction),
            f"{metric_prefix}/selected_value_fraction": float(selected_value_fraction),
            f"{metric_prefix}/selected_eos_fraction": float(selected_eos_fraction),
            f"{metric_prefix}/final_masked_unknown_fraction": float(
                final_masked_unknown_fraction
            ),
            f"{metric_prefix}/checkpoint_is_live": float(checkpoint_source == "live"),
            f"{metric_prefix}/checkpoint_is_latest": float(
                checkpoint_source in {"latest", "periodic", "root"}
            ),
            f"{metric_prefix}/checkpoint_is_best": float(checkpoint_source == "best"),
            f"{metric_prefix}/checkpoint_is_final": float(checkpoint_source == "final"),
            f"{metric_prefix}/num_examples": float(totals["num_examples"]),
            f"{metric_prefix}/num_pred_cells": float(totals["total_pred"]),
            f"{metric_prefix}/num_batches": float(totals["num_batches"]),
            f"{metric_prefix}/sampler_steps": float(sampler_spec["n_steps"]),
        }
        return metrics

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
            primary_prefix = prefix
            if run_all_sampler_modes:
                primary_spec = next(
                    (
                        spec
                        for spec in sampler_specs
                        if spec["label"] == primary_sampler_label
                    ),
                    sampler_specs[0],
                )
                primary_prefix = str(primary_spec["metrics_prefix"])
            summary_acc = metrics.get(f"{primary_prefix}/acc_complete_puzzle")
            summary_strict = metrics.get(f"{primary_prefix}/acc_solve_strict")
            if summary_acc is not None and summary_strict is not None:
                print(
                    f"[eval:sudoku] step={step_i} primary={primary_sampler_label} "
                    f"acc_complete_puzzle={float(summary_acc):.4f} "
                    f"acc_solve_strict={float(summary_strict):.4f}",
                    flush=True,
                )

        return metrics

    return maybe_eval
