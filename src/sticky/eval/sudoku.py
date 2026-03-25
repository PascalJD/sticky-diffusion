from __future__ import annotations

from typing import Dict, Optional

import hydra
import jax
import numpy as np
from omegaconf import DictConfig

from sticky.data.sudoku import make_sudoku_iterator
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

    row = triples[:, :, 0]
    col = triples[:, :, 1]
    val = triples[:, :, 2]

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
    if model_name not in {"sjd", "mdlm"}:
        raise ValueError("Sudoku evaluator currently supports model.name in {'sjd', 'mdlm'}.")
    if task is None or model is None:
        raise ValueError("Sudoku evaluator requires `task` and `model`.")

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
    else:
        n_steps = (
            int(sample_timesteps_override)
            if sample_timesteps_override is not None
            else int(cfg.sampler.get("n_steps", cfg.model.get("timesteps", 50)))
        )

    prefix = str(eval_cfg.get("prefix", "eval"))
    verbose = bool(eval_cfg.get("verbose", False))
    progress_every_batches = int(eval_cfg.get("sudoku_progress_every_batches", 20))
    eval_seed_offset = int(eval_cfg.get("sudoku_eval_seed_offset", 1776))
    sample_seed_offset = int(eval_cfg.get("sudoku_sample_seed_offset", 314159))
    fold_in_step = bool(eval_cfg.get("sudoku_eval_fold_in_step", False))
    num_batches_default = int(eval_cfg.get("sudoku_num_batches", 64))
    num_batches_force = int(eval_cfg.get("sudoku_num_batches_force", -1))

    shape = tuple(task.spec.data_shape)

    if model_name == "sjd":
        @jax.jit
        def _sample_conditional(params, rng, known_idx, known_mask):
            a_table = model.apply({"params": params}, method=model.anchor_table)
            anchors = AnchorTable(table_float=a_table)
            out = sjd_sampling.simple_generate(
                rng=rng,
                params=params,
                model=model,
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
    else:
        from sticky.models.mdlm.sudoku_sampling import conditional_generate

        @jax.jit
        def _sample_conditional(params, rng, known_idx, known_mask):
            return conditional_generate(
                rng,
                {"params": params, "ema_params": None},
                model=model,
                known_tokens=known_idx,
                known_token_mask=known_mask,
                timesteps=n_steps,
                conditioning=None,
                use_ema=False,
                return_diagnostics=True,
            )

    def _run_eval(step_i: int, params_for_sampling, *, num_batches: int) -> Dict[str, float]:
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
            "valid_complete": 0,
            "board_exact": 0,
            "strict_complete": 0,
            "given_token_correct": 0,
            "given_token_total": 0,
            "num_batches": 0,
            "example_step_count": 0.0,
            "masked_unknown_total_across_steps": 0.0,
            "selected_count_total_across_steps": 0.0,
            "selected_margin_sum_total": 0.0,
            "selected_margin_count_total": 0.0,
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

            known_token_len = (3 * start_index)[:, None]
            token_pos = np.arange(243, dtype=np.int32)[None, :]
            known_mask = token_pos < known_token_len
            known_idx = np.where(known_mask, input_seq, 0).astype(np.int32)

            rng_batch = jax.random.fold_in(base_rng, totals["num_batches"])
            sample_out = _sample_conditional(
                params_for_sampling,
                rng_batch,
                known_idx,
                known_mask,
            )
            if model_name == "mdlm":
                pred_seq, diag = sample_out
                diag = jax.device_get(diag)
                for key in (
                    "example_step_count",
                    "masked_unknown_total_across_steps",
                    "selected_count_total_across_steps",
                    "selected_margin_sum_total",
                    "selected_margin_count_total",
                    "unknown_token_total",
                    "final_masked_unknown_total",
                ):
                    totals[key] += float(diag[key])
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
        step_den = max(float(totals["example_step_count"]), 1.0)
        margin_den = max(float(totals["selected_margin_count_total"]), 1.0)
        unknown_den = max(float(totals["unknown_token_total"]), 1.0)
        mean_masked_unknown_per_step = (
            float(totals["masked_unknown_total_across_steps"]) / step_den
        )
        mean_k_selected_per_step = (
            float(totals["selected_count_total_across_steps"]) / step_den
        )
        mean_selected_top_prob_margin = (
            float(totals["selected_margin_sum_total"]) / margin_den
        )
        final_masked_unknown_fraction = (
            float(totals["final_masked_unknown_total"]) / unknown_den
        )
        checkpoint_source = str(eval_cfg.get("checkpoint_source", "live")).strip().lower()

        metrics = {
            f"{prefix}/acc": float(acc),
            f"{prefix}/acc_complete_puzzle": float(complete),
            f"{prefix}/acc_board_exact": float(board_exact),
            f"{prefix}/acc_solve_strict": float(strict_solve),
            f"{prefix}/given_token_acc": float(given_token_acc),
            f"{prefix}/solve_rate": float(strict_solve),
            f"{prefix}/mean_masked_unknown_positions_per_step": float(
                mean_masked_unknown_per_step
            ),
            f"{prefix}/mean_k_selected_per_step": float(mean_k_selected_per_step),
            f"{prefix}/mean_selected_top_prob_margin": float(
                mean_selected_top_prob_margin
            ),
            f"{prefix}/final_masked_unknown_fraction": float(
                final_masked_unknown_fraction
            ),
            f"{prefix}/checkpoint_is_live": float(checkpoint_source == "live"),
            f"{prefix}/checkpoint_is_latest": float(
                checkpoint_source in {"latest", "periodic", "root"}
            ),
            f"{prefix}/checkpoint_is_best": float(checkpoint_source == "best"),
            f"{prefix}/checkpoint_is_final": float(checkpoint_source == "final"),
            f"{prefix}/num_examples": float(totals["num_examples"]),
            f"{prefix}/num_pred_cells": float(totals["total_pred"]),
            f"{prefix}/num_batches": float(totals["num_batches"]),
            f"{prefix}/sampler_steps": float(n_steps),
        }
        if wandb_mod is not None:
            wandb_mod.log(metrics, step=step_i)
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
        return _run_eval(step_i, params_for_sampling, num_batches=int(num_batches))

    return maybe_eval
