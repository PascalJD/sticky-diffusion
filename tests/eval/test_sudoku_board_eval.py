from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pytest
from omegaconf import OmegaConf
from types import SimpleNamespace

import sticky.eval.sudoku as sudoku_eval_mod
from sticky.eval.sudoku import _evaluate_board_batch_counts, build_sudoku_eval_logger


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _solution_board() -> np.ndarray:
    return np.asarray(
        [[((row * 3 + row // 3 + col) % 9) + 1 for row in range(9) for col in range(9)]],
        dtype=np.int32,
    )


def _clue_inputs() -> tuple[np.ndarray, np.ndarray]:
    solution = _solution_board()
    clue_mask = np.zeros((1, 81), dtype=np.bool_)
    clue_mask[0, :70] = True
    clue_board = np.where(clue_mask, solution, 0).astype(np.int32)
    return clue_board, clue_mask


def test_ground_truth_solution_has_perfect_board_metrics():
    solution = _solution_board()
    clue_board, clue_mask = _clue_inputs()

    counts = _evaluate_board_batch_counts(
        pred_board=solution,
        solution_board=solution,
        clue_board=clue_board,
        clue_mask=clue_mask,
    )

    assert counts["solve_count"] == 1
    assert counts["board_exact"] == 1
    assert counts["unknown_cell_correct"] == counts["unknown_cell_total"]
    assert counts["row_valid_total"] == 9
    assert counts["col_valid_total"] == 9
    assert counts["box_valid_total"] == 9


def test_one_wrong_unknown_cell_breaks_solve_rate():
    solution = _solution_board()
    clue_board, clue_mask = _clue_inputs()
    pred = solution.copy()
    wrong_idx = int(np.flatnonzero(~clue_mask[0])[0])
    pred[0, wrong_idx] = (pred[0, wrong_idx] % 9) + 1

    counts = _evaluate_board_batch_counts(
        pred_board=pred,
        solution_board=solution,
        clue_board=clue_board,
        clue_mask=clue_mask,
    )

    assert counts["solve_count"] == 0
    assert counts["board_exact"] == 0


def test_cell_acc_unknown_counts_only_non_clue_positions():
    solution = _solution_board()
    clue_board, clue_mask = _clue_inputs()
    pred = solution.copy()
    unknown_positions = np.flatnonzero(~clue_mask[0])
    pred[0, unknown_positions[:3]] = (pred[0, unknown_positions[:3]] % 9) + 1

    counts = _evaluate_board_batch_counts(
        pred_board=pred,
        solution_board=solution,
        clue_board=clue_board,
        clue_mask=clue_mask,
    )

    assert counts["unknown_cell_total"] == int((~clue_mask).sum())
    assert counts["unknown_cell_correct"] == counts["unknown_cell_total"] - 3


@dataclass
class _FakeBoardModel:
    sampler: str = "top_prob_margin"
    sampling_grid: str = "loglinear"
    categorical_sampling_policy: str = "exact"
    timesteps: int = 4


def _fake_board_task():
    return SimpleNamespace(
        eval_batch_size=1,
        data_dir=None,
        train_file=None,
        test_file=None,
        mmap=False,
        max_test_examples=1,
        auto_download=False,
        download_timeout_sec=1,
        download_retries=0,
        spec=SimpleNamespace(name="mdm_sudoku_inpaint", data_shape=(81,)),
    )


def test_board_eval_logger_runs_all_sampler_modes(monkeypatch):
    solution = _solution_board()
    clue_board, clue_mask = _clue_inputs()
    batch = {
        "solution_board": solution,
        "clue_board": clue_board,
        "clue_mask": clue_mask,
        "image": solution,
    }

    monkeypatch.setattr(
        sudoku_eval_mod,
        "make_sudoku_board_iterator",
        lambda **kwargs: iter([batch]),
    )

    calls = []

    def _fake_conditional_generate(
        rng,
        train_state,
        *,
        model,
        known_tokens,
        known_token_mask,
        timesteps=None,
        conditioning=None,
        use_ema=True,
        return_diagnostics=False,
        sampler_override=None,
    ):
        del rng, train_state, known_tokens, known_token_mask, conditioning, use_ema, sampler_override
        calls.append((model.sampler, timesteps))
        diag = {
            "example_step_count": jnp.asarray(1.0, dtype=jnp.float32),
            "masked_unknown_total_across_steps": jnp.asarray(2.0, dtype=jnp.float32),
            "selected_count_total_across_steps": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_top_probability_sum_total": jnp.asarray(0.8, dtype=jnp.float32),
            "selected_top_probability_count_total": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_top_prob_margin_sum_total": jnp.asarray(0.4, dtype=jnp.float32),
            "selected_top_prob_margin_count_total": jnp.asarray(1.0, dtype=jnp.float32),
            "unknown_token_total": jnp.asarray(float((~clue_mask).sum()), dtype=jnp.float32),
            "final_masked_unknown_total": jnp.asarray(0.0, dtype=jnp.float32),
        }
        output = jnp.asarray(solution, dtype=jnp.int32)
        return (output, diag) if return_diagnostics else output

    monkeypatch.setattr(
        __import__("sticky.models.baselines.mdm.board_sampling", fromlist=["conditional_generate"]),
        "conditional_generate",
        _fake_conditional_generate,
    )

    cfg = OmegaConf.create(
        {
            "model": {"name": "mdm_inpaint", "timesteps": 4},
            "sampler": {
                "method": "top_prob_margin",
                "n_steps": 4,
                "sampling_grid": "loglinear",
                "categorical_sampling_policy": "exact",
            },
            "training": {"seed": 0},
        }
    )
    eval_cfg = OmegaConf.create(
        {
            "mode": "sudoku",
            "prefix": "eval",
            "verbose": False,
            "sudoku_every": 1,
            "sudoku_num_batches": 1,
            "sudoku_num_batches_force": -1,
            "sudoku_num_batches_per_sampler": 1,
            "sudoku_eval_seed_offset": 1776,
            "sudoku_sample_seed_offset": 314159,
            "sudoku_eval_fold_in_step": False,
            "sudoku_progress_every_batches": 20,
            "sudoku_run_all_sampler_modes": True,
            "sudoku_primary_sampler_label": "top_prob_margin",
            "sudoku_eval_samplers": {
                "vanilla": {"method": "vanilla"},
                "top_probability": {"method": "top_probability"},
                "top_prob_margin": {"method": "top_prob_margin"},
            },
        }
    )

    maybe_eval = build_sudoku_eval_logger(
        cfg=cfg,
        eval_cfg=eval_cfg,
        task=_fake_board_task(),
        model=_FakeBoardModel(),
        wandb_mod=None,
        eval_every=1,
        log_at_step_zero=False,
    )
    metrics = maybe_eval(10, params_for_sampling={})

    assert calls == [
        ("uniform", 4),
        ("top_probability", 4),
        ("top_prob_margin", 4),
    ]
    assert metrics["eval/vanilla/solve_rate"] == 1.0
    assert metrics["eval/top_probability/board_acc_exact"] == 1.0
    assert metrics["eval/top_prob_margin/cell_acc_unknown"] == 1.0
