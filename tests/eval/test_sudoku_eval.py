from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pytest
from omegaconf import OmegaConf
from types import SimpleNamespace

from sticky.data.sudoku import pack_sudoku_seq2seq
import sticky.eval.sudoku as sudoku_eval_mod

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from sticky.eval.sudoku import (
    _board_from_sequence,
    _evaluate_batch_counts,
    build_sudoku_eval_logger,
    valid_solution,
)


def _solved_board_triples() -> np.ndarray:
    triples = np.zeros((81, 3), dtype=np.int32)
    idx = 0
    for row in range(9):
        for col in range(9):
            triples[idx] = (row, col, ((row * 3 + row // 3 + col) % 9) + 1)
            idx += 1
    return triples


def test_strict_solve_metric_requires_exact_reconstructed_board():
    triples = _solved_board_triples()
    pred_triples = triples.copy()
    pred_triples[:, 2] = (pred_triples[:, 2] % 9) + 1

    pred_seq = pred_triples.reshape(1, -1)
    puzzle_sol = _board_from_sequence(triples.reshape(-1))[None, :]

    assert valid_solution(pred_seq[0]) is True
    assert np.array_equal(_board_from_sequence(pred_seq[0]), puzzle_sol[0]) is False

    counts = _evaluate_batch_counts(
        pred_seq=pred_seq,
        puzzle_sol=puzzle_sol,
        start_index=np.zeros((1, 1), dtype=np.int32),
        input_seq=pred_seq,
    )

    assert counts["valid_complete"] == 1
    assert counts["board_exact"] == 0
    assert counts["strict_complete"] == 0


def test_batch_counts_report_token_and_duplicate_coordinate_diagnostics():
    triples = _solved_board_triples()
    pred_triples = triples.copy()
    pred_triples[79, 1] = pred_triples[78, 1]
    pred_triples[80, 0] = (pred_triples[80, 0] + 1) % 9
    pred_triples[80, 2] = (pred_triples[80, 2] % 9) + 1

    pred_seq = pred_triples.reshape(1, -1)
    input_seq = triples.reshape(1, -1)
    puzzle_sol = _board_from_sequence(triples.reshape(-1))[None, :]

    counts = _evaluate_batch_counts(
        pred_seq=pred_seq,
        puzzle_sol=puzzle_sol,
        start_index=np.asarray([[78]], dtype=np.int32),
        input_seq=input_seq,
    )

    assert counts["total_pred"] == 3
    assert counts["row_token_correct"] == 2
    assert counts["col_token_correct"] == 2
    assert counts["value_token_correct"] == 2
    assert counts["rowcol_correct_total"] == 1
    assert counts["value_given_correct_rowcol"] == 1
    assert counts["duplicate_coordinate_total"] == 1


@pytest.fixture
def solved_sudoku_batch():
    triples = _solved_board_triples().reshape(1, -1)
    puzzle_sol = _board_from_sequence(triples.reshape(-1))[None, :]
    return {
        "image": triples,
        "puzzle": puzzle_sol,
        "start_index": np.asarray([[78]], dtype=np.int32),
    }


class _FakeWandb:
    def __init__(self):
        self.calls = []

    def log(self, metrics, step=None):
        self.calls.append((step, dict(metrics)))


def _fake_sudoku_task():
    return SimpleNamespace(
        eval_batch_size=1,
        data_dir=None,
        train_file=None,
        test_file=None,
        seq_order="fixed",
        mmap=False,
        max_test_examples=1,
        auto_download=False,
        download_timeout_sec=1,
        download_retries=0,
        spec=SimpleNamespace(data_shape=(245,)),
    )


@dataclass
class _FakeMDMModel:
    sampler: str = "top_prob_margin"
    sampling_grid: str = "loglinear"
    categorical_sampling_policy: str = "exact"
    decoding_style: str = "monotone_reveal"
    revealed_token_sample_mode: str = "sample"
    cache_predictions: bool = False
    oracle_noise_type: str = "gumbel"
    oracle_noise_scale: float = 0.5
    timesteps: int = 50


def test_mdm_eval_runs_all_sampler_modes_with_distinct_metric_prefixes(
    monkeypatch,
    capsys,
    solved_sudoku_batch,
):
    batch = solved_sudoku_batch
    packed = pack_sudoku_seq2seq(
        triplet_seq=batch["image"],
        start_index=batch["start_index"],
    )

    monkeypatch.setattr(
        sudoku_eval_mod,
        "make_sudoku_iterator",
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
    ):
        del rng, train_state, conditioning, use_ema
        calls.append(
            (
                model.sampler,
                model.decoding_style,
                model.oracle_noise_type,
                float(model.oracle_noise_scale),
                timesteps,
                tuple(known_tokens.shape),
                tuple(known_token_mask.shape),
            )
        )
        margin_map = {
            "uniform": 0.1,
            "top_probability": 0.2,
            "top_prob_margin": 0.3,
        }
        diag = {
            "example_step_count": jnp.asarray(1.0, dtype=jnp.float32),
            "masked_unknown_total_across_steps": jnp.asarray(2.0, dtype=jnp.float32),
            "selected_count_total_across_steps": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_top_probability_sum_total": jnp.asarray(0.8, dtype=jnp.float32),
            "selected_top_probability_count_total": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_top_prob_margin_sum_total": jnp.asarray(
                margin_map[model.sampler], dtype=jnp.float32
            ),
            "selected_top_prob_margin_count_total": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_margin_sum_total": jnp.asarray(
                margin_map[model.sampler], dtype=jnp.float32
            ),
            "selected_margin_count_total": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_row_total_across_steps": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_col_total_across_steps": jnp.asarray(0.0, dtype=jnp.float32),
            "selected_value_total_across_steps": jnp.asarray(0.0, dtype=jnp.float32),
            "selected_eos_total_across_steps": jnp.asarray(0.0, dtype=jnp.float32),
            "unknown_token_total": jnp.asarray(3.0, dtype=jnp.float32),
            "final_masked_unknown_total": jnp.asarray(0.0, dtype=jnp.float32),
        }
        output = jnp.asarray(packed["packed_seq"], dtype=jnp.int32)
        return (output, diag) if return_diagnostics else output

    monkeypatch.setattr(
        __import__("sticky.models.baselines.mdm", fromlist=["conditional_generate"]),
        "conditional_generate",
        _fake_conditional_generate,
    )

    cfg = OmegaConf.create(
        {
            "model": {"name": "mdm", "timesteps": 50},
            "sampler": {
                "method": "top_prob_margin",
                "n_steps": 50,
                "sampling_grid": "loglinear",
                "categorical_sampling_policy": "exact",
                "revealed_token_sample_mode": "sample",
                "cache_predictions": False,
                "oracle_noise_type": "gumbel",
                "oracle_noise_scale": 0.5,
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
                    "vanilla": {
                        "method": "vanilla",
                        "decoding_style": "monotone_reveal",
                        "oracle_noise_type": "none",
                        "oracle_noise_scale": 0.0,
                    },
                    "top_probability": {
                        "method": "top_probability",
                        "decoding_style": "topk_remask",
                        "oracle_noise_type": "gumbel",
                        "oracle_noise_scale": 0.5,
                    },
                    "top_prob_margin": {
                        "method": "top_prob_margin",
                        "decoding_style": "topk_remask",
                        "oracle_noise_type": "gumbel",
                        "oracle_noise_scale": 0.5,
                    },
            },
        }
    )
    wandb = _FakeWandb()
    task = _fake_sudoku_task()
    model = _FakeMDMModel()

    maybe_eval = build_sudoku_eval_logger(
        cfg=cfg,
        eval_cfg=eval_cfg,
        task=task,
        model=model,
        wandb_mod=wandb,
        eval_every=1,
        log_at_step_zero=False,
    )
    metrics = maybe_eval(10, params_for_sampling={})

    assert metrics["eval/vanilla/acc_complete_puzzle"] == 1.0
    assert metrics["eval/top_probability/acc_complete_puzzle"] == 1.0
    assert metrics["eval/top_prob_margin/acc_complete_puzzle"] == 1.0
    assert metrics["eval/vanilla/mean_selected_top_prob_margin"] == pytest.approx(0.1)
    assert metrics["eval/top_probability/mean_selected_top_prob_margin"] == pytest.approx(0.2)
    assert metrics["eval/top_prob_margin/mean_selected_top_prob_margin"] == pytest.approx(0.3)
    assert metrics["eval/top_prob_margin/selected_row_fraction"] == pytest.approx(1.0)
    assert metrics["eval/top_prob_margin/selected_eos_fraction"] == pytest.approx(0.0)
    assert "eval/acc_complete_puzzle" not in metrics
    assert calls == [
        ("uniform", "monotone_reveal", "none", 0.0, 50, (1, 245), (1, 245)),
        ("top_probability", "topk_remask", "gumbel", 0.5, 50, (1, 245), (1, 245)),
        ("top_prob_margin", "topk_remask", "gumbel", 0.5, 50, (1, 245), (1, 245)),
    ]
    assert len(wandb.calls) == 1
    assert "eval/top_prob_margin/acc_complete_puzzle" in wandb.calls[0][1]

    stdout = capsys.readouterr().out
    assert "primary=top_prob_margin" in stdout
    assert "acc_complete_puzzle=1.0000" in stdout


def test_mdm_eval_single_sampler_fallback_keeps_base_metric_names(
    monkeypatch,
    solved_sudoku_batch,
):
    batch = solved_sudoku_batch
    packed = pack_sudoku_seq2seq(
        triplet_seq=batch["image"],
        start_index=batch["start_index"],
    )

    monkeypatch.setattr(
        sudoku_eval_mod,
        "make_sudoku_iterator",
        lambda **kwargs: iter([batch]),
    )

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
    ):
        del rng, train_state, model, timesteps, conditioning, use_ema
        output = jnp.asarray(packed["packed_seq"], dtype=jnp.int32)
        diag = {
            "example_step_count": jnp.asarray(1.0, dtype=jnp.float32),
            "masked_unknown_total_across_steps": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_count_total_across_steps": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_margin_sum_total": jnp.asarray(0.5, dtype=jnp.float32),
            "selected_margin_count_total": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_row_total_across_steps": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_col_total_across_steps": jnp.asarray(0.0, dtype=jnp.float32),
            "selected_value_total_across_steps": jnp.asarray(0.0, dtype=jnp.float32),
            "selected_eos_total_across_steps": jnp.asarray(0.0, dtype=jnp.float32),
            "unknown_token_total": jnp.asarray(3.0, dtype=jnp.float32),
            "final_masked_unknown_total": jnp.asarray(0.0, dtype=jnp.float32),
        }
        return (output, diag) if return_diagnostics else output

    monkeypatch.setattr(
        __import__("sticky.models.baselines.mdm", fromlist=["conditional_generate"]),
        "conditional_generate",
        _fake_conditional_generate,
    )

    cfg = OmegaConf.create(
        {
            "model": {"name": "mdm", "timesteps": 50},
            "sampler": {
                "method": "top_prob_margin",
                "n_steps": 50,
                "sampling_grid": "loglinear",
                "categorical_sampling_policy": "exact",
                "revealed_token_sample_mode": "sample",
                "cache_predictions": False,
                "oracle_noise_type": "gumbel",
                "oracle_noise_scale": 0.5,
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
            "sudoku_eval_seed_offset": 1776,
            "sudoku_sample_seed_offset": 314159,
            "sudoku_eval_fold_in_step": False,
            "sudoku_progress_every_batches": 20,
            "sudoku_run_all_sampler_modes": False,
            "sudoku_primary_sampler_label": "top_prob_margin",
        }
    )
    maybe_eval = build_sudoku_eval_logger(
        cfg=cfg,
        eval_cfg=eval_cfg,
        task=_fake_sudoku_task(),
        model=_FakeMDMModel(),
        wandb_mod=None,
        eval_every=1,
        log_at_step_zero=False,
    )
    metrics = maybe_eval(3, params_for_sampling={})

    assert metrics["eval/acc_complete_puzzle"] == 1.0
    assert metrics["eval/solve_rate"] == 1.0
    assert metrics["eval/selected_eos_fraction"] == 0.0
    assert "eval/top_prob_margin/acc_complete_puzzle" not in metrics
