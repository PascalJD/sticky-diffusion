from __future__ import annotations

import csv
from dataclasses import dataclass
import numpy as np
import pytest
from omegaconf import OmegaConf
from types import SimpleNamespace

import sticky.eval.sudoku as sudoku_eval_mod
from sticky.eval.sudoku import _evaluate_board_batch_counts, build_sudoku_eval_logger
from sticky.eval.sudoku_prop52 import Prop52DiagnosticsResult


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
    revealed_token_sample_mode: str = "sample"
    cache_predictions: bool = False
    oracle_noise_type: str = "none"
    oracle_noise_scale: float = 0.0
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


def _fake_mdlm_task():
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
        spec=SimpleNamespace(name="mdlm_sudoku", data_shape=(81,)),
    )


def _fake_sjd_task():
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
        beta=SimpleNamespace(T=1.0),
        hazard=object(),
        jump=SimpleNamespace(eta=0.97),
        spec=SimpleNamespace(name="sjd_sudoku_inpaint", data_shape=(81,), vocab_size=9),
    )


@dataclass
class _FakeSJDModel:
    def anchor_table(self):
        raise NotImplementedError

    def apply(self, variables, method=None):
        del variables, method
        return jnp.zeros((9, 4), dtype=jnp.float32)


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
            "model": {"name": "mdm", "timesteps": 4},
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


def test_mdlm_board_eval_logger_runs_all_sampler_modes(monkeypatch):
    solution = _solution_board()
    clue_board, clue_mask = _clue_inputs()
    batch = {
        "solution_board": solution,
        "clue_board": clue_board,
        "clue_mask": clue_mask,
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
    ):
        del rng, train_state, known_tokens, known_token_mask, conditioning, use_ema
        calls.append((model.sampler, timesteps))
        diag = {
            "example_step_count": jnp.asarray(1.0, dtype=jnp.float32),
            "masked_unknown_total_across_steps": jnp.asarray(2.0, dtype=jnp.float32),
            "selected_count_total_across_steps": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_margin_sum_total": jnp.asarray(0.4, dtype=jnp.float32),
            "selected_margin_count_total": jnp.asarray(1.0, dtype=jnp.float32),
            "unknown_token_total": jnp.asarray(float((~clue_mask).sum()), dtype=jnp.float32),
            "final_masked_unknown_total": jnp.asarray(0.0, dtype=jnp.float32),
        }
        output = jnp.asarray(solution, dtype=jnp.int32)
        return (output, diag) if return_diagnostics else output

    monkeypatch.setattr(
        __import__("sticky.models.baselines.mdlm.sudoku_sampling", fromlist=["conditional_generate"]),
        "conditional_generate",
        _fake_conditional_generate,
    )

    cfg = OmegaConf.create(
        {
            "model": {"name": "mdlm", "timesteps": 4},
            "sampler": {
                "method": "top_prob_margin",
                "n_steps": 4,
                "sampling_grid": "loglinear",
                "categorical_sampling_policy": "exact",
                "revealed_token_sample_mode": "sample",
                "cache_predictions": False,
                "oracle_noise_type": "none",
                "oracle_noise_scale": 0.0,
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
                "uniform": {"sampler": "mdlm_sudoku_uniform"},
                "top_probability": {"sampler": "mdlm_sudoku_top_probability"},
                "top_prob_margin": {"sampler": "mdlm_sudoku_top_prob_margin"},
            },
        }
    )

    maybe_eval = build_sudoku_eval_logger(
        cfg=cfg,
        eval_cfg=eval_cfg,
        task=_fake_mdlm_task(),
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
    assert metrics["eval/uniform/solve_rate"] == 1.0
    assert metrics["eval/top_probability/board_acc_exact"] == 1.0
    assert metrics["eval/top_prob_margin/cell_acc_unknown"] == 1.0


def test_sjd_board_eval_logger_runs_all_policies_and_logs_table(monkeypatch):
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
        *,
        params,
        model,
        anchors,
        beta,
        hazard,
        jump,
        known_tokens,
        known_token_mask,
        n_steps,
        policy,
        sampling_grid,
        logit_temperature=1.0,
        intensity_mode="full",
        log_ratio_clip=10.0,
        init_std=1.0,
        stochastic_k=False,
        eta=None,
        return_diagnostics=False,
    ):
        del (
            rng,
            params,
            model,
            anchors,
            beta,
            hazard,
            jump,
            known_tokens,
            known_token_mask,
            sampling_grid,
            logit_temperature,
            intensity_mode,
            log_ratio_clip,
            init_std,
            stochastic_k,
        )
        calls.append((policy, n_steps, eta))
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
        __import__("sticky.models.sjd.board_sampling", fromlist=["conditional_generate"]),
        "conditional_generate",
        _fake_conditional_generate,
    )

    def _fake_prop52(**kwargs):
        del kwargs
        return Prop52DiagnosticsResult(
            metrics={
                "diag/eta_0p50/mean_v_state": 0.4,
                "diag/eta_1p00/mean_v_state": 0.02,
            },
            wandb_payload={"diag/eta_0p50/delta_mse_vs_t": ("line", "delta")},
            rows_by_eta={"0p50": [{"eta": 0.5, "t_bin_center": 0.5, "count": 1}]},
            eta_summary_rows=[{"eta_label": "0p50", "eta": 0.5, "mean_v_state": 0.4}],
            collapse_summary={"v_state_collapsed": True},
        )

    monkeypatch.setattr(sudoku_eval_mod, "run_sudoku_prop52_diagnostics", _fake_prop52)

    logged = {}

    class _FakeWandb:
        class Table:
            def __init__(self, *, columns, data):
                self.columns = columns
                self.data = data

        def log(self, payload, step=None):
            logged["payload"] = payload
            logged["step"] = step

    cfg = OmegaConf.create(
        {
            "model": {"name": "sjd"},
            "sampler": {
                "n_steps": 4,
                "sampling_grid": "uniform",
                "logit_temperature": 1.0,
                "intensity_mode": "full",
                "log_ratio_clip": 10.0,
                "init_std": 1.0,
            },
            "training": {"seed": 0},
            "forward": {"jump": {"eta": 0.97}},
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
            "sudoku_primary_sampler_label": "plugin_hazard_eta_0p97",
            "sudoku_log_policy_table": True,
            "sudoku_prop52_enabled": True,
            "sudoku_eval_sjd_runs": {
                "linear_survival": {"kind": "policy", "policy": "linear_survival", "n_steps": 4, "sampling_grid": "uniform", "eta": 1.0},
                "plugin_hazard_eta_0p97": {"kind": "policy", "policy": "plugin_hazard", "n_steps": 4, "eta": 0.97},
                "plugin_hazard_eta_1p00": {"kind": "policy", "policy": "plugin_hazard", "n_steps": 4, "eta": 1.0},
            },
        }
    )

    maybe_eval = build_sudoku_eval_logger(
        cfg=cfg,
        eval_cfg=eval_cfg,
        task=_fake_sjd_task(),
        model=_FakeSJDModel(),
        wandb_mod=_FakeWandb(),
        eval_every=1,
        log_at_step_zero=False,
    )
    metrics = maybe_eval(10, params_for_sampling={})

    assert calls == [
        ("linear_survival", 4, 1.0),
        ("plugin_hazard", 4, 0.97),
        ("plugin_hazard", 4, 1.0),
    ]
    assert metrics["eval/linear_survival/solve_rate"] == 1.0
    assert metrics["eval/plugin_hazard_eta_0p97/full_cell_acc"] == 1.0
    assert metrics["eval/plugin_hazard_eta_1p00/board_acc"] == 1.0
    assert metrics["diag/eta_1p00/mean_v_state"] == 0.02
    assert "eval/policy_table" in logged["payload"]
    assert "diag/eta_0p50/delta_mse_vs_t" in logged["payload"]
    table = logged["payload"]["eval/policy_table"]
    assert table.columns == [
        "label",
        "kind",
        "policy",
        "eta",
        "predictor_corrector",
        "gate_type",
        "corrector_substeps",
        "corrector_strength",
        "n_steps",
        "nfe_total",
        "solve_rate",
        "cell_acc_unknown",
        "board_acc_exact",
        "clue_consistency_fraction",
        "wallclock_sec_per_board",
    ]
    assert len(table.data) == 3
    row_by_label = {row[0]: row for row in table.data}
    assert row_by_label["linear_survival"][1] == "policy"
    assert row_by_label["linear_survival"][2] == "linear_survival"
    assert row_by_label["plugin_hazard_eta_1p00"][2] == "plugin_hazard"
    assert maybe_eval.sudoku_prop52_collapse_summary == {"v_state_collapsed": True}


def test_sjd_sampler_profile_eval_logger_runs_pc_profiles_and_logs_table(monkeypatch):
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

    def _fake_conditional_generate_board(
        *,
        rng,
        params,
        model,
        anchors,
        beta,
        hazard,
        jump,
        known_tokens,
        known_token_mask,
        cfg,
    ):
        del rng, params, model, anchors, beta, hazard, jump, known_tokens, known_token_mask
        calls.append((bool(cfg.pc_enabled), int(cfg.corrector_substeps), float(cfg.corrector_step_scale), str(cfg.pc_gate)))
        diag = {
            "sampling/nfe_total": jnp.asarray(6.0 if cfg.pc_enabled else 4.0, dtype=jnp.float32),
            "sampling/continuous_to_anchor_commits_total": jnp.asarray(1.5, dtype=jnp.float32),
            "sampling/anchor_to_continuous_unstick_attempts_total": jnp.asarray(2.0, dtype=jnp.float32),
            "sampling/anchor_to_continuous_unstick_accepts_total": jnp.asarray(1.0, dtype=jnp.float32),
            "sampling/langevin_updates_total": jnp.asarray(3.0, dtype=jnp.float32),
            "sampling/gate_mean_continuous": jnp.asarray(0.25, dtype=jnp.float32),
            "sampling/gate_mean_committed": jnp.asarray(0.75, dtype=jnp.float32),
            "sampling/frac_committed_pre_force": jnp.asarray(0.5, dtype=jnp.float32),
            "sampling/frac_committed_final": jnp.asarray(1.0, dtype=jnp.float32),
            "sampling/fill_frac_by_final_jump": jnp.asarray(0.1, dtype=jnp.float32),
            "example_step_count": jnp.asarray(1.0, dtype=jnp.float32),
            "masked_unknown_total_across_steps": jnp.asarray(2.0, dtype=jnp.float32),
            "selected_count_total_across_steps": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_top_probability_sum_total": jnp.asarray(0.0, dtype=jnp.float32),
            "selected_top_probability_count_total": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_top_prob_margin_sum_total": jnp.asarray(0.0, dtype=jnp.float32),
            "selected_top_prob_margin_count_total": jnp.asarray(1.0, dtype=jnp.float32),
            "unknown_token_total": jnp.asarray(float((~clue_mask).sum()), dtype=jnp.float32),
            "final_masked_unknown_total": jnp.asarray(0.0, dtype=jnp.float32),
        }
        return jnp.asarray(solution, dtype=jnp.int32), diag

    monkeypatch.setattr(
        __import__("sticky.models.sjd.sampling", fromlist=["conditional_generate_board"]),
        "conditional_generate_board",
        _fake_conditional_generate_board,
    )

    logged = {}

    class _FakeWandb:
        class Table:
            def __init__(self, *, columns, data):
                self.columns = columns
                self.data = data

        def log(self, payload, step=None):
            logged["payload"] = payload
            logged["step"] = step

    cfg = OmegaConf.create(
        {
            "model": {"name": "sjd"},
            "sampler": {
                "n_steps": 4,
                "sampling_grid": "uniform",
                "score_scale": 1.0,
                "logit_temperature": 1.0,
                "categorical_sampling_policy": "exact",
                "alloc_mode": "argmax",
                "intensity_mode": "full",
                "log_ratio_clip": 10.0,
                "init_std": 1.0,
                "pc_enabled": False,
                "corrector_substeps": 0,
                "corrector_step_scale": 0.0,
                "pc_gate": "constant_one",
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
            "sudoku_primary_sampler_label": "pc_margin_l1_s0p10",
            "sudoku_log_policy_table": True,
            "sudoku_prop52_enabled": False,
            "sudoku_eval_sjd_runs": {
                "predictor_only": {"kind": "sampler", "sampler": "sjd_sudoku_predictor"},
                "pc_constant_l1_s0p10": {"kind": "sampler", "sampler": "sjd_sudoku_pc_constant"},
                "pc_margin_l1_s0p10": {"kind": "sampler", "sampler": "sjd_sudoku_pc_margin"},
            },
        }
    )

    maybe_eval = build_sudoku_eval_logger(
        cfg=cfg,
        eval_cfg=eval_cfg,
        task=_fake_sjd_task(),
        model=_FakeSJDModel(),
        wandb_mod=_FakeWandb(),
        eval_every=1,
        log_at_step_zero=False,
    )
    metrics = maybe_eval(10, params_for_sampling={})

    assert calls == [
        (False, 0, 0.0, "constant_one"),
        (True, 1, 0.1, "constant_one"),
        (True, 1, 0.1, "margin"),
    ]
    assert metrics["eval/predictor_only/solve_rate"] == 1.0
    assert metrics["eval/pc_margin_l1_s0p10/sampling/nfe_total"] == 6.0
    assert metrics["eval/pc_constant_l1_s0p10/sampling/langevin_updates_total"] == 3.0
    assert "eval/policy_table" in logged["payload"]
    table = logged["payload"]["eval/policy_table"]
    assert table.columns == [
        "label",
        "kind",
        "policy",
        "eta",
        "predictor_corrector",
        "gate_type",
        "corrector_substeps",
        "corrector_strength",
        "n_steps",
        "nfe_total",
        "solve_rate",
        "cell_acc_unknown",
        "board_acc_exact",
        "clue_consistency_fraction",
        "wallclock_sec_per_board",
    ]
    assert len(table.data) == 3
    assert metrics["eval/pc_margin_l1_s0p10/sampling/wallclock_sec_per_board"] >= 0.0


def test_sjd_eval_logger_writes_deduped_progress_csv(monkeypatch, tmp_path):
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

    def _policy_diag():
        return {
            "example_step_count": jnp.asarray(1.0, dtype=jnp.float32),
            "masked_unknown_total_across_steps": jnp.asarray(2.0, dtype=jnp.float32),
            "selected_count_total_across_steps": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_top_probability_sum_total": jnp.asarray(0.8, dtype=jnp.float32),
            "selected_top_probability_count_total": jnp.asarray(1.0, dtype=jnp.float32),
            "selected_top_prob_margin_sum_total": jnp.asarray(0.4, dtype=jnp.float32),
            "selected_top_prob_margin_count_total": jnp.asarray(1.0, dtype=jnp.float32),
            "unknown_token_total": jnp.asarray(float((~clue_mask).sum()), dtype=jnp.float32),
            "final_masked_unknown_total": jnp.asarray(1.0, dtype=jnp.float32),
        }

    def _fake_policy_conditional_generate(
        rng,
        *,
        params,
        model,
        anchors,
        beta,
        hazard,
        jump,
        known_tokens,
        known_token_mask,
        n_steps,
        policy,
        sampling_grid,
        logit_temperature,
        intensity_mode,
        log_ratio_clip,
        init_std,
        stochastic_k,
        eta,
        return_diagnostics,
    ):
        del (
            rng,
            params,
            model,
            anchors,
            beta,
            hazard,
            jump,
            known_tokens,
            known_token_mask,
            n_steps,
            policy,
            sampling_grid,
            logit_temperature,
            intensity_mode,
            log_ratio_clip,
            init_std,
            stochastic_k,
            eta,
        )
        pred = jnp.asarray(solution, dtype=jnp.int32)
        diag = _policy_diag()
        return (pred, diag) if return_diagnostics else pred

    def _fake_conditional_generate_board(
        *,
        rng,
        params,
        model,
        anchors,
        beta,
        hazard,
        jump,
        known_tokens,
        known_token_mask,
        cfg,
    ):
        del rng, params, model, anchors, beta, hazard, jump, known_tokens, known_token_mask
        diag = dict(_policy_diag())
        diag.update(
            {
                "sampling/nfe_total": jnp.asarray(6.0, dtype=jnp.float32),
                "sampling/continuous_to_anchor_commits_total": jnp.asarray(1.0, dtype=jnp.float32),
                "sampling/anchor_to_continuous_unstick_attempts_total": jnp.asarray(2.0, dtype=jnp.float32),
                "sampling/anchor_to_continuous_unstick_accepts_total": jnp.asarray(1.0, dtype=jnp.float32),
                "sampling/langevin_updates_total": jnp.asarray(3.0, dtype=jnp.float32),
                "sampling/gate_mean_continuous": jnp.asarray(0.25, dtype=jnp.float32),
                "sampling/gate_mean_committed": jnp.asarray(0.75, dtype=jnp.float32),
                "sampling/frac_committed_pre_force": jnp.asarray(0.5, dtype=jnp.float32),
                "sampling/frac_committed_final": jnp.asarray(1.0, dtype=jnp.float32),
                "sampling/fill_frac_by_final_jump": jnp.asarray(0.1, dtype=jnp.float32),
            }
        )
        pred = jnp.asarray(solution, dtype=jnp.int32)
        return pred, diag

    monkeypatch.setattr(
        __import__("sticky.models.sjd.board_sampling", fromlist=["conditional_generate"]),
        "conditional_generate",
        _fake_policy_conditional_generate,
    )
    monkeypatch.setattr(
        __import__("sticky.models.sjd.sampling", fromlist=["conditional_generate_board"]),
        "conditional_generate_board",
        _fake_conditional_generate_board,
    )

    cfg = OmegaConf.create(
        {
            "model": {"name": "sjd"},
            "sampler": {
                "T": 1.0,
                "n_steps": 4,
                "sampling_grid": "uniform",
                "score_scale": 1.0,
                "logit_temperature": 1.0,
                "categorical_sampling_policy": "exact",
                "hazard_mode": "plugin",
                "alloc_mode": "argmax",
                "intensity_mode": "full",
                "log_ratio_clip": 10.0,
                "intensity_chunk_size": 256,
                "init_std": 1.0,
                "force_classify_at_end": True,
                "refresh_logits_after_em_step": False,
                "pc_enabled": False,
                "corrector_substeps": 0,
                "corrector_step_scale": 0.0,
                "pc_gate": "constant_one",
                "pc_clamp_known": True,
                "pc_refresh_logits_after_langevin": False,
                "pc_allow_unstick_unknown_only": True,
                "metrics_count_nfe": True,
            },
            "training": {"seed": 0, "metrics_dir": "metrics"},
        }
    )
    progress_path = tmp_path / "metrics" / "sudoku_sjd_progress.csv"
    latest_path = tmp_path / "metrics" / "sudoku_sjd_latest.csv"
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
            "sudoku_primary_sampler_label": "pc_margin_l1_s0p10",
            "sudoku_log_policy_table": False,
            "sudoku_prop52_enabled": False,
            "sudoku_write_progress_csv": True,
            "sudoku_progress_csv_path": str(progress_path),
            "sudoku_write_latest_csv": True,
            "sudoku_latest_csv_path": str(latest_path),
            "sudoku_eval_sjd_runs": {
                "linear_survival": {
                    "kind": "policy",
                    "policy": "linear_survival",
                    "n_steps": 4,
                    "sampling_grid": "uniform",
                    "eta": 1.0,
                },
                "pc_margin_l1_s0p10": {
                    "kind": "sampler",
                    "sampler": "sjd_sudoku_pc_margin",
                },
            },
        }
    )

    maybe_eval = build_sudoku_eval_logger(
        cfg=cfg,
        eval_cfg=eval_cfg,
        task=_fake_sjd_task(),
        model=_FakeSJDModel(),
        wandb_mod=None,
        eval_every=1,
        log_at_step_zero=False,
    )
    maybe_eval(10, params_for_sampling={})
    maybe_eval(10, params_for_sampling={})

    with progress_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    rows_by_label = {row["label"]: row for row in rows}

    policy_row = rows_by_label["linear_survival"]
    sampler_row = rows_by_label["pc_margin_l1_s0p10"]

    assert policy_row["step"] == "10"
    assert sampler_row["step"] == "10"
    assert policy_row["kind"] == "policy"
    assert sampler_row["kind"] == "sampler"
    assert policy_row["sampling_nfe_total"] == ""
    assert policy_row["sampling_langevin_updates_total"] == ""
    assert sampler_row["sampling_nfe_total"] == "6.0"
    assert sampler_row["sampling_langevin_updates_total"] == "3.0"
    assert sampler_row["sampling_wallclock_sec_per_board"] != ""
    assert policy_row["full_cell_acc"] == "1.0"
    assert sampler_row["board_acc_exact"] == "1.0"
    assert policy_row["final_masked_unknown_fraction"] != ""

    with latest_path.open("r", encoding="utf-8", newline="") as handle:
        latest_rows = list(csv.DictReader(handle))
    assert len(latest_rows) == 2
    assert {row["label"] for row in latest_rows} == {"linear_survival", "pc_margin_l1_s0p10"}
