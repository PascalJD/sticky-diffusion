from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from omegaconf import OmegaConf

import sticky.eval.sudoku_prop52 as prop52_mod
from sticky.eval.sudoku_prop52 import run_sudoku_prop52_diagnostics
from sticky.models.sjd.hazard import make_hazard_poly_alpha
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sdes import make_beta


jax = __import__("jax")
jnp = __import__("jax.numpy", fromlist=["jnp"])


def _solution_board(batch_size: int = 2) -> np.ndarray:
    board = np.asarray(
        [[((row * 3 + row // 3 + col) % 9) + 1 for row in range(9) for col in range(9)]],
        dtype=np.int32,
    )
    return np.repeat(board, batch_size, axis=0)


def _board_batch(batch_size: int = 2) -> dict[str, np.ndarray]:
    solution = _solution_board(batch_size)
    clue_mask = np.zeros((batch_size, 81), dtype=np.bool_)
    clue_mask[:, :18] = True
    clue_board = np.where(clue_mask, solution, 0).astype(np.int32)
    return {
        "solution_board": solution,
        "clue_board": clue_board,
        "clue_mask": clue_mask,
    }


@dataclass
class _FakeModel:
    anchor_dim: int = 4

    def __post_init__(self):
        ordered = jnp.linspace(-1.0, 1.0, 9, dtype=jnp.float32)[:, None]
        residual = jnp.zeros((9, self.anchor_dim - 1), dtype=jnp.float32)
        self._table = jnp.concatenate([ordered, residual], axis=1)

    def anchor_table(self):
        return self._table

    def apply(self, variables, y=None, t=None, train=False, method=None):
        del variables, t, train
        if method is not None:
            if getattr(method, "__name__", "") == "anchor_table":
                return self._table
            return method()
        logits = jnp.zeros(y.shape[:-1] + (9,), dtype=jnp.float32)
        return logits, {}


def _fake_task():
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_poly_alpha(beta, p=1.0, eps=1.0e-4)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.97, std_floor=1.0e-6)
    return type(
        "FakeTask",
        (),
        {
            "eval_batch_size": 2,
            "data_dir": None,
            "train_file": None,
            "test_file": None,
            "mmap": False,
            "max_test_examples": 4,
            "auto_download": False,
            "download_timeout_sec": 1,
            "download_retries": 0,
            "beta": beta,
            "hazard": hazard,
            "jump": jump,
        },
    )()


def test_prop52_eta_one_collapses_state_variance(monkeypatch):
    batches = [_board_batch(), _board_batch()]
    monkeypatch.setattr(
        prop52_mod,
        "make_sudoku_board_iterator",
        lambda **kwargs: iter(batches),
    )

    cfg = OmegaConf.create({"training": {"seed": 7}})
    eval_cfg = OmegaConf.create(
        {
            "sudoku_prop52_num_bins": 8,
            "sudoku_prop52_num_batches": 2,
            "sudoku_prop52_time_eps": 0.6,
            "sudoku_prop52_seed_offset": 101,
            "sudoku_prop52_fold_in_step": False,
        }
    )
    policy_specs = [
        {"policy": "plugin_hazard", "eta": 0.5, "logit_temperature": 1.0, "intensity_mode": "full", "log_ratio_clip": 10.0},
        {"policy": "plugin_hazard", "eta": 1.0, "logit_temperature": 1.0, "intensity_mode": "full", "log_ratio_clip": 10.0},
    ]

    result = run_sudoku_prop52_diagnostics(
        cfg=cfg,
        eval_cfg=eval_cfg,
        task=_fake_task(),
        model=_FakeModel(),
        params={},
        step_i=0,
        policy_specs=policy_specs,
        wandb_mod=None,
    )

    mean_v_eta_half = result.metrics["diag/eta_0p50/mean_v_state"]
    mean_v_eta_one = result.metrics["diag/eta_1p00/mean_v_state"]
    assert np.isfinite(mean_v_eta_half)
    assert np.isfinite(mean_v_eta_one)
    assert mean_v_eta_one < 0.25 * mean_v_eta_half
    assert result.collapse_summary["v_state_collapsed"] is True
