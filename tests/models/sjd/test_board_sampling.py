from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from sticky.models.sjd.anchors import AnchorTable
import sticky.models.sjd.board_sampling as board_sampling_mod
from sticky.models.sjd.hazard import make_hazard_poly_alpha
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sdes import make_beta


def _solution_board() -> np.ndarray:
    return np.asarray(
        [[((row * 3 + row // 3 + col) % 9) + 1 for row in range(9) for col in range(9)]],
        dtype=np.int32,
    )


def _clue_inputs() -> tuple[np.ndarray, np.ndarray]:
    solution = _solution_board()
    clue_mask = np.zeros((1, 81), dtype=np.bool_)
    clue_mask[0, :24] = True
    clue_board = np.where(clue_mask, solution, 0).astype(np.int32)
    return clue_board, clue_mask


def _anchor_table() -> AnchorTable:
    base = jnp.eye(9, dtype=jnp.float32)
    return AnchorTable(table_float=base)


class _RecordingSudokuModel:
    def __init__(self, *, solution_digits: np.ndarray, anchor_table: AnchorTable, known_mask: np.ndarray):
        self.solution_idx = jnp.asarray(solution_digits - 1, dtype=jnp.int32)
        self.anchor_table_value = anchor_table.table_float
        self.known_mask = np.asarray(known_mask, dtype=np.bool_)
        self.calls: list[np.ndarray] = []

    def anchor_table(self):
        return self.anchor_table_value

    def apply(self, variables, *args, method=None, **kwargs):
        del variables, kwargs
        if method is not None and getattr(method, "__name__", "") == "anchor_table":
            return self.anchor_table()

        y = jnp.asarray(args[0], dtype=jnp.float32)
        self.calls.append(np.asarray(y))
        dists = jnp.sum(
            (y[..., None, :] - self.anchor_table_value[None, None, :, :]) ** 2,
            axis=-1,
        )
        logits = -dists
        logits = logits.at[..., :].add(-0.5)
        logits = logits.at[
            jnp.arange(y.shape[0])[:, None],
            jnp.arange(y.shape[1])[None, :],
            self.solution_idx,
        ].add(4.0)
        return logits, {}


def test_expected_reveal_counts_matches_linear_formula():
    masked = jnp.asarray([[True, True, True, True, False]], dtype=jnp.bool_)
    counts = board_sampling_mod.expected_reveal_counts(masked, alpha_t=0.2, alpha_s=0.6)
    manual = round(4 * (0.6 - 0.2) / (1.0 - 0.2))
    assert int(counts[0]) == manual


@pytest.mark.parametrize(
    "policy_kwargs",
    [
        {"policy": "linear_survival", "sampling_grid": "uniform"},
        {"policy": "cosine_survival", "sampling_grid": "cosine"},
        {"policy": "linear_topk_probability", "sampling_grid": "uniform"},
        {"policy": "linear_topk_margin", "sampling_grid": "uniform"},
        {"policy": "plugin_hazard", "sampling_grid": "uniform", "eta": 1.0},
    ],
)
def test_sjd_board_policies_clamp_hints_and_produce_digits(policy_kwargs):
    solution = _solution_board()
    clue_board, clue_mask = _clue_inputs()
    anchors = _anchor_table()
    model = _RecordingSudokuModel(
        solution_digits=solution,
        anchor_table=anchors,
        known_mask=clue_mask,
    )
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_poly_alpha(beta, p=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.8)

    pred, diag = board_sampling_mod.conditional_generate(
        jax.random.PRNGKey(0),
        params={},
        model=model,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        known_tokens=jnp.asarray(clue_board, dtype=jnp.int32),
        known_token_mask=jnp.asarray(clue_mask, dtype=jnp.bool_),
        n_steps=6,
        return_diagnostics=True,
        **policy_kwargs,
    )

    pred_np = np.asarray(pred)
    assert pred_np.shape == (1, 81)
    assert set(np.unique(pred_np)).issubset(set(range(1, 10)))
    np.testing.assert_array_equal(pred_np[clue_mask], clue_board[clue_mask])
    assert np.isfinite(np.asarray(jax.device_get(diag["selected_count_total_across_steps"]))).all()

    known_idx = clue_board[clue_mask] - 1
    known_anchor_rows = np.asarray(anchors.table_float)[known_idx]
    for seen in model.calls:
        seen_known = seen[0, clue_mask[0], :]
        np.testing.assert_allclose(seen_known, known_anchor_rows, atol=1e-5)


def test_plugin_hazard_eta_1_uses_plugin_hazard_selection(monkeypatch):
    solution = _solution_board()
    clue_board, clue_mask = _clue_inputs()
    anchors = _anchor_table()
    model = _RecordingSudokuModel(
        solution_digits=solution,
        anchor_table=anchors,
        known_mask=clue_mask,
    )
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_poly_alpha(beta, p=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.8)

    seen_policies: list[str] = []
    original = board_sampling_mod._selection_scores_from_policy

    def _capture_policy(**kwargs):
        seen_policies.append(str(kwargs["policy"]))
        return original(**kwargs)

    monkeypatch.setattr(board_sampling_mod, "_selection_scores_from_policy", _capture_policy)

    board_sampling_mod.conditional_generate(
        jax.random.PRNGKey(1),
        params={},
        model=model,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        known_tokens=jnp.asarray(clue_board, dtype=jnp.int32),
        known_token_mask=jnp.asarray(clue_mask, dtype=jnp.bool_),
        n_steps=4,
        policy="plugin_hazard",
        sampling_grid="uniform",
        eta=1.0,
    )

    assert "plugin_hazard" in seen_policies
    assert "linear_survival" not in seen_policies


def test_final_masked_unknown_total_reports_pre_decode_unresolved_cells(monkeypatch):
    solution = _solution_board()
    clue_board, clue_mask = _clue_inputs()
    anchors = _anchor_table()
    model = _RecordingSudokuModel(
        solution_digits=solution,
        anchor_table=anchors,
        known_mask=clue_mask,
    )
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_poly_alpha(beta, p=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.8)

    def _zero_plugin(**kwargs):
        logits = kwargs["logits"]
        probs = jax.nn.softmax(logits, axis=-1)
        lam = jnp.zeros(logits.shape[:-1], dtype=jnp.float32)
        return lam, probs

    monkeypatch.setattr(board_sampling_mod, "plugin_intensity_and_probs", _zero_plugin)

    _, diag = board_sampling_mod.conditional_generate(
        jax.random.PRNGKey(2),
        params={},
        model=model,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        known_tokens=jnp.asarray(clue_board, dtype=jnp.int32),
        known_token_mask=jnp.asarray(clue_mask, dtype=jnp.bool_),
        n_steps=2,
        policy="plugin_hazard",
        sampling_grid="uniform",
        eta=0.97,
        return_diagnostics=True,
    )

    assert float(diag["final_masked_unknown_total"]) == float(np.sum(~clue_mask))
