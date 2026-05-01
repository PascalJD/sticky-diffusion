"""Tests for the composed `plugin_hazard_topk_margin` policy.

The policy combines:
  * reveal *count* from the plug-in p_commit sum (same as plugin_hazard), and
  * site *selection score* = top1(plug-in choice_probs) - top2(plug-in choice_probs).

This is the "best of both" cell of the Sudoku SJD policy table: the count is
state-dependent (driven by the plug-in commit rate), but the ranking inside
that count is decided by which-digit confidence rather than the commit rate
itself.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from sticky.models.sjd.anchors import AnchorTable
import sticky.models.sjd.board_sampling as board_sampling_mod
from sticky.models.sjd.board_sampling import (
    _selection_scores_from_policy,
    conditional_generate,
    conditional_generate_with_slack,
    normalize_policy_name,
)
from sticky.models.sjd.hazard import make_hazard_poly_alpha
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sdes import make_beta


# ---------- shared fixtures (mirrors test_board_sampling.py) ----------


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
    return AnchorTable(table_float=jnp.eye(9, dtype=jnp.float32))


class _RecordingSudokuModel:
    def __init__(self, *, solution_digits: np.ndarray, anchor_table: AnchorTable, known_mask: np.ndarray):
        self.solution_idx = jnp.asarray(solution_digits - 1, dtype=jnp.int32)
        self.anchor_table_value = anchor_table.table_float
        self.known_mask = np.asarray(known_mask, dtype=np.bool_)

    def anchor_table(self):
        return self.anchor_table_value

    def apply(self, variables, *args, method=None, **kwargs):
        del variables, kwargs
        if method is not None and getattr(method, "__name__", "") == "anchor_table":
            return self.anchor_table()

        y = jnp.asarray(args[0], dtype=jnp.float32)
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


def _schedule():
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_poly_alpha(beta, p=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.8)
    return beta, hazard, jump


# ---------- 1. policy registered ----------


def test_plugin_hazard_topk_margin_is_registered():
    """normalize_policy_name accepts the new name without raising."""
    assert normalize_policy_name("plugin_hazard_topk_margin") == "plugin_hazard_topk_margin"


# ---------- 2. selection score equals linear_topk_margin formula on the plug-in alloc ----------


def test_selection_score_equals_top_margin_of_choice_probs():
    """The selection score for `plugin_hazard_topk_margin` must be
    top1(choice_probs) - top2(choice_probs), independent of p_commit.
    `_selection_scores_from_policy` is called with `choice_probs` already
    set to the plug-in allocation `pi_t^*`, so this is identical in form
    to `linear_topk_margin` but operates on the plug-in distribution."""
    rng = jax.random.PRNGKey(0)
    # Hand-built choice_probs with a clear margin ranking that differs from
    # any plausible p_commit ordering.
    #   site 0: peaked one-hot, top-margin ~ 0.96
    #   site 1: roughly uniform, top-margin ~ 0.0
    #   site 2: two-mode, top-margin ~ 0.0
    choice_probs = jnp.asarray(
        [[
            [0.98, 0.005, 0.005, 0.0025, 0.0025, 0.001, 0.001, 0.001, 0.001],
            [1.0 / 9] * 9,
            [0.45, 0.45, 0.02, 0.02, 0.02, 0.02, 0.02, 0.0, 0.0],
        ]],
        dtype=jnp.float32,
    )
    masked_unknown_mask = jnp.ones((1, 3), dtype=jnp.bool_)
    # p_commit values that would rank sites 1, 2, 0 (descending) — the
    # opposite of the margin ranking. The new policy must ignore p_commit
    # for ranking.
    p_commit = jnp.asarray([[0.10, 0.95, 0.50]], dtype=jnp.float32)

    margin_scores = _selection_scores_from_policy(
        rng=rng,
        policy="plugin_hazard_topk_margin",
        masked_unknown_mask=masked_unknown_mask,
        choice_probs=choice_probs,
        p_commit=p_commit,
    )
    # Must equal the linear_topk_margin formula.
    expected_scores = _selection_scores_from_policy(
        rng=rng,
        policy="linear_topk_margin",
        masked_unknown_mask=masked_unknown_mask,
        choice_probs=choice_probs,
        p_commit=None,
    )
    np.testing.assert_allclose(np.asarray(margin_scores), np.asarray(expected_scores), atol=1e-6)
    # And must NOT equal the plugin_hazard score (= p_commit).
    plugin_scores = _selection_scores_from_policy(
        rng=rng,
        policy="plugin_hazard",
        masked_unknown_mask=masked_unknown_mask,
        choice_probs=choice_probs,
        p_commit=p_commit,
    )
    assert not np.allclose(np.asarray(margin_scores), np.asarray(plugin_scores), atol=1e-6)
    # Hand check: site 0 should have the highest margin (≈0.975).
    margin_np = np.asarray(margin_scores).reshape(-1)
    assert int(np.argmax(margin_np)) == 0


# ---------- 3. reveal count matches plugin_hazard step-for-step ----------


def test_reveal_count_matches_plugin_hazard(monkeypatch):
    """For matching seeds and inputs, the per-step `reveal_count` and the
    selection RNG path under `plugin_hazard_topk_margin` must produce the
    same count as `plugin_hazard`. Both pull count from `sum_j p_commit_j`.
    We verify by patching `_selection_mask_from_scores` to log the
    `reveal_count` it received on each call."""
    solution = _solution_board()
    clue_board, clue_mask = _clue_inputs()
    anchors = _anchor_table()
    beta, hazard, jump = _schedule()

    counts_log: dict[str, list[np.ndarray]] = {"plugin_hazard": [], "plugin_hazard_topk_margin": []}
    # Snapshot the unwrapped function once so each label's wrapper delegates to
    # the real implementation, not to a previous label's wrapper.
    original_selection_mask = board_sampling_mod._selection_mask_from_scores

    def make_capture(label: str):
        def _capture(*, scores, masked_unknown_mask, reveal_count):
            counts_log[label].append(np.asarray(reveal_count))
            return original_selection_mask(
                scores=scores,
                masked_unknown_mask=masked_unknown_mask,
                reveal_count=reveal_count,
            )

        return _capture

    common = dict(
        params={},
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        known_tokens=jnp.asarray(clue_board, dtype=jnp.int32),
        known_token_mask=jnp.asarray(clue_mask, dtype=jnp.bool_),
        n_steps=4,
        sampling_grid="uniform",
        eta=1.0,
    )

    for label in ("plugin_hazard", "plugin_hazard_topk_margin"):
        model = _RecordingSudokuModel(
            solution_digits=solution,
            anchor_table=anchors,
            known_mask=clue_mask,
        )
        monkeypatch.setattr(
            board_sampling_mod, "_selection_mask_from_scores", make_capture(label)
        )
        conditional_generate(
            jax.random.PRNGKey(42),
            model=model,
            policy=label,
            **common,
        )

    assert len(counts_log["plugin_hazard"]) == len(counts_log["plugin_hazard_topk_margin"])
    for c1, c2 in zip(counts_log["plugin_hazard"], counts_log["plugin_hazard_topk_margin"]):
        np.testing.assert_array_equal(c1, c2)


# ---------- 4. end-to-end shape + value range, cell-only sampler ----------


def test_conditional_generate_shape_and_range_plugin_hazard_topk_margin():
    solution = _solution_board()
    clue_board, clue_mask = _clue_inputs()
    anchors = _anchor_table()
    beta, hazard, jump = _schedule()
    model = _RecordingSudokuModel(
        solution_digits=solution,
        anchor_table=anchors,
        known_mask=clue_mask,
    )
    pred, diag = conditional_generate(
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
        policy="plugin_hazard_topk_margin",
        sampling_grid="uniform",
        eta=1.0,
        return_diagnostics=True,
    )
    pred_np = np.asarray(pred)
    assert pred_np.shape == (1, 81)
    assert set(np.unique(pred_np)).issubset(set(range(1, 10)))
    np.testing.assert_array_equal(pred_np[clue_mask], clue_board[clue_mask])
    assert np.isfinite(np.asarray(jax.device_get(diag["selected_count_total_across_steps"]))).all()


# ---------- 5. slack-aware sampler accepts the new policy ----------


class _ZeroLogitsSlackModel:
    """Mirrors the helper from test_board_sampling_slack.py."""

    def __init__(self, anchor_dim: int = 9):
        self._table = jnp.eye(anchor_dim, dtype=jnp.float32)

    def apply(self, variables, *args, method=None, **kwargs):
        del variables
        if method is not None and getattr(method, "__name__", "") == "anchor_table":
            return self._table
        y_cells = jnp.asarray(args[0], dtype=jnp.float32)
        B = int(y_cells.shape[0])
        return jnp.zeros((B, 108, 9), dtype=jnp.float32), {}


def test_slack_sampler_accepts_plugin_hazard_topk_margin():
    beta, hazard, jump = _schedule()
    boards = _solution_board()
    known_tokens = jnp.asarray(boards, dtype=jnp.int32)
    known_mask = jnp.zeros_like(known_tokens, dtype=jnp.bool_)
    model = _ZeroLogitsSlackModel()
    pred = conditional_generate_with_slack(
        rng=jax.random.PRNGKey(0),
        params={},
        model=model,
        anchors=AnchorTable(table_float=jnp.eye(9, dtype=jnp.float32)),
        beta=beta,
        hazard=hazard,
        jump=jump,
        known_tokens=known_tokens,
        known_token_mask=known_mask,
        n_steps=4,
        policy="plugin_hazard_topk_margin",
        sampling_grid="uniform",
        eta=1.0,
        project_slacks_after_step=False,
    )
    pred_np = np.asarray(pred)
    assert pred_np.shape == (1, 81)
    assert int(np.min(pred_np)) >= 1
    assert int(np.max(pred_np)) <= 9
