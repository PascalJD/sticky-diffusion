"""End-to-end test of the multi-axis SJD path: SJD.apply_layout."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sticky.models.sjd.anchors import AnchorTableConfig
from sticky.models.sjd.sjd_model import SJD
from sticky.models.sjd.state_layout import (
    SUDOKU_CELL_ONLY_LAYOUT,
    SUDOKU_SLACK_LAYOUT,
)


def _make_model(layout, *, use_sudoku_structural: bool) -> SJD:
    return SJD(
        anchor_config=AnchorTableConfig(
            family="simplex_vertex", vocab_size=9, anchor_dim=9
        ),
        learnable_anchors=False,
        feature_dim=8,
        n_layers=1,
        num_heads=2,
        sequence_backbone="gpt2_like",
        sequence_max_length=layout.total_site_count,
        sequence_causal=False,
        sequence_mlp_hidden_dim=16,
        vocab_size=9,
        state_layout=layout,
        use_sudoku_structural=use_sudoku_structural,
    )


def _zero_state(layout, B: int = 1):
    return {
        a.name: jnp.zeros((B, a.site_count, a.embedding_dim), dtype=jnp.float32)
        for a in layout.axes
    }


def test_apply_layout_returns_per_axis_logits_for_anchored_axes():
    layout = SUDOKU_SLACK_LAYOUT
    model = _make_model(layout, use_sudoku_structural=True)
    state = _zero_state(layout, B=2)
    rng = jax.random.PRNGKey(0)
    params = model.init(
        rng,
        method=model.apply_layout,
        state=state,
        t=jnp.zeros((2,), dtype=jnp.float32),
        train=False,
    )["params"]
    per_axis, _ = model.apply(
        {"params": params},
        method=model.apply_layout,
        state=state,
        t=jnp.zeros((2,), dtype=jnp.float32),
        train=False,
    )
    # Anchored axes only.
    assert set(per_axis.keys()) == {"cells"}
    assert per_axis["cells"].shape == (2, 81, 9)


def test_apply_layout_is_callable_via_default_call_with_state_kwarg():
    layout = SUDOKU_SLACK_LAYOUT
    model = _make_model(layout, use_sudoku_structural=False)
    state = _zero_state(layout, B=1)
    rng = jax.random.PRNGKey(1)
    # Init via __call__ with state kwarg.
    params = model.init(
        rng,
        jnp.zeros((1, 81, 9), dtype=jnp.float32),  # y_t (unused when state is set)
        jnp.zeros((1,), dtype=jnp.float32),
        state=state,
        train=False,
    )["params"]
    per_axis, _ = model.apply(
        {"params": params},
        jnp.zeros((1, 81, 9), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
        state=state,
        train=False,
    )
    assert set(per_axis.keys()) == {"cells"}
    assert per_axis["cells"].shape == (1, 81, 9)


def test_apply_layout_cell_only_layout_returns_one_axis_logits():
    layout = SUDOKU_CELL_ONLY_LAYOUT
    model = _make_model(layout, use_sudoku_structural=False)
    state = _zero_state(layout, B=2)
    rng = jax.random.PRNGKey(2)
    params = model.init(
        rng,
        method=model.apply_layout,
        state=state,
        t=jnp.zeros((2,), dtype=jnp.float32),
        train=False,
    )["params"]
    per_axis, _ = model.apply(
        {"params": params},
        method=model.apply_layout,
        state=state,
        t=jnp.zeros((2,), dtype=jnp.float32),
        train=False,
    )
    assert set(per_axis.keys()) == {"cells"}
    assert per_axis["cells"].shape == (2, 81, 9)


def test_apply_layout_init_includes_sudoku_structural_when_flag_on():
    layout = SUDOKU_SLACK_LAYOUT
    model = _make_model(layout, use_sudoku_structural=True)
    state = _zero_state(layout, B=1)
    params = model.init(
        jax.random.PRNGKey(3),
        method=model.apply_layout,
        state=state,
        t=jnp.zeros((1,), dtype=jnp.float32),
        train=False,
    )["params"]
    assert "sudoku_structural" in params
    assert "row_emb" in params["sudoku_structural"]
    assert "col_emb" in params["sudoku_structural"]
    assert "box_emb" in params["sudoku_structural"]
    assert "group_idx_emb" in params["sudoku_structural"]
    # All zero-initialized.
    for k in ("row_emb", "col_emb", "box_emb", "group_idx_emb"):
        np.testing.assert_array_equal(np.asarray(params["sudoku_structural"][k]), 0.0)


def test_apply_layout_sudoku_structural_off_by_default():
    layout = SUDOKU_SLACK_LAYOUT
    model = _make_model(layout, use_sudoku_structural=False)
    state = _zero_state(layout, B=1)
    params = model.init(
        jax.random.PRNGKey(4),
        method=model.apply_layout,
        state=state,
        t=jnp.zeros((1,), dtype=jnp.float32),
        train=False,
    )["params"]
    assert "sudoku_structural" not in params


def test_apply_layout_rejects_when_layout_is_none():
    """A model without a state_layout cannot run apply_layout."""
    model = SJD(
        anchor_config=AnchorTableConfig(
            family="simplex_vertex", vocab_size=9, anchor_dim=9
        ),
        learnable_anchors=False,
        feature_dim=8,
        n_layers=1,
        num_heads=2,
        sequence_backbone="gpt2_like",
        sequence_max_length=81,
        sequence_causal=False,
        vocab_size=9,
    )
    state = {"cells": jnp.zeros((1, 81, 9), dtype=jnp.float32)}
    rng = jax.random.PRNGKey(5)
    try:
        model.init(
            rng,
            method=model.apply_layout,
            state=state,
            t=jnp.zeros((1,), dtype=jnp.float32),
            train=False,
        )
    except ValueError as exc:
        assert "state_layout" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing state_layout")
