from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sticky.models.sjd.multi_axis_input import MultiAxisInputProj
from sticky.models.sjd.state_layout import (
    SUDOKU_CELL_ONLY_LAYOUT,
    SUDOKU_SLACK_LAYOUT,
)
from sticky.models.sjd.sudoku_structural import SudokuStructuralAdapter


def _zero_state(layout, B: int = 2) -> dict[str, jnp.ndarray]:
    return {
        a.name: jnp.zeros((B, a.site_count, a.embedding_dim), dtype=jnp.float32)
        for a in layout.axes
    }


def test_multi_axis_input_proj_sudoku_slack_output_shape():
    module = MultiAxisInputProj(layout=SUDOKU_SLACK_LAYOUT, feature_dim=16)
    state = _zero_state(SUDOKU_SLACK_LAYOUT, B=2)
    rng = jax.random.PRNGKey(0)
    params = module.init(rng, state)
    out = module.apply(params, state)
    assert out.shape == (2, 108, 16)
    p = params["params"]
    # Per-axis Dense must exist for every axis.
    for axis in SUDOKU_SLACK_LAYOUT.axes:
        assert f"{axis.name}_proj" in p, f"missing {axis.name}_proj"
    assert p["site_type_emb"].shape == (4, 16)


def test_multi_axis_input_proj_cell_only_output_shape():
    module = MultiAxisInputProj(layout=SUDOKU_CELL_ONLY_LAYOUT, feature_dim=16)
    state = _zero_state(SUDOKU_CELL_ONLY_LAYOUT, B=3)
    rng = jax.random.PRNGKey(0)
    params = module.init(rng, state)
    out = module.apply(params, state)
    assert out.shape == (3, 81, 16)
    assert params["params"]["site_type_emb"].shape == (1, 16)


def test_concat_order_matches_layout_axis_order():
    """The first 81 sites must come from `cells`, then `row_slacks`,
    `col_slacks`, `box_slacks` in that order."""
    layout = SUDOKU_SLACK_LAYOUT
    module = MultiAxisInputProj(layout=layout, feature_dim=4)
    rng = jax.random.PRNGKey(1)
    state = {
        "cells": jnp.full((1, 81, 9), 1.0, dtype=jnp.float32),
        "row_slacks": jnp.full((1, 9, 9), 2.0, dtype=jnp.float32),
        "col_slacks": jnp.full((1, 9, 9), 3.0, dtype=jnp.float32),
        "box_slacks": jnp.full((1, 9, 9), 4.0, dtype=jnp.float32),
    }
    params = module.init(rng, state)
    out = module.apply(params, state)
    p = params["params"]
    # Cells slice — first 81 rows. Compute by hand: x @ cells_proj.kernel + bias + site_type[0].
    expected_cells_first = (
        state["cells"][0, 0] @ p["cells_proj"]["kernel"]
        + p["cells_proj"]["bias"]
        + p["site_type_emb"][0]
    )
    np.testing.assert_allclose(np.asarray(out[0, 0]), np.asarray(expected_cells_first), atol=1e-6)
    # First slack at offset 81.
    expected_row_slack_first = (
        state["row_slacks"][0, 0] @ p["row_slacks_proj"]["kernel"]
        + p["row_slacks_proj"]["bias"]
        + p["site_type_emb"][1]
    )
    np.testing.assert_allclose(np.asarray(out[0, 81]), np.asarray(expected_row_slack_first), atol=1e-6)
    # First box slack at offset 99.
    expected_box_slack_first = (
        state["box_slacks"][0, 0] @ p["box_slacks_proj"]["kernel"]
        + p["box_slacks_proj"]["bias"]
        + p["site_type_emb"][3]
    )
    np.testing.assert_allclose(np.asarray(out[0, 99]), np.asarray(expected_box_slack_first), atol=1e-6)


def test_structural_offsets_are_added_per_axis():
    layout = SUDOKU_SLACK_LAYOUT
    feat = 8
    module = MultiAxisInputProj(layout=layout, feature_dim=feat)
    rng = jax.random.PRNGKey(2)
    state = _zero_state(layout, B=1)
    structural = {
        "cells": jnp.ones((81, feat), dtype=jnp.float32),
        "row_slacks": jnp.full((9, feat), 7.0, dtype=jnp.float32),
    }
    params = module.init(rng, state, structural_offsets=structural)
    out_with = module.apply(params, state, structural_offsets=structural)
    out_without = module.apply(params, state)
    # Cells region: differs by exactly +1.
    np.testing.assert_allclose(
        np.asarray(out_with[0, :81] - out_without[0, :81]),
        np.ones((81, feat), dtype=np.float32),
        atol=1e-6,
    )
    # Row slacks (rows 81..89): differs by +7.
    np.testing.assert_allclose(
        np.asarray(out_with[0, 81:90] - out_without[0, 81:90]),
        7.0 * np.ones((9, feat), dtype=np.float32),
        atol=1e-6,
    )
    # Col & box slacks: unchanged.
    np.testing.assert_allclose(
        np.asarray(out_with[0, 90:108]),
        np.asarray(out_without[0, 90:108]),
        atol=1e-6,
    )


def test_rejects_missing_axis_in_state():
    module = MultiAxisInputProj(layout=SUDOKU_SLACK_LAYOUT, feature_dim=4)
    rng = jax.random.PRNGKey(0)
    state = _zero_state(SUDOKU_SLACK_LAYOUT, B=1)
    state.pop("col_slacks")
    with pytest.raises(KeyError, match="col_slacks"):
        module.init(rng, state)


def test_rejects_wrong_site_count():
    module = MultiAxisInputProj(layout=SUDOKU_SLACK_LAYOUT, feature_dim=4)
    rng = jax.random.PRNGKey(0)
    state = _zero_state(SUDOKU_SLACK_LAYOUT, B=1)
    state["cells"] = jnp.zeros((1, 80, 9), dtype=jnp.float32)
    with pytest.raises(ValueError, match="site_count=81"):
        module.init(rng, state)


def test_rejects_wrong_embedding_dim():
    module = MultiAxisInputProj(layout=SUDOKU_SLACK_LAYOUT, feature_dim=4)
    rng = jax.random.PRNGKey(0)
    state = _zero_state(SUDOKU_SLACK_LAYOUT, B=1)
    state["row_slacks"] = jnp.zeros((1, 9, 8), dtype=jnp.float32)
    with pytest.raises(ValueError, match="embedding_dim=9"):
        module.init(rng, state)


def test_rejects_structural_offset_with_wrong_shape():
    module = MultiAxisInputProj(layout=SUDOKU_SLACK_LAYOUT, feature_dim=8)
    rng = jax.random.PRNGKey(0)
    state = _zero_state(SUDOKU_SLACK_LAYOUT, B=1)
    bad_struct = {"cells": jnp.zeros((81, 16), dtype=jnp.float32)}
    with pytest.raises(ValueError, match="expected shape"):
        module.init(rng, state, structural_offsets=bad_struct)


# ---------------- SudokuStructuralAdapter ----------------


def test_sudoku_structural_adapter_emits_init_zero_offsets():
    feat = 6
    adapter = SudokuStructuralAdapter(feature_dim=feat)
    rng = jax.random.PRNGKey(3)
    params = adapter.init(rng)
    offsets = adapter.apply(params)
    assert set(offsets.keys()) == {"cells", "row_slacks", "col_slacks", "box_slacks"}
    np.testing.assert_array_equal(np.asarray(offsets["cells"]), 0.0)
    np.testing.assert_array_equal(np.asarray(offsets["row_slacks"]), 0.0)
    np.testing.assert_array_equal(np.asarray(offsets["col_slacks"]), 0.0)
    np.testing.assert_array_equal(np.asarray(offsets["box_slacks"]), 0.0)
    assert offsets["cells"].shape == (81, feat)
    assert offsets["row_slacks"].shape == (9, feat)


def test_sudoku_structural_adapter_distinguishes_groups_after_training():
    feat = 4
    adapter = SudokuStructuralAdapter(feature_dim=feat)
    params = adapter.init(jax.random.PRNGKey(0))
    p = dict(params["params"])
    # Set group_idx_emb to non-zero — slacks for different rows now diverge.
    p["group_idx_emb"] = jax.random.normal(jax.random.PRNGKey(7), (9, feat), dtype=jnp.float32)
    offsets = adapter.apply({"params": p})
    # Within a single slack axis: row 0 vs row 1 are now different.
    assert not jnp.allclose(offsets["row_slacks"][0], offsets["row_slacks"][1])


def test_sudoku_structural_adapter_distinguishes_cells_after_training():
    feat = 4
    adapter = SudokuStructuralAdapter(feature_dim=feat)
    params = adapter.init(jax.random.PRNGKey(0))
    p = dict(params["params"])
    p["row_emb"] = jax.random.normal(jax.random.PRNGKey(1), (9, feat), dtype=jnp.float32)
    offsets = adapter.apply({"params": p})
    # Cell at (0,0) and (1,0) belong to different rows; they should differ.
    assert not jnp.allclose(offsets["cells"][0], offsets["cells"][9])
