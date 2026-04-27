from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sticky.models.sjd.joint_input import (
    CELL_BOX_IDX,
    CELL_COL_IDX,
    CELL_ROW_IDX,
    SLACK_GROUP_IDX,
    SLACK_GROUP_TYPE_IDX,
    SudokuJointInputProj,
)


def _init_and_call(module: SudokuJointInputProj, cell_x, slack_x, *, key=0):
    rng = jax.random.PRNGKey(key)
    params = module.init(rng, cell_x, slack_x)
    return module.apply(params, cell_x, slack_x), params


def _zero_state(B: int = 2):
    cell = jnp.zeros((B, 81, 9), dtype=jnp.float32)
    slack = jnp.zeros((B, 27, 9), dtype=jnp.float32)
    return cell, slack


def test_output_shape_is_concatenated_to_108():
    module = SudokuJointInputProj(feature_dim=32)
    cell, slack = _zero_state()
    out, params = _init_and_call(module, cell, slack)
    assert out.shape == (2, 108, 32)
    p = params["params"]
    for key in (
        "cell_proj",
        "slack_proj",
        "site_type_emb",
        "row_emb",
        "col_emb",
        "box_emb",
        "group_type_emb",
        "group_idx_emb",
    ):
        assert key in p, f"missing param: {key}"
    assert p["site_type_emb"].shape == (2, 32)
    assert p["row_emb"].shape == (9, 32)
    assert p["col_emb"].shape == (9, 32)
    assert p["box_emb"].shape == (9, 32)
    assert p["group_type_emb"].shape == (3, 32)
    assert p["group_idx_emb"].shape == (9, 32)


def test_structural_embeddings_are_zero_initialized():
    module = SudokuJointInputProj(feature_dim=16)
    cell, slack = _zero_state(B=1)
    _, params = _init_and_call(module, cell, slack)
    p = params["params"]
    np.testing.assert_array_equal(np.asarray(p["row_emb"]), 0.0)
    np.testing.assert_array_equal(np.asarray(p["col_emb"]), 0.0)
    np.testing.assert_array_equal(np.asarray(p["box_emb"]), 0.0)
    np.testing.assert_array_equal(np.asarray(p["group_type_emb"]), 0.0)
    np.testing.assert_array_equal(np.asarray(p["group_idx_emb"]), 0.0)


def test_init_time_output_matches_no_structural_baseline():
    """Bit-equivalence: with structural embeddings init-zero, the init-time
    forward output equals what the previous (no-structural-emb) version would
    produce for the same cell_proj / slack_proj / site_type params."""
    module = SudokuJointInputProj(feature_dim=8)
    cell = jax.random.normal(jax.random.PRNGKey(0), (2, 81, 9), dtype=jnp.float32)
    slack = jax.random.normal(jax.random.PRNGKey(1), (2, 27, 9), dtype=jnp.float32)
    out, params = _init_and_call(module, cell, slack)

    # Recompute the legacy formula by hand from the same params.
    p = params["params"]
    cell_proj = cell @ p["cell_proj"]["kernel"] + p["cell_proj"]["bias"]
    slack_proj = slack @ p["slack_proj"]["kernel"] + p["slack_proj"]["bias"]
    cell_proj = cell_proj + p["site_type_emb"][0][None, None, :]
    slack_proj = slack_proj + p["site_type_emb"][1][None, None, :]
    expected = jnp.concatenate([cell_proj, slack_proj], axis=1)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), atol=1e-6)


def test_geometric_position_embeddings_distinguish_groups_after_training():
    """Once structural embeddings are non-zero (here we just set them by hand),
    two slacks belonging to different rows must produce distinguishable
    pre-attention vectors. Init-time they are identical (both rows live at
    group_type=0); post-training (or here, after we set group_idx_emb to
    non-zero) the row-0 slack and row-1 slack diverge."""
    module = SudokuJointInputProj(feature_dim=8)
    cell, slack = _zero_state(B=1)
    _, params = _init_and_call(module, cell, slack)
    p = dict(params["params"])

    rng = jax.random.PRNGKey(2)
    p_alt = dict(p)
    p_alt["group_idx_emb"] = jax.random.normal(rng, (9, 8), dtype=jnp.float32)
    out_alt = module.apply({"params": p_alt}, cell, slack)
    # Slack 0 (row 0) vs slack 1 (row 1): same group_type=row, different group_idx.
    assert not jnp.allclose(out_alt[0, 81], out_alt[0, 82])
    # Slack 0 (row 0) vs slack 9 (col 0): same group_idx=0, different group_type.
    # With group_type_emb still zero this would NOT distinguish them, so set it too.
    p_alt["group_type_emb"] = jax.random.normal(
        jax.random.PRNGKey(3), (3, 8), dtype=jnp.float32
    )
    out_alt2 = module.apply({"params": p_alt}, cell, slack)
    assert not jnp.allclose(out_alt2[0, 81], out_alt2[0, 90])


def test_cell_structural_indices_are_correct():
    """Sanity-check the CELL_{ROW,COL,BOX}_IDX module-level tables."""
    np.testing.assert_array_equal(
        np.asarray(CELL_ROW_IDX),
        np.repeat(np.arange(9, dtype=np.int32), 9),
    )
    np.testing.assert_array_equal(
        np.asarray(CELL_COL_IDX),
        np.tile(np.arange(9, dtype=np.int32), 9),
    )
    expected_box = np.zeros(81, dtype=np.int32)
    for r in range(9):
        for c in range(9):
            expected_box[r * 9 + c] = 3 * (r // 3) + (c // 3)
    np.testing.assert_array_equal(np.asarray(CELL_BOX_IDX), expected_box)


def test_slack_structural_indices_are_correct():
    np.testing.assert_array_equal(
        np.asarray(SLACK_GROUP_TYPE_IDX),
        np.concatenate(
            [np.zeros(9, dtype=np.int32), np.ones(9, dtype=np.int32), 2 * np.ones(9, dtype=np.int32)]
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(SLACK_GROUP_IDX),
        np.tile(np.arange(9, dtype=np.int32), 3),
    )


def test_rejects_wrong_site_counts():
    module = SudokuJointInputProj(feature_dim=8)
    rng = jax.random.PRNGKey(0)
    cell_short = jnp.zeros((1, 80, 9), dtype=jnp.float32)
    slack = jnp.zeros((1, 27, 9), dtype=jnp.float32)
    with pytest.raises(ValueError, match="81 sites"):
        module.init(rng, cell_short, slack)

    cell = jnp.zeros((1, 81, 9), dtype=jnp.float32)
    slack_short = jnp.zeros((1, 26, 9), dtype=jnp.float32)
    with pytest.raises(ValueError, match="27 sites"):
        module.init(rng, cell, slack_short)


def test_rejects_non_rank3_inputs():
    module = SudokuJointInputProj(feature_dim=8)
    cell = jnp.zeros((2, 81, 9), dtype=jnp.float32)
    slack_bad = jnp.zeros((2, 27 * 9), dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="rank-3"):
        module.init(rng, cell, slack_bad)


def test_rejects_batch_mismatch():
    module = SudokuJointInputProj(feature_dim=8)
    cell = jnp.zeros((2, 81, 9), dtype=jnp.float32)
    slack = jnp.zeros((3, 27, 9), dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="batch dim"):
        module.init(rng, cell, slack)
