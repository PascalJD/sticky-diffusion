from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sticky.models.sjd.joint_input import SudokuJointInputProj


def _init_and_call(module: SudokuJointInputProj, cell_x, slack_x, *, key=0):
    rng = jax.random.PRNGKey(key)
    params = module.init(rng, cell_x, slack_x)
    return module.apply(params, cell_x, slack_x), params


def test_output_shape_is_concatenated_to_108():
    module = SudokuJointInputProj(feature_dim=32)
    cell = jnp.zeros((2, 81, 9), dtype=jnp.float32)
    slack = jnp.zeros((2, 27, 9), dtype=jnp.float32)
    out, params = _init_and_call(module, cell, slack)
    assert out.shape == (2, 108, 32)
    assert "cell_proj" in params["params"]
    assert "slack_proj" in params["params"]
    assert "site_type_emb" in params["params"]
    assert params["params"]["site_type_emb"].shape == (2, 32)


def test_zero_inputs_yield_only_site_type_offsets():
    """With zero data, output rows equal the 2 learned site-type embedding rows."""
    module = SudokuJointInputProj(feature_dim=16)
    cell = jnp.zeros((1, 81, 9), dtype=jnp.float32)
    slack = jnp.zeros((1, 27, 9), dtype=jnp.float32)
    out, params = _init_and_call(module, cell, slack)
    site_emb = params["params"]["site_type_emb"]  # (2, 16)
    cell_bias = params["params"]["cell_proj"]["bias"]
    slack_bias = params["params"]["slack_proj"]["bias"]
    expected_cell = cell_bias + site_emb[0]
    expected_slack = slack_bias + site_emb[1]
    np.testing.assert_allclose(out[0, 0], expected_cell, atol=1e-6)
    np.testing.assert_allclose(out[0, 80], expected_cell, atol=1e-6)
    np.testing.assert_allclose(out[0, 81], expected_slack, atol=1e-6)
    np.testing.assert_allclose(out[0, 107], expected_slack, atol=1e-6)


def test_swapping_cell_and_slack_changes_output():
    """The site-type embedding must do something: swapping inputs of equal
    feature shape should produce different first-row outputs."""
    module = SudokuJointInputProj(feature_dim=8)
    cell = jax.random.normal(jax.random.PRNGKey(0), (1, 9, 9), dtype=jnp.float32)
    slack = jax.random.normal(jax.random.PRNGKey(1), (1, 9, 9), dtype=jnp.float32)
    rng = jax.random.PRNGKey(7)
    params = module.init(rng, cell, slack)
    out_a = module.apply(params, cell, slack)
    out_b = module.apply(params, slack, cell)
    # The first 9 rows of out_a use cell_proj + cell_type; out_b first 9 use
    # cell_proj + cell_type but on slack data — so first rows differ from out_a's.
    assert not jnp.allclose(out_a[:, :9], out_b[:, :9])


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
