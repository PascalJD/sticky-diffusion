"""Tests for the cell_only_input_proj flag on SJD.

When True, an explicit Dense(feature_dim) layer projects the cell-only input
before it reaches the classifier, matching what the slack-augmented path does
via SudokuJointInputProj.cell_proj. The flag exists so that cell-only
baselines compared against the slack-augmented variant are architecturally
apples-to-apples (one Dense input projection on cell features in both paths,
in the same place under the same param name pattern).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sticky.models.sjd.anchors import AnchorTableConfig
from sticky.models.sjd.sjd_model import SJD


def _make_model(*, cell_only_input_proj: bool) -> SJD:
    return SJD(
        anchor_config=AnchorTableConfig(
            family="simplex_vertex", vocab_size=9, anchor_dim=9
        ),
        learnable_anchors=False,
        enable_joint_input=False,
        cell_only_input_proj=cell_only_input_proj,
        feature_dim=8,
        n_layers=1,
        num_heads=2,
        sequence_backbone="gpt2_like",
        sequence_max_length=81,
        sequence_causal=False,
        sequence_mlp_hidden_dim=16,
        vocab_size=9,
    )


def _init_params(model: SJD, *, with_anchor_ids: bool):
    cell = jnp.zeros((1, 81, 9), dtype=jnp.float32)
    t = jnp.zeros((1,), dtype=jnp.float32)
    if with_anchor_ids:
        return model.init(
            jax.random.PRNGKey(0),
            cell,
            t,
            anchor_token_ids=jnp.zeros((1, 81), dtype=jnp.int32),
            train=False,
        )["params"]
    return model.init(jax.random.PRNGKey(0), cell, t, train=False)["params"]


def test_default_is_false_and_no_extra_param_exists():
    model = _make_model(cell_only_input_proj=False)
    params = _init_params(model, with_anchor_ids=True)
    assert "cell_only_input_proj" not in params


def test_flag_introduces_dense_layer_with_expected_shape():
    model = _make_model(cell_only_input_proj=True)
    params = _init_params(model, with_anchor_ids=True)
    assert "cell_only_input_proj" in params
    kernel = params["cell_only_input_proj"]["kernel"]
    bias = params["cell_only_input_proj"]["bias"]
    assert kernel.shape == (9, 8)
    assert bias.shape == (8,)


def test_cell_only_path_works_with_flag_off_and_on():
    """With the flag, the path projects 9 -> feature_dim outside the backbone;
    without it, the backbone's own input_proj fires. Both produce (B, 81, V)."""
    cell = jax.random.normal(jax.random.PRNGKey(7), (2, 81, 9), dtype=jnp.float32)
    t = jnp.full((2,), 0.5, dtype=jnp.float32)

    for flag in (False, True):
        model = _make_model(cell_only_input_proj=flag)
        params = _init_params(model, with_anchor_ids=True)
        logits, _ = model.apply({"params": params}, cell, t, train=False)
        assert logits.shape == (2, 81, 9), f"flag={flag}: got {logits.shape}"
        assert np.all(np.isfinite(np.asarray(logits)))
