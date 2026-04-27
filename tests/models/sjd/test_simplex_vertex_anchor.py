from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from sticky.models.sjd.anchors import (
    AnchorTableConfig,
    AnchorTransformConfig,
    anchor_learnable_from_mapping,
    build_anchor_table,
    build_anchor_table_views,
)


def _config(vocab_size: int = 9, anchor_dim: int = 9) -> AnchorTableConfig:
    return AnchorTableConfig(
        family="simplex_vertex",
        vocab_size=vocab_size,
        anchor_dim=anchor_dim,
    )


def test_simplex_vertex_table_is_identity():
    table = build_anchor_table(_config())
    np.testing.assert_array_equal(np.asarray(table), np.eye(9, dtype=np.float32))


def test_simplex_vertex_default_transform_is_a_no_op():
    views = build_anchor_table_views(_config())
    np.testing.assert_array_equal(np.asarray(views.raw), np.eye(9, dtype=np.float32))
    np.testing.assert_array_equal(
        np.asarray(views.transformed), np.eye(9, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(views.final), np.eye(9, dtype=np.float32)
    )


def test_simplex_vertex_requires_anchor_dim_equals_vocab_size():
    with pytest.raises(ValueError, match="anchor_dim == vocab_size"):
        build_anchor_table(
            AnchorTableConfig(
                family="simplex_vertex",
                vocab_size=9,
                anchor_dim=8,
            )
        )


def test_simplex_vertex_learnable_resolution_from_mapping():
    cfg = {
        "vocab_size": 9,
        "anchor": {"family": "simplex_vertex", "dim": 9, "learnable": False},
    }
    assert anchor_learnable_from_mapping(cfg) is False


def test_simplex_vertex_works_at_other_vocab_sizes():
    table = build_anchor_table(_config(vocab_size=4, anchor_dim=4))
    np.testing.assert_array_equal(np.asarray(table), np.eye(4, dtype=np.float32))


def test_simplex_vertex_round_trip_through_transform_pipeline():
    cfg = AnchorTableConfig(
        family="simplex_vertex",
        vocab_size=9,
        anchor_dim=9,
        transform=AnchorTransformConfig(scale=1.0),
    )
    table = build_anchor_table(cfg)
    assert tuple(table.shape) == (9, 9)
    assert jnp.allclose(table, jnp.eye(9, dtype=jnp.float32))
