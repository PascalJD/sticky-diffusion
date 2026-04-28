"""Per-position anchor table tests for downstream consumers.

Covers Tasks 7-9 of the per-cell-anchors-sudoku plan: dhm_log_ratio,
classifier_induced_score, and the sampler's commit gather.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import jax.numpy as jnp

from sticky.models.sjd.anchors import AnchorTable
from sticky.models.sjd.corruption import classifier_induced_score
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.plugin_intensity import dhm_log_ratio
from sticky.models.sjd.sampler import _gather_committed_anchor
from sticky.models.sjd.sdes import make_beta


def _make_kit():
    beta = make_beta(beta_min=0.1, beta_max=0.3, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=1.7)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.75, std_floor=1e-3)
    return beta, hazard, jump


# -------------------------------------------------------------------------
# Task 7: dhm_log_ratio
# -------------------------------------------------------------------------


def test_dhm_log_ratio_n_positions_eq_1_bit_identical_when_table_replicated():
    """Replicating a (V, d) table to (P, V, d) where every position holds the
    same anchors should produce the same per-anchor log-ratios as the rank-2
    path."""
    beta, hazard, jump = _make_kit()
    L, d = 3, 2
    P = 2  # site axis

    base_table = jnp.asarray(
        [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.5]], dtype=jnp.float32
    )
    rank3_table = jnp.broadcast_to(base_table[None, :, :], (P, L, d))

    anchors_2d = AnchorTable(table_float=base_table)
    anchors_3d = AnchorTable(table_float=rank3_table)

    y = jnp.asarray(
        [[[-0.4, 0.2], [0.7, -0.1]], [[0.5, 0.9], [-0.8, 0.3]]],
        dtype=jnp.float32,
    )
    t_img = jnp.asarray([0.2, 0.8], dtype=jnp.float32)

    out_2d = dhm_log_ratio(
        y=y, t_img=t_img, anchors=anchors_2d, beta=beta, hazard=hazard, jump=jump,
    )
    out_3d = dhm_log_ratio(
        y=y, t_img=t_img, anchors=anchors_3d, beta=beta, hazard=hazard, jump=jump,
    )
    np.testing.assert_allclose(np.asarray(out_3d), np.asarray(out_2d), atol=1e-5)


def test_dhm_log_ratio_per_position_shape():
    beta, hazard, jump = _make_kit()
    L, d, P, B = 3, 2, 4, 2
    rng = np.random.default_rng(0)
    table = jnp.asarray(rng.normal(size=(P, L, d)).astype(np.float32))
    anchors = AnchorTable(table_float=table)
    y = jnp.asarray(rng.normal(size=(B, P, d)).astype(np.float32))
    t_img = jnp.asarray([0.3, 0.7], dtype=jnp.float32)

    out = dhm_log_ratio(
        y=y, t_img=t_img, anchors=anchors, beta=beta, hazard=hazard, jump=jump,
    )
    assert out.shape == (B, P, L)


def test_dhm_log_ratio_per_position_uses_distinct_anchors_per_site():
    """If every position has its own *unique* anchor 0, then the log-ratio
    at site p label 0 should differ across p (because the anchor differs).
    """
    beta, hazard, jump = _make_kit()
    L, d, P = 3, 2, 4
    rng = np.random.default_rng(1)
    table = jnp.asarray(rng.normal(size=(P, L, d)).astype(np.float32))
    anchors = AnchorTable(table_float=table)
    y = jnp.zeros((1, P, d), dtype=jnp.float32)
    t_img = jnp.asarray([0.5], dtype=jnp.float32)

    out = dhm_log_ratio(
        y=y, t_img=t_img, anchors=anchors, beta=beta, hazard=hazard, jump=jump,
    )
    # Slice the label-0 column at every position — should not all be equal.
    label0_per_site = np.asarray(out[0, :, 0])
    assert not np.allclose(label0_per_site, label0_per_site[0])


# -------------------------------------------------------------------------
# Task 8: classifier_induced_score
# -------------------------------------------------------------------------


def test_classifier_induced_score_n_positions_eq_1_bit_identical():
    beta, hazard, jump = _make_kit()
    L, d, P, B = 3, 2, 2, 2

    base_table = jnp.asarray(
        [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.5]], dtype=jnp.float32
    )
    rank3_table = jnp.broadcast_to(base_table[None, :, :], (P, L, d))
    anchors_2d = AnchorTable(table_float=base_table)
    anchors_3d = AnchorTable(table_float=rank3_table)

    rng = np.random.default_rng(2)
    logits = jnp.asarray(rng.normal(size=(B, P, L)).astype(np.float32))
    y = jnp.asarray(rng.normal(size=(B, P, d)).astype(np.float32))
    t = jnp.asarray([0.3, 0.7], dtype=jnp.float32)

    s_2d = classifier_induced_score(
        y, t,
        anchor_logits=logits, anchors=anchors_2d,
        beta=beta, hazard=hazard, jump=jump,
    )
    s_3d = classifier_induced_score(
        y, t,
        anchor_logits=logits, anchors=anchors_3d,
        beta=beta, hazard=hazard, jump=jump,
    )
    np.testing.assert_allclose(np.asarray(s_3d), np.asarray(s_2d), atol=1e-5)


def test_classifier_induced_score_per_position_shape():
    beta, hazard, jump = _make_kit()
    L, d, P, B = 3, 2, 4, 2
    rng = np.random.default_rng(3)
    table = jnp.asarray(rng.normal(size=(P, L, d)).astype(np.float32))
    anchors = AnchorTable(table_float=table)
    logits = jnp.asarray(rng.normal(size=(B, P, L)).astype(np.float32))
    y = jnp.asarray(rng.normal(size=(B, P, d)).astype(np.float32))
    t = jnp.asarray([0.3, 0.7], dtype=jnp.float32)

    score = classifier_induced_score(
        y, t,
        anchor_logits=logits, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump,
    )
    assert score.shape == y.shape
    assert np.all(np.isfinite(np.asarray(score)))


def test_classifier_induced_score_per_position_rejects_size_mismatch():
    beta, hazard, jump = _make_kit()
    L, d, P_table, B = 3, 2, 4, 2
    P_y = 5  # mismatch
    rng = np.random.default_rng(4)
    table = jnp.asarray(rng.normal(size=(P_table, L, d)).astype(np.float32))
    anchors = AnchorTable(table_float=table)
    logits = jnp.asarray(rng.normal(size=(B, P_y, L)).astype(np.float32))
    y = jnp.asarray(rng.normal(size=(B, P_y, d)).astype(np.float32))
    t = jnp.asarray([0.3, 0.7], dtype=jnp.float32)

    import pytest
    with pytest.raises(ValueError, match="P=4"):
        classifier_induced_score(
            y, t,
            anchor_logits=logits, anchors=anchors,
            beta=beta, hazard=hazard, jump=jump,
        )


# -------------------------------------------------------------------------
# Task 9: sampler commit gather helper
# -------------------------------------------------------------------------


def test_gather_committed_anchor_rank2_is_table_indexed_by_label():
    a_table = jnp.asarray(np.arange(6.0).reshape(3, 2), dtype=jnp.float32)
    a_idx = jnp.asarray([[0, 2, 1, 0]], dtype=jnp.int32)
    out = _gather_committed_anchor(a_table, a_idx)
    expected = np.stack([np.asarray(a_table[i]) for i in [0, 2, 1, 0]])[None, :, :]
    np.testing.assert_allclose(np.asarray(out), expected)


def test_gather_committed_anchor_rank3_is_position_aware():
    P, V, d = 4, 3, 2
    a_table = jnp.asarray(
        np.arange(P * V * d, dtype=np.float32).reshape(P, V, d)
    )
    # Two batch elements, different label patterns across positions.
    a_idx = jnp.asarray([[2, 0, 1, 2], [1, 1, 2, 0]], dtype=jnp.int32)
    out = _gather_committed_anchor(a_table, a_idx)
    assert out.shape == (2, P, d)
    for b in range(2):
        for p in range(P):
            np.testing.assert_allclose(
                np.asarray(out[b, p]),
                np.asarray(a_table[p, a_idx[b, p]]),
            )
