from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from sticky.models.sjd.anchors import (
    AnchorTable,
    AnchorTableConfig,
    AnchorTransformConfig,
    TokenAnchors,
    anchor_table_config_from_mapping,
    apply_anchor_transforms,
    build_anchor_table,
    clamp_known_state,
)


# -- Task 1: AnchorTableConfig.n_positions parsing ---------------------------


def test_anchor_table_config_default_n_positions_is_one():
    cfg = AnchorTableConfig(
        family="normal",
        vocab_size=9,
        anchor_dim=8,
    )
    assert cfg.n_positions == 1


def test_anchor_table_config_from_mapping_reads_nested_n_positions():
    mapping = {
        "name": "sjd",
        "anchor": {
            "family": "normal",
            "dim": 8,
            "seed": 0,
            "normalize_at_use": True,
            "n_positions": 81,
        },
    }
    cfg = anchor_table_config_from_mapping(mapping, vocab_size=9)
    assert cfg.n_positions == 81
    assert cfg.normalize_at_use is True


def test_anchor_table_config_from_mapping_reads_legacy_flat_n_positions():
    mapping = {
        "name": "sjd",
        "anchor_init": "normal",
        "anchor_dim": 8,
        "anchors_init_std": 1.0,
        "anchor_normalize_at_use": True,
        "anchor_n_positions": 81,
    }
    cfg = anchor_table_config_from_mapping(mapping, vocab_size=9)
    assert cfg.n_positions == 81


def test_anchor_table_config_rejects_zero_n_positions():
    with pytest.raises(ValueError):
        anchor_table_config_from_mapping(
            {
                "name": "sjd",
                "anchor": {
                    "family": "normal",
                    "dim": 4,
                    "n_positions": 0,
                    "normalize_at_use": True,
                },
            },
            vocab_size=9,
        )


# -- Task 2: rank-3 normal table builder + validation ------------------------


def test_normal_family_returns_rank3_when_n_positions_gt_1():
    cfg = AnchorTableConfig(
        family="normal",
        vocab_size=9,
        anchor_dim=4,
        seed=42,
        normalize_at_use=True,
        n_positions=5,
    )
    table = build_anchor_table(cfg)
    assert table.shape == (5, 9, 4)


def test_normal_family_per_position_is_deterministic_given_seed():
    cfg = AnchorTableConfig(
        family="normal",
        vocab_size=9,
        anchor_dim=4,
        seed=42,
        normalize_at_use=True,
        n_positions=5,
    )
    a = build_anchor_table(cfg)
    b = build_anchor_table(cfg)
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_normal_family_per_position_slices_are_independent():
    cfg = AnchorTableConfig(
        family="normal",
        vocab_size=9,
        anchor_dim=4,
        seed=7,
        normalize_at_use=True,
        n_positions=4,
    )
    table = np.asarray(build_anchor_table(cfg))
    # No two positions should be identical.
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.allclose(table[i], table[j])


def test_non_normal_family_rejects_n_positions_gt_1():
    cfg = AnchorTableConfig(
        family="ordered_normal",
        vocab_size=9,
        anchor_dim=4,
        seed=0,
        normalize_at_use=True,
        n_positions=4,
    )
    with pytest.raises(ValueError, match="normal_normalized"):
        build_anchor_table(cfg)


def test_normal_without_normalize_at_use_rejects_n_positions_gt_1():
    cfg = AnchorTableConfig(
        family="normal",
        vocab_size=9,
        anchor_dim=4,
        seed=0,
        normalize_at_use=False,
        n_positions=81,
    )
    with pytest.raises(ValueError, match="normalize_at_use"):
        build_anchor_table(cfg)


def test_n_positions_eq_1_returns_rank2_table():
    cfg = AnchorTableConfig(
        family="normal",
        vocab_size=9,
        anchor_dim=4,
        seed=0,
    )
    table = build_anchor_table(cfg)
    assert table.shape == (9, 4)


# -- Task 3: per-position transform application ------------------------------


def test_transform_applies_per_position():
    rng = np.random.default_rng(0)
    table = jnp.asarray(rng.normal(size=(3, 8, 4)).astype(np.float32))
    transform = AnchorTransformConfig(
        equalize_row_norms=True,
        target_row_norm=1.0,
    )
    out = apply_anchor_transforms(table, transform, include_scale=True)
    assert out.shape == (3, 8, 4)
    norms = np.linalg.norm(np.asarray(out), axis=-1)
    np.testing.assert_allclose(norms, np.ones((3, 8)), atol=1e-5)


def test_transform_rank_other_than_2_or_3_rejected():
    table = jnp.zeros((2, 2, 2, 2), dtype=jnp.float32)
    with pytest.raises(ValueError):
        apply_anchor_transforms(table, AnchorTransformConfig())


# -- Task 4: TokenAnchors rank-3 param + position-aware gather --------------


def test_token_anchors_param_shape_n_positions_eq_1_unchanged():
    cfg = AnchorTableConfig(
        family="normal",
        vocab_size=9,
        anchor_dim=4,
        seed=0,
    )
    module = TokenAnchors(config=cfg, learnable=True)
    ids = jnp.zeros((1, 9), dtype=jnp.int32)
    variables = module.init({"params": jnp.zeros((2,), dtype=jnp.uint32)}, ids)
    table = variables["params"]["table"]
    assert table.shape == (9, 4)


def test_token_anchors_param_shape_per_position():
    cfg = AnchorTableConfig(
        family="normal",
        vocab_size=9,
        anchor_dim=4,
        seed=0,
        normalize_at_use=True,
        n_positions=81,
    )
    module = TokenAnchors(config=cfg, learnable=True)
    ids = jnp.zeros((1, 81), dtype=jnp.int32)
    variables = module.init({"params": jnp.zeros((2,), dtype=jnp.uint32)}, ids)
    table = variables["params"]["table"]
    assert table.shape == (81, 9, 4)


def test_token_anchors_per_position_gather_matches_explicit():
    cfg = AnchorTableConfig(
        family="normal",
        vocab_size=9,
        anchor_dim=4,
        seed=3,
        normalize_at_use=True,
        n_positions=12,
    )
    module = TokenAnchors(config=cfg, learnable=True)
    ids = jnp.asarray(
        np.random.default_rng(0).integers(0, 9, size=(2, 12)),
        dtype=jnp.int32,
    )
    variables = module.init({"params": jnp.zeros((2,), dtype=jnp.uint32)}, ids)
    out = module.apply(variables, ids)
    raw_table = variables["params"]["table"]
    # Manually replicate the lookup including normalize_at_use.
    norms = jnp.linalg.norm(raw_table, axis=-1, keepdims=True)
    unit = raw_table / jnp.maximum(norms, 1e-12)
    scale = jnp.sqrt(jnp.asarray(raw_table.shape[-1], dtype=raw_table.dtype))
    table_norm = unit * scale
    s_idx = jnp.arange(12)
    expected = table_norm[jnp.broadcast_to(s_idx, ids.shape), ids]
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), atol=1e-6)


def test_token_anchors_per_position_rejects_wrong_seq_len():
    cfg = AnchorTableConfig(
        family="normal",
        vocab_size=9,
        anchor_dim=4,
        seed=0,
        normalize_at_use=True,
        n_positions=81,
    )
    module = TokenAnchors(config=cfg, learnable=True)
    bad_ids = jnp.zeros((1, 80), dtype=jnp.int32)
    with pytest.raises(ValueError, match="last-axis size"):
        module.init({"params": jnp.zeros((2,), dtype=jnp.uint32)}, bad_ids)


# -- Task 5: AnchorTable predicates ------------------------------------------


def test_anchor_table_predicates_rank2():
    table = jnp.zeros((9, 4), dtype=jnp.float32)
    at = AnchorTable(table_float=table)
    assert at.per_position is False
    assert at.n_positions == 1
    assert at.L == 9
    assert at.d == 4


def test_anchor_table_predicates_rank3():
    table = jnp.zeros((81, 9, 4), dtype=jnp.float32)
    at = AnchorTable(table_float=table)
    assert at.per_position is True
    assert at.n_positions == 81
    assert at.L == 9
    assert at.d == 4


def test_anchor_table_effective_log_w_shape_rank3():
    table = jnp.zeros((81, 9, 4), dtype=jnp.float32)
    at = AnchorTable(table_float=table)
    # log_w defaults to per-label only, shape (L,) = (9,).
    assert at.effective_log_w.shape == (9,)


# -- Task 6: clamp_known_state per-position gather ---------------------------


def test_clamp_known_state_rank2_unchanged():
    a_table = jnp.arange(20.0).reshape(5, 4)
    y = jnp.zeros((1, 3, 4))
    committed = jnp.zeros((1, 3), dtype=jnp.bool_)
    k_idx = jnp.zeros((1, 3), dtype=jnp.int32)
    known_mask = jnp.asarray([[True, False, True]])
    known_idx = jnp.asarray([[1, 0, 4]], dtype=jnp.int32)
    y_out, committed_out, k_idx_out = clamp_known_state(
        y=y,
        committed=committed,
        k_idx=k_idx,
        known_mask=known_mask,
        known_idx=known_idx,
        a_table=a_table,
    )
    expected_y = np.zeros((1, 3, 4), dtype=np.float32)
    expected_y[0, 0] = np.asarray(a_table[1])
    expected_y[0, 2] = np.asarray(a_table[4])
    np.testing.assert_allclose(np.asarray(y_out), expected_y)


def test_clamp_known_state_per_position_gathers_position_aware_anchor():
    P, V, d = 4, 3, 2
    a_table = jnp.asarray(
        np.arange(P * V * d, dtype=np.float32).reshape(P, V, d)
    )
    y = jnp.zeros((1, P, d))
    committed = jnp.zeros((1, P), dtype=jnp.bool_)
    k_idx = jnp.zeros((1, P), dtype=jnp.int32)
    # known_idx selects different labels at each position, all known.
    known_idx = jnp.asarray([[2, 0, 1, 2]], dtype=jnp.int32)
    known_mask = jnp.ones((1, P), dtype=jnp.bool_)
    y_out, _, _ = clamp_known_state(
        y=y,
        committed=committed,
        k_idx=k_idx,
        known_mask=known_mask,
        known_idx=known_idx,
        a_table=a_table,
    )
    expected = np.stack(
        [np.asarray(a_table[p, known_idx[0, p]]) for p in range(P)],
        axis=0,
    )[None, :, :]
    np.testing.assert_allclose(np.asarray(y_out), expected)


def test_clamp_known_state_per_position_rejects_wrong_site_size():
    P, V, d = 4, 3, 2
    a_table = jnp.zeros((P, V, d))
    y = jnp.zeros((1, 5, d))
    committed = jnp.zeros((1, 5), dtype=jnp.bool_)
    k_idx = jnp.zeros((1, 5), dtype=jnp.int32)
    known_mask = jnp.ones((1, 5), dtype=jnp.bool_)
    known_idx = jnp.zeros((1, 5), dtype=jnp.int32)
    with pytest.raises(ValueError, match="must equal"):
        clamp_known_state(
            y=y,
            committed=committed,
            k_idx=k_idx,
            known_mask=known_mask,
            known_idx=known_idx,
            a_table=a_table,
        )
