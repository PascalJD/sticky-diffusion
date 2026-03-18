from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from sticky.models.sjd.anchors import (
    AnchorTableConfig,
    AnchorTransformConfig,
    anchor_learnable_from_mapping,
    anchor_table_config_from_mapping,
    build_anchor_table,
    build_anchor_table_views,
    build_anchor_table_from_config,
)


VOCAB_SIZE = 256


@pytest.mark.parametrize(
    ("config", "expected_dim"),
    [
        (
            AnchorTableConfig(
                family="ordered_scalar",
                vocab_size=VOCAB_SIZE,
                anchor_dim=1,
            ),
            1,
        ),
        (
            AnchorTableConfig(
                family="normal",
                vocab_size=VOCAB_SIZE,
                anchor_dim=8,
                seed=7,
            ),
            8,
        ),
        (
            AnchorTableConfig(
                family="ordered_normal",
                vocab_size=VOCAB_SIZE,
                anchor_dim=8,
                seed=7,
                order_weight=2.0,
                residual_weight=0.5,
            ),
            8,
        ),
        (
            AnchorTableConfig(
                family="thermometer",
                vocab_size=VOCAB_SIZE,
                anchor_dim=8,
                projection_seed=11,
            ),
            8,
        ),
    ],
)
def test_anchor_family_shapes(config: AnchorTableConfig, expected_dim: int):
    table = build_anchor_table(config)
    assert table.shape == (VOCAB_SIZE, expected_dim)


@pytest.mark.parametrize(
    "config",
    [
        AnchorTableConfig(
            family="ordered_scalar",
            vocab_size=VOCAB_SIZE,
            anchor_dim=1,
        ),
        AnchorTableConfig(
            family="normal",
            vocab_size=VOCAB_SIZE,
            anchor_dim=8,
            seed=17,
        ),
        AnchorTableConfig(
            family="ordered_normal",
            vocab_size=VOCAB_SIZE,
            anchor_dim=8,
            seed=17,
        ),
        AnchorTableConfig(
            family="thermometer",
            vocab_size=VOCAB_SIZE,
            anchor_dim=8,
            projection_seed=17,
        ),
    ],
)
def test_deterministic_anchor_builds_are_reproducible(config: AnchorTableConfig):
    first = build_anchor_table(config)
    second = build_anchor_table(config)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_row_norm_equalization_hits_target_norm():
    config = AnchorTableConfig(
        family="normal",
        vocab_size=VOCAB_SIZE,
        anchor_dim=16,
        seed=23,
        transform=AnchorTransformConfig(
            equalize_row_norms=True,
            target_row_norm=3.25,
        ),
    )

    table = build_anchor_table(config)
    row_norms = jnp.linalg.norm(table, axis=1)
    np.testing.assert_allclose(
        np.asarray(row_norms),
        np.full((VOCAB_SIZE,), 3.25, dtype=np.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def test_ordered_scalar_matches_legacy_values_exactly():
    table = build_anchor_table(
        AnchorTableConfig(
            family="ordered_scalar",
            vocab_size=VOCAB_SIZE,
            anchor_dim=1,
        )
    )
    expected = (
        -1.0
        + (
            2.0
            * jnp.arange(VOCAB_SIZE, dtype=jnp.float32)
            / jnp.asarray(255.0, dtype=jnp.float32)
        )
    )[:, None]
    np.testing.assert_array_equal(np.asarray(table), np.asarray(expected))


def test_build_anchor_table_from_config_parses_transform_block():
    cfg = {
        "anchor_init": "normal",
        "anchor_dim": 4,
        "anchor_seed": 31,
        "anchor_transform": {
            "equalize_row_norms": True,
            "target_row_norm": 2.0,
        },
    }

    table = build_anchor_table_from_config(cfg, vocab_size=VOCAB_SIZE)
    assert table.shape == (VOCAB_SIZE, 4)
    row_norms = jnp.linalg.norm(table, axis=1)
    np.testing.assert_allclose(
        np.asarray(row_norms),
        np.full((VOCAB_SIZE,), 2.0, dtype=np.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def test_build_anchor_table_views_separates_prescale_and_final_tables():
    config = AnchorTableConfig(
        family="normal",
        vocab_size=VOCAB_SIZE,
        anchor_dim=8,
        seed=5,
        transform=AnchorTransformConfig(
            center_columns=True,
            scale=3.0,
        ),
    )

    views = build_anchor_table_views(config)
    np.testing.assert_allclose(
        np.asarray(views.final),
        3.0 * np.asarray(views.transformed),
        atol=1e-6,
        rtol=1e-6,
    )
    assert views.raw.shape == (VOCAB_SIZE, 8)
    assert views.transformed.shape == (VOCAB_SIZE, 8)
    assert views.final.shape == (VOCAB_SIZE, 8)


def test_ordered_scalar_scale_applies_in_final_view_only():
    config = AnchorTableConfig(
        family="ordered_scalar",
        vocab_size=VOCAB_SIZE,
        anchor_dim=1,
        transform=AnchorTransformConfig(scale=2.5),
    )

    views = build_anchor_table_views(config)
    np.testing.assert_allclose(
        np.asarray(views.transformed),
        np.asarray(views.raw),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(views.final),
        2.5 * np.asarray(views.raw),
        atol=1e-6,
        rtol=1e-6,
    )


def test_nested_anchor_config_takes_precedence_and_logs_warning(caplog):
    cfg = {
        "anchor": {
            "family": "thermometer",
            "dim": 8,
            "learnable": False,
        },
        "anchor_init": "normal",
        "anchor_dim": 16,
        "learnable_anchors": True,
    }

    with caplog.at_level("WARNING"):
        parsed = anchor_table_config_from_mapping(cfg, vocab_size=VOCAB_SIZE)
        learnable = anchor_learnable_from_mapping(cfg)

    assert parsed.family == "thermometer"
    assert parsed.anchor_dim == 8
    assert learnable is False
    assert "nested `model.anchor.*` config" in caplog.text


def test_flat_anchor_config_still_parses():
    cfg = {
        "anchor_init": "ordered_normal",
        "anchor_dim": 6,
        "anchors_init_std": 0.25,
        "anchor_seed": 9,
        "anchor_order_weight": 2.0,
        "anchor_residual_weight": 0.75,
        "learnable_anchors": False,
        "anchor_transform": {
            "equalize_row_norms": True,
            "target_row_norm": 1.5,
        },
    }

    parsed = anchor_table_config_from_mapping(cfg, vocab_size=VOCAB_SIZE)
    assert parsed.family == "ordered_normal"
    assert parsed.anchor_dim == 6
    assert parsed.init_std == pytest.approx(0.25)
    assert parsed.seed == 9
    assert parsed.order_weight == pytest.approx(2.0)
    assert parsed.residual_weight == pytest.approx(0.75)
    assert parsed.transform.equalize_row_norms is True
    assert parsed.transform.target_row_norm == pytest.approx(1.5)
    assert anchor_learnable_from_mapping(cfg) is False


def test_legacy_anchor_family_aliases_normalize_to_canonical_names():
    ordered_cfg = {
        "anchor_init": "ordered_random_residual",
        "anchor_dim": 6,
        "anchor_seed": 9,
    }
    thermometer_cfg = {
        "anchor_init": "thermometer_projected",
        "anchor_dim": 6,
        "anchor_projection_seed": 11,
    }

    ordered = anchor_table_config_from_mapping(ordered_cfg, vocab_size=VOCAB_SIZE)
    thermometer = anchor_table_config_from_mapping(
        thermometer_cfg,
        vocab_size=VOCAB_SIZE,
    )

    assert ordered.family == "ordered_normal"
    assert thermometer.family == "thermometer"
