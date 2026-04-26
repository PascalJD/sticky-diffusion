from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from sticky.models.sjd.anchors import (
    AnchorTableConfig,
    anchor_table_config_from_mapping,
    build_anchor_table,
)


def _write_fake_wte(tmp_path: Path, vocab: int, dim: int) -> Path:
    arr = (
        np.arange(vocab * dim, dtype=np.float32).reshape(vocab, dim)
        / float(vocab * dim)
    )
    p = tmp_path / "wte.npz"
    np.savez(p, wte=arr)
    return p


def test_pretrained_family_loads_npz(tmp_path):
    path = _write_fake_wte(tmp_path, vocab=256, dim=64)
    cfg = AnchorTableConfig(
        family="pretrained",
        vocab_size=256,
        anchor_dim=64,
        pretrained_path=str(path),
    )
    table = build_anchor_table(cfg)
    assert table.shape == (256, 64)
    assert jnp.isfinite(table).all()


def test_pretrained_family_rejects_shape_mismatch(tmp_path):
    path = _write_fake_wte(tmp_path, vocab=100, dim=16)
    cfg = AnchorTableConfig(
        family="pretrained",
        vocab_size=256,
        anchor_dim=64,
        pretrained_path=str(path),
    )
    with pytest.raises(ValueError):
        build_anchor_table(cfg)


def test_pretrained_family_requires_path():
    cfg = AnchorTableConfig(
        family="pretrained",
        vocab_size=256,
        anchor_dim=64,
        pretrained_path=None,
    )
    with pytest.raises(ValueError):
        build_anchor_table(cfg)


def test_pretrained_family_missing_file(tmp_path):
    cfg = AnchorTableConfig(
        family="pretrained",
        vocab_size=256,
        anchor_dim=64,
        pretrained_path=str(tmp_path / "does_not_exist.npz"),
    )
    with pytest.raises(FileNotFoundError):
        build_anchor_table(cfg)


def test_anchor_table_config_from_mapping_reads_pretrained_path(tmp_path):
    path = _write_fake_wte(tmp_path, vocab=256, dim=64)
    model_cfg = {
        "vocab_size": 256,
        "anchor": {
            "family": "pretrained",
            "dim": 64,
            "pretrained_path": str(path),
        },
    }
    cfg = anchor_table_config_from_mapping(model_cfg)
    assert cfg.family == "pretrained"
    assert cfg.pretrained_path == str(path)
    table = build_anchor_table(cfg)
    assert table.shape == (256, 64)
