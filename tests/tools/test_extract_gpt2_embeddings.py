from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "tools" / "extract_gpt2_embeddings.py"
    spec = importlib.util.spec_from_file_location(
        "sticky_tools.extract_gpt2_embeddings", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_save_gpt2_wte_writes_expected_keys(tmp_path, monkeypatch):
    module = _load_module()

    def fake_load_wte(model_name: str = "gpt2"):
        return np.ones((50257, 768), dtype=np.float32), model_name

    monkeypatch.setattr(module, "_load_wte_from_hf", fake_load_wte)
    out = tmp_path / "gpt2_wte.npz"
    module.save_gpt2_wte(out_path=out, model_name="gpt2")

    with np.load(out) as data:
        assert data["wte"].shape == (50257, 768)
        assert data["wte"].dtype == np.float32
        assert str(data["model_name"]) == "gpt2"


def test_save_gpt2_wte_rejects_unexpected_shape(tmp_path, monkeypatch):
    module = _load_module()

    def fake_load_wte(model_name: str = "gpt2"):
        return np.ones((1000, 64), dtype=np.float32), model_name

    monkeypatch.setattr(module, "_load_wte_from_hf", fake_load_wte)
    out = tmp_path / "bad.npz"

    import pytest
    with pytest.raises(ValueError):
        module.save_gpt2_wte(
            out_path=out, model_name="gpt2",
            expected_vocab_size=50257, expected_dim=768,
        )
