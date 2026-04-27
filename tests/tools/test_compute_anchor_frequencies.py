"""Unit tests for tools/compute_anchor_frequencies.compute_log_w."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "compute_anchor_frequencies.py"
)


def _load_compute_log_w():
    spec = importlib.util.spec_from_file_location(
        "compute_anchor_frequencies", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_log_w


def test_uniform_p0_yields_zero_log_w():
    """Uniform empirical distribution -> w(a) = -log(1/K) is constant; after
    mean-normalization w == 1 and log_w == 0 elementwise."""
    compute_log_w = _load_compute_log_w()
    K = 16
    counts = np.full((K,), 100, dtype=np.int64)
    blob = compute_log_w(
        counts,
        smoothing_alpha=1.0,
        normalize_mean=True,
        clip_min=None,
        clip_max=None,
    )
    np.testing.assert_allclose(blob["w"], np.ones(K, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(blob["log_w"], np.zeros(K, dtype=np.float32), atol=1e-6)


def test_p0_sums_to_one():
    """The reported p0 is a valid probability."""
    compute_log_w = _load_compute_log_w()
    rng = np.random.default_rng(42)
    counts = rng.integers(0, 1000, size=(32,), endpoint=True).astype(np.int64)
    counts[5] = 0  # force at least one anchor with zero raw count
    blob = compute_log_w(counts, smoothing_alpha=1.0, normalize_mean=False, clip_min=None, clip_max=None)
    assert np.all(blob["p0"] > 0.0), "smoothing failed: zero p0 entries remain"
    np.testing.assert_allclose(float(blob["p0"].sum()), 1.0, atol=1e-9)


def test_clip_bounds_w():
    """clip_min / clip_max bound the weight array, not log_w directly."""
    compute_log_w = _load_compute_log_w()
    K = 8
    counts = np.array([1, 1, 1, 1, 1_000_000, 1_000_000, 1_000_000, 1_000_000], dtype=np.int64)
    blob = compute_log_w(
        counts,
        smoothing_alpha=1.0,
        normalize_mean=False,
        clip_min=0.5,
        clip_max=2.0,
    )
    w = blob["w"]
    assert float(w.min()) >= 0.5 - 1e-6
    assert float(w.max()) <= 2.0 + 1e-6


def test_normalize_mean_makes_w_mean_one():
    """With clip disabled, mean(w) should be exactly 1 after normalization."""
    compute_log_w = _load_compute_log_w()
    rng = np.random.default_rng(7)
    counts = rng.integers(1, 5000, size=(64,), endpoint=True).astype(np.int64)
    blob = compute_log_w(
        counts,
        smoothing_alpha=1.0,
        normalize_mean=True,
        clip_min=None,
        clip_max=None,
    )
    np.testing.assert_allclose(float(blob["w"].mean()), 1.0, atol=1e-6)


def test_zero_smoothing_with_zero_count_raises():
    """Without smoothing, anchors with zero count have p0 = 0 and would
    produce -inf weights; the function should reject this loudly."""
    compute_log_w = _load_compute_log_w()
    counts = np.array([10, 0, 5], dtype=np.int64)
    with pytest.raises(ValueError):
        compute_log_w(counts, smoothing_alpha=0.0)
