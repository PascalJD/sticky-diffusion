"""Tests for the 2D-grid convex_mix blur-kernel mode.

    W(rho, sigma) = (1 - rho) * I + rho * K_sigma,
    K_sigma = row-softmax(-d^2 / (2 sigma^2))  (rowstoch, self included).

Validates:
  T1 - kernel properties (row sums, non-negativity, self-weight >= 1-rho,
       shape/dtype) for rho in {0.0, 0.05, 0.15, 0.30, 1.0}.
  T2 - rho=1 bitwise identity to the rowstoch kernel (array + fingerprint),
       the strict-containment endpoint (gate G1's invariant).
  T3 - rho=0 exact identity (bit-exact, not just allclose; gate G2's builder
       invariant — training configs reject rho=0, see T7).
  T4 - builder dispatch: dict round-trip bitwise-equals the direct call;
       missing rho / wrong-kind / unknown-normalization error cases.
  T5 - parameter range errors (rho outside [0,1], sigma <= 0).
  T6 - DictConfig round-trip via the yaml file.
  T7 - task-build-time rejection of rho=0 / missing-rho training configs
       (tasks/factory._validate_image_blur_cfg).

CPU-only, no GPU required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]

from sticky.models.sjd.blur import (
    build_blur_kernel,
    gaussian_2d_position_kernel,
    gaussian_2d_position_kernel_convex_mix,
    kernel_fingerprint,
)
from sticky.tasks.factory import _validate_image_blur_cfg

H, W = 32, 32
N = H * W


# ---------------------------------------------------------------------------
# T1 — kernel properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho", [0.0, 0.05, 0.15, 0.30, 1.0])
def test_kernel_properties(rho):
    """Rows sum to 1, entries are non-negative, self-weight >= 1-rho,
    shape is (1024, 1024), dtype is float32."""
    Wk = np.asarray(gaussian_2d_position_kernel_convex_mix(H, W, sigma=1.5, rho=rho))

    assert Wk.shape == (N, N), f"shape mismatch: {Wk.shape}"
    assert Wk.dtype == np.float32, f"dtype mismatch: {Wk.dtype}"

    # Rows sum to 1 (convex combination of two row-stochastic matrices).
    row_sums = Wk.astype(np.float64).sum(axis=-1)
    np.testing.assert_allclose(
        row_sums,
        np.ones(N),
        atol=1e-5,
        err_msg=f"rows do not sum to 1 for rho={rho}",
    )

    # Non-negativity.
    assert np.all(Wk >= 0.0), f"negative entries for rho={rho}"

    # Self-weight floor: W[i, i] = (1-rho) + rho*K[i, i] >= 1 - rho.
    diag = np.diag(Wk).astype(np.float64)
    assert np.all(diag >= (1.0 - rho) - 1e-6), (
        f"self-weight below 1-rho={1.0 - rho} for rho={rho}: min={diag.min()}"
    )


# ---------------------------------------------------------------------------
# T2 — rho=1 bitwise identity to rowstoch (strict containment endpoint)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [1.0, 1.5, 2.5])
def test_rho1_bitwise_equals_rowstoch(sigma):
    Wmix = gaussian_2d_position_kernel_convex_mix(H, W, sigma=sigma, rho=1.0)
    Krow = gaussian_2d_position_kernel(H, W, sigma=sigma, normalization="rowstoch")
    assert np.array_equal(np.asarray(Wmix), np.asarray(Krow)), (
        f"rho=1 not bitwise-identical to rowstoch at sigma={sigma}"
    )
    assert kernel_fingerprint(Wmix) == kernel_fingerprint(Krow)


# ---------------------------------------------------------------------------
# T3 — rho=0 exact identity (builder only; training configs reject it, T7)
# ---------------------------------------------------------------------------

def test_rho0_exact_identity():
    Wk = np.asarray(gaussian_2d_position_kernel_convex_mix(H, W, sigma=1.5, rho=0.0))
    assert np.array_equal(Wk, np.eye(N, dtype=np.float32)), (
        "rho=0 is not bit-exact identity"
    )


# ---------------------------------------------------------------------------
# T4 — builder dispatch and error cases
# ---------------------------------------------------------------------------

def test_builder_dispatch_bitwise():
    cfg = {
        "enabled": True,
        "kind": "gaussian_2d",
        "normalization": "convex_mix",
        "rho": 0.15,
        "sigma": 1.5,
    }
    Wb = build_blur_kernel(cfg, grid_shape=(H, W))
    Wd = gaussian_2d_position_kernel_convex_mix(H, W, sigma=1.5, rho=0.15)
    assert np.array_equal(np.asarray(Wb), np.asarray(Wd))
    assert kernel_fingerprint(Wb) == kernel_fingerprint(Wd)


def test_builder_missing_rho_raises():
    cfg = {
        "enabled": True,
        "kind": "gaussian_2d",
        "normalization": "convex_mix",
        "sigma": 1.5,
    }
    with pytest.raises(ValueError, match="explicit 'rho'"):
        build_blur_kernel(cfg, grid_shape=(H, W))


def test_convex_mix_on_gaussian_1d_raises():
    cfg = {
        "enabled": True,
        "kind": "gaussian_1d",
        "normalization": "convex_mix",
        "rho": 0.15,
        "sigma": 1.5,
    }
    with pytest.raises(ValueError, match="only implemented for the gaussian_2d"):
        build_blur_kernel(cfg, seq_len=64)


def test_convex_mix_on_sudoku_raises():
    cfg = {
        "enabled": True,
        "kind": "sudoku_constraint",
        "normalization": "convex_mix",
        "rho": 0.15,
        "sigma": 1.5,
    }
    with pytest.raises(ValueError, match="only implemented for the gaussian_2d"):
        build_blur_kernel(cfg)


def test_convex_on_gaussian_2d_raises():
    """The sudoku-only 'convex' mode stays rejected for the 2D grid."""
    cfg = {
        "enabled": True,
        "kind": "gaussian_2d",
        "normalization": "convex",
        "rho": 0.15,
        "sigma": 1.5,
    }
    with pytest.raises(ValueError, match="'additive', 'rowstoch' or 'convex_mix'"):
        build_blur_kernel(cfg, grid_shape=(H, W))


def test_legacy_gaussian_2d_paths_unchanged():
    """rowstoch/additive dispatch is bit-exact with and without the new mode."""
    for norm in ("rowstoch", "additive"):
        cfg = {"enabled": True, "kind": "gaussian_2d", "sigma": 1.5,
               "normalization": norm}
        Wb = build_blur_kernel(cfg, grid_shape=(H, W))
        Wd = gaussian_2d_position_kernel(H, W, sigma=1.5, normalization=norm)
        assert np.array_equal(np.asarray(Wb), np.asarray(Wd))


# ---------------------------------------------------------------------------
# T5 — parameter range errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho", [-0.01, 1.01])
def test_rho_out_of_range_raises(rho):
    with pytest.raises(ValueError, match="rho must be in"):
        gaussian_2d_position_kernel_convex_mix(H, W, sigma=1.5, rho=rho)


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_sigma_nonpositive_raises(sigma):
    with pytest.raises(ValueError, match="sigma must be positive"):
        gaussian_2d_position_kernel_convex_mix(H, W, sigma=sigma, rho=0.15)


# ---------------------------------------------------------------------------
# T6 — DictConfig round-trip via the yaml file
# ---------------------------------------------------------------------------

def test_dictconfig_roundtrip_yaml():
    yaml_path = REPO_ROOT / "config" / "forward" / "blur" / "gaussian_2d_convex_mix.yaml"
    cfg = OmegaConf.load(yaml_path)
    assert bool(cfg.enabled) is True
    assert str(cfg.kind) == "gaussian_2d"
    assert str(cfg.normalization) == "convex_mix"
    Wb = build_blur_kernel(cfg, grid_shape=(H, W))
    Wd = gaussian_2d_position_kernel_convex_mix(
        H, W, sigma=float(cfg.sigma), rho=float(cfg.rho)
    )
    assert np.array_equal(np.asarray(Wb), np.asarray(Wd))


# ---------------------------------------------------------------------------
# T7 — task-build-time rejection of the rho=0 / missing-rho training configs
# ---------------------------------------------------------------------------

def _mix_cfg(**kw):
    base = {"enabled": True, "kind": "gaussian_2d",
            "normalization": "convex_mix", "sigma": 1.5}
    base.update(kw)
    return OmegaConf.create(base)


def test_factory_rejects_rho0():
    with pytest.raises(ValueError, match=r"rho in \(0, 1\]"):
        _validate_image_blur_cfg(_mix_cfg(rho=0.0))


def test_factory_rejects_missing_rho():
    with pytest.raises(ValueError, match=r"rho in \(0, 1\]"):
        _validate_image_blur_cfg(_mix_cfg())


def test_factory_rejects_rho_above_one():
    with pytest.raises(ValueError, match=r"rho in \(0, 1\]"):
        _validate_image_blur_cfg(_mix_cfg(rho=1.5))


@pytest.mark.parametrize("rho", [0.05, 0.15, 0.30, 1.0])
def test_factory_accepts_positive_rho(rho):
    _validate_image_blur_cfg(_mix_cfg(rho=rho))  # must not raise


def test_factory_accepts_rowstoch_and_disabled():
    _validate_image_blur_cfg(OmegaConf.create(
        {"enabled": True, "kind": "gaussian_2d", "normalization": "rowstoch",
         "sigma": 1.5}
    ))
    _validate_image_blur_cfg(OmegaConf.create({"enabled": False}))
    _validate_image_blur_cfg(None)


def test_factory_rejects_non_gaussian2d_kind():
    with pytest.raises(ValueError, match="only\\s+forward.blur.kind='gaussian_2d'"):
        _validate_image_blur_cfg(OmegaConf.create(
            {"enabled": True, "kind": "sudoku_constraint"}
        ))
