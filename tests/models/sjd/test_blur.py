from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sticky.models.sjd.blur import gaussian_position_kernel


def test_gaussian_position_kernel_small_sigma_is_identity():
    """sigma -> 0 with self-inclusion should recover I_L within 1e-3."""
    L = 16
    W = gaussian_position_kernel(L, sigma=1e-3, include_self=True)
    np.testing.assert_allclose(np.asarray(W), np.eye(L), atol=1e-3)


def test_gaussian_position_kernel_diagonal_is_one():
    """Every diagonal entry must be exactly 1 after rescaling."""
    W = gaussian_position_kernel(32, sigma=1.5, include_self=True)
    diag = np.asarray(jnp.diagonal(W))
    np.testing.assert_allclose(diag, np.ones(32), atol=1e-6)


def test_gaussian_position_kernel_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        gaussian_position_kernel(16, sigma=0.0)
    with pytest.raises(ValueError):
        gaussian_position_kernel(16, sigma=-1.0)


def test_gaussian_position_kernel_no_self_is_finite_and_row_stochastic():
    """include_self=False must NOT produce NaN/Inf. Diagonal is zero by design;
    rows sum to 1 because softmax is row-stochastic and no rescale is applied."""
    W = gaussian_position_kernel(8, sigma=1.0, include_self=False)
    W_np = np.asarray(W)
    assert np.isfinite(W_np).all(), "include_self=False produced non-finite entries"
    np.testing.assert_allclose(np.diagonal(W_np), np.zeros(8), atol=1e-6)
    np.testing.assert_allclose(W_np.sum(axis=-1), np.ones(8), atol=1e-6)
