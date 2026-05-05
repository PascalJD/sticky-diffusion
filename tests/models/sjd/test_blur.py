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


from sticky.models.sjd.blur import sudoku_constraint_kernel


def test_sudoku_kernel_small_sigma_is_identity():
    W = sudoku_constraint_kernel(sigma=1e-3)
    np.testing.assert_allclose(np.asarray(W), np.eye(81), atol=1e-3)


def test_sudoku_kernel_no_share_is_zero():
    """Cells (0,0) and (8,8) share no row/col/box -> W[0, 80] == 0."""
    W = sudoku_constraint_kernel(sigma=1.5)
    assert float(W[0, 80]) == 0.0


def test_sudoku_kernel_row_neighbor_nonzero():
    """Cells (0,0) and (0,1) share row 0 -> W[0, 1] > 0."""
    W = sudoku_constraint_kernel(sigma=1.5)
    assert float(W[0, 1]) > 0.0


def test_sudoku_kernel_col_neighbor_nonzero():
    """Cells (0,0) and (1,0) share col 0 -> W[0, 9] > 0."""
    W = sudoku_constraint_kernel(sigma=1.5)
    assert float(W[0, 9]) > 0.0


def test_sudoku_kernel_box_only_nonzero():
    """Cells (0,0) and (1,1) share top-left box but no row/col -> W[0, 10] > 0."""
    W = sudoku_constraint_kernel(sigma=1.5)
    assert float(W[0, 10]) > 0.0


def test_sudoku_kernel_distance_monotone_within_row():
    """Closer column dominates: W[0, 1] > W[0, 8] (both row-mates of (0,0))."""
    W = sudoku_constraint_kernel(sigma=1.5)
    assert float(W[0, 1]) > float(W[0, 8])


def test_sudoku_kernel_disable_row_zeros_row_neighbor():
    """include_row=False -> W[0, 1] == 0 (no longer share any active group)."""
    W = sudoku_constraint_kernel(sigma=1.5, include_row=False)
    assert float(W[0, 1]) == 0.0
