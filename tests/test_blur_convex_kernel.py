"""Tests for the convex blur-kernel mode (W(rho) = (1-rho)*I + rho*B).

Validates:
  T1 - kernel properties (row sums, zero diagonal of B, off-support zeros,
       shape/dtype) for rho in {0.0, 0.1, 0.35, 0.5, 1.0}.
  T2 - rho=0 exact identity (bit-exact, not just allclose).
  T3 - include_* flags correctly trim the support.
  T4 - legacy path bit-exactness + error cases.

CPU-only, no GPU required.
"""
from __future__ import annotations

import numpy as np
import pytest

from sticky.models.sjd.blur import (
    build_blur_kernel,
    gaussian_position_kernel,
    sudoku_constraint_kernel,
    sudoku_constraint_kernel_convex,
)


# ---------------------------------------------------------------------------
# T1 — kernel properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho", [0.0, 0.1, 0.35, 0.5, 1.0])
def test_kernel_properties(rho):
    """Rows sum to 1, B has zero diagonal, off-support entries are exactly 0,
    shape is (81, 81), dtype is float32."""
    W = np.asarray(sudoku_constraint_kernel_convex(sigma=1.5, rho=rho))

    # Shape and dtype
    assert W.shape == (81, 81), f"shape mismatch: {W.shape}"
    assert W.dtype == np.float32, f"dtype mismatch: {W.dtype}"

    # Rows sum to 1
    row_sums = W.sum(axis=-1)
    np.testing.assert_allclose(
        row_sums,
        np.ones(81, dtype=np.float32),
        atol=1e-6,
        err_msg=f"rows do not sum to 1 for rho={rho}",
    )

    # B has zero diagonal: W(1.0) is B, and B_ii == 0
    B = np.asarray(sudoku_constraint_kernel_convex(sigma=1.5, rho=1.0))
    diag_B = np.diag(B)
    np.testing.assert_array_equal(
        diag_B,
        np.zeros(81, dtype=np.float32),
        err_msg="B (rho=1.0) should have zero diagonal",
    )

    # Off-support entries are exactly 0.
    # Each cell has exactly 20 peers (8 row + 8 col + 4 box-exclusive).
    # For rho in (0,1): W = (1-rho)*I + rho*B, so diagonal is non-zero
    # (= 1-rho) and off-support non-diagonal entries must be exactly 0.
    # For rho=0: W==I, off-diagonal all 0.
    # For rho=1.0: W==B, off-support all 0, diagonal all 0.
    if rho == 0.0:
        # Pure identity: everything off-diagonal is 0
        off_diag = W - np.diag(np.diag(W))
        assert np.all(off_diag == 0.0), "rho=0: off-diagonal entries should all be 0"
    else:
        # Build the expected support mask:
        # support[i,j] = 1 if i==j or j is a constraint-neighbor of i
        idx = np.arange(81)
        rows_grid = idx // 9
        cols_grid = idx % 9
        boxes_grid = (rows_grid // 3) * 3 + (cols_grid // 3)

        same_row = rows_grid[:, None] == rows_grid[None, :]
        same_col = cols_grid[:, None] == cols_grid[None, :]
        same_box = boxes_grid[:, None] == boxes_grid[None, :]
        box_excl = same_box & ~same_row & ~same_col

        neighbor = same_row | same_col | box_excl
        eye_mask = np.eye(81, dtype=bool)
        if rho < 1.0:
            support = neighbor | eye_mask  # diagonal stays non-zero
        else:
            support = neighbor  # rho=1 → B: diagonal must be zero, no self-edge

        off_support = ~support
        assert np.all(W[off_support] == 0.0), (
            f"rho={rho}: off-support entries are not exactly 0"
        )

    # Nonzero count per row check
    # rho in (0,1): 20 off-diagonal peers + 1 diagonal = 21 nonzeros
    # rho=1.0: 20 off-diagonal peers, diagonal=0 → 20 nonzeros
    # rho=0.0: 1 nonzero (the diagonal) per row
    if rho == 0.0:
        for i in range(81):
            assert np.count_nonzero(W[i]) == 1, (
                f"rho=0: row {i} has {np.count_nonzero(W[i])} nonzeros, expected 1"
            )
    elif rho == 1.0:
        for i in range(81):
            assert np.count_nonzero(W[i]) == 20, (
                f"rho=1.0: row {i} has {np.count_nonzero(W[i])} nonzeros, expected 20"
            )
    else:
        for i in range(81):
            assert np.count_nonzero(W[i]) == 21, (
                f"rho={rho}: row {i} has {np.count_nonzero(W[i])} nonzeros, expected 21"
            )


# ---------------------------------------------------------------------------
# T2 — rho=0 exact identity
# ---------------------------------------------------------------------------

def test_rho0_exact_identity():
    """W(rho=0) must be bit-exactly equal to the 81x81 identity matrix."""
    W = np.asarray(sudoku_constraint_kernel_convex(sigma=1.5, rho=0.0))
    assert np.array_equal(W, np.eye(81, dtype=np.float32)), (
        "W(rho=0) is not bit-exactly equal to eye(81, dtype=float32)"
    )


# ---------------------------------------------------------------------------
# T3 — include_* support flags
# ---------------------------------------------------------------------------

def test_include_row_false_drops_row_peers():
    """include_row=False: each cell has 12 constraint peers (8 col + 4 box)."""
    W = np.asarray(
        sudoku_constraint_kernel_convex(
            sigma=1.5, rho=1.0, include_row=False, include_col=True, include_box=True
        )
    )
    # rho=1 → B; diagonal==0, so nonzero count == peer count
    for i in range(81):
        nz = np.count_nonzero(W[i])
        assert nz == 12, (
            f"include_row=False: row {i} has {nz} nonzeros, expected 12"
        )


def test_include_col_false_drops_col_peers():
    """include_col=False: each cell has 12 constraint peers (8 row + 4 box)."""
    W = np.asarray(
        sudoku_constraint_kernel_convex(
            sigma=1.5, rho=1.0, include_row=True, include_col=False, include_box=True
        )
    )
    for i in range(81):
        nz = np.count_nonzero(W[i])
        assert nz == 12, (
            f"include_col=False: row {i} has {nz} nonzeros, expected 12"
        )


def test_include_box_false_drops_box_peers():
    """include_box=False: each cell has 16 constraint peers (8 row + 8 col)."""
    W = np.asarray(
        sudoku_constraint_kernel_convex(
            sigma=1.5, rho=1.0, include_row=True, include_col=True, include_box=False
        )
    )
    for i in range(81):
        nz = np.count_nonzero(W[i])
        assert nz == 16, (
            f"include_box=False: row {i} has {nz} nonzeros, expected 16"
        )


# ---------------------------------------------------------------------------
# T4 — legacy bit-exactness and error cases
# ---------------------------------------------------------------------------

def test_legacy_sudoku_no_normalization_key():
    """build_blur_kernel without 'normalization' key must be bit-identical to
    sudoku_constraint_kernel(sigma=1.5)."""
    cfg = {
        "enabled": True,
        "kind": "sudoku_constraint",
        "sigma": 1.5,
        "include_row": True,
        "include_col": True,
        "include_box": True,
    }
    W_built = np.asarray(build_blur_kernel(cfg))
    W_ref = np.asarray(sudoku_constraint_kernel(sigma=1.5))
    assert np.array_equal(W_built, W_ref), (
        "build_blur_kernel (no normalization key) != sudoku_constraint_kernel"
    )


def test_legacy_sudoku_explicit_additive():
    """build_blur_kernel with normalization='additive' must be bit-identical to
    sudoku_constraint_kernel(sigma=1.5)."""
    cfg = {
        "enabled": True,
        "kind": "sudoku_constraint",
        "sigma": 1.5,
        "include_row": True,
        "include_col": True,
        "include_box": True,
        "normalization": "additive",
    }
    W_built = np.asarray(build_blur_kernel(cfg))
    W_ref = np.asarray(sudoku_constraint_kernel(sigma=1.5))
    assert np.array_equal(W_built, W_ref), (
        "build_blur_kernel (normalization='additive') != sudoku_constraint_kernel"
    )


def test_legacy_gaussian_1d_no_normalization_key():
    """build_blur_kernel gaussian_1d without normalization key is bit-identical
    to gaussian_position_kernel(seq_len=16, sigma=1.0)."""
    cfg = {
        "enabled": True,
        "kind": "gaussian_1d",
        "sigma": 1.0,
    }
    W_built = np.asarray(build_blur_kernel(cfg, seq_len=16))
    W_ref = np.asarray(gaussian_position_kernel(seq_len=16, sigma=1.0))
    assert np.array_equal(W_built, W_ref), (
        "build_blur_kernel gaussian_1d (no normalization key) != gaussian_position_kernel"
    )


def test_unknown_normalization_raises():
    """normalization='bogus' must raise ValueError."""
    cfg = {
        "enabled": True,
        "kind": "sudoku_constraint",
        "sigma": 1.5,
        "normalization": "bogus",
    }
    with pytest.raises(ValueError, match="bogus"):
        build_blur_kernel(cfg)


def test_convex_on_gaussian_1d_raises():
    """normalization='convex' with kind='gaussian_1d' must raise ValueError."""
    cfg = {
        "enabled": True,
        "kind": "gaussian_1d",
        "sigma": 1.0,
        "normalization": "convex",
        "rho": 0.5,
    }
    with pytest.raises(ValueError):
        build_blur_kernel(cfg, seq_len=16)


def test_rho_below_range_raises():
    """rho < 0.0 must raise ValueError."""
    with pytest.raises(ValueError):
        sudoku_constraint_kernel_convex(sigma=1.5, rho=-0.1)


def test_rho_above_range_raises():
    """rho > 1.0 must raise ValueError."""
    with pytest.raises(ValueError):
        sudoku_constraint_kernel_convex(sigma=1.5, rho=1.5)


def test_sigma_nonpositive_raises():
    """sigma <= 0 must raise ValueError."""
    with pytest.raises(ValueError, match="sigma must be positive"):
        sudoku_constraint_kernel_convex(sigma=0.0, rho=0.5)
    with pytest.raises(ValueError, match="sigma must be positive"):
        sudoku_constraint_kernel_convex(sigma=-1.0, rho=0.5)
