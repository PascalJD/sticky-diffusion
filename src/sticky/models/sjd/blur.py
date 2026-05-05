"""Spatially-blurred anchor-mean kernels for non-local SJD un-sticking.

Provides fixed (non-learnable) site-blending matrices W (shape (N, N)) used to
replace the SJD per-anchor center alpha(t)*e(a_i) with alpha(t)*mu_i(X_0),
where mu_i(X_0) = (W @ E(X_0))_i. Convolution closure of the VP-matched
sticky-jump kernel is preserved because the blur only changes the mean, not
the un-sticking variance.

Public API:
    gaussian_position_kernel(seq_len, sigma, include_self=True, dtype=jnp.float32)
    sudoku_constraint_kernel(sigma, include_row=True, include_col=True, include_box=True, dtype=jnp.float32)
    blur_means(e, kernel)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "gaussian_position_kernel",
    "sudoku_constraint_kernel",
    "blur_means",
]


def gaussian_position_kernel(
    seq_len: int,
    sigma: float,
    include_self: bool = True,
    dtype=jnp.float32,
) -> Array:
    """Row-rescaled 1D Gaussian kernel.

    Logits L[i, j] = -(i - j)^2 / (2 sigma^2). If not include_self: mask
    diagonal with -inf before softmax. W = softmax(L, axis=-1); then
    W = W / W.diagonal()[:, None] so W[i, i] == 1 and rows do NOT sum to 1.

    Raises ValueError on sigma <= 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    idx = jnp.arange(seq_len, dtype=dtype)
    diff_sq = (idx[None, :] - idx[:, None]) ** 2  # (L, L)
    logits = -diff_sq / (2.0 * sigma * sigma)

    if not include_self:
        eye_mask = jnp.eye(seq_len, dtype=jnp.bool_)
        logits = jnp.where(eye_mask, -jnp.inf, logits)

    W = jax.nn.softmax(logits, axis=-1)
    if include_self:
        W = W / jnp.diagonal(W)[:, None]
    return W.astype(dtype)


def sudoku_constraint_kernel(
    sigma: float,
    include_row: bool = True,
    include_col: bool = True,
    include_box: bool = True,
    dtype=jnp.float32,
) -> Array:
    """Constraint-graph kernel for the flat-81 Sudoku representation.

    Cells are flat-indexed p = 9 * r + c with (r, c) in [0, 9) x [0, 9).
    Cells i, j are CONSTRAINT-NEIGHBORS iff they share at least one of:
      - row: r_i == r_j
      - column: c_i == c_j
      - box (exclusive): same 3x3 box AND different row AND different column
    (with the include_row/_col/_box flags toggling each group individually).

    The box group is exclusive of row and column pairs so that toggling
    include_row=False correctly zeros out pure-row pairs (which would otherwise
    remain connected via the shared box for cells in the same row-within-box).

    Within the non-zero support, weights follow the 2D Gaussian on (r, c):
      L[i, j] = -((r_i - r_j)^2 + (c_i - c_j)^2) / (2 sigma^2)
      L[i, j] = -inf for non-neighbors (and i != j).
    Diagonal is always included (W[i, i] == 1 after rescaling).

    NOTE: The non-uniformity within each constraint group is essential.
    A uniform-within-group kernel collapses to a no-op on valid Sudokus
    because each row/col/box is a permutation of {1,...,9}, so the
    averaged anchor embedding is invariant across cells in that group.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    N = 81
    idx = jnp.arange(N)
    rows = idx // 9
    cols = idx % 9
    boxes = (rows // 3) * 3 + (cols // 3)

    same_row = rows[:, None] == rows[None, :]
    same_col = cols[:, None] == cols[None, :]
    same_box = boxes[:, None] == boxes[None, :]
    # Box-exclusive: cells that share a box but do NOT share a row or column.
    # Row- and col-sharing pairs are already captured by the row/col groups;
    # using only the exclusive interior avoids double-counting and ensures
    # that toggling include_row=False zeros out pure-row pairs.
    box_exclusive = same_box & ~same_row & ~same_col

    neighbor = jnp.zeros((N, N), dtype=jnp.bool_)
    if include_row:
        neighbor = neighbor | same_row
    if include_col:
        neighbor = neighbor | same_col
    if include_box:
        neighbor = neighbor | box_exclusive

    # Always allow the diagonal so we can rescale W[i,i] -> 1.
    eye_mask = jnp.eye(N, dtype=jnp.bool_)
    neighbor = neighbor | eye_mask

    dr = (rows[:, None] - rows[None, :]).astype(dtype)
    dc = (cols[:, None] - cols[None, :]).astype(dtype)
    dist_sq = dr * dr + dc * dc
    logits = -dist_sq / (2.0 * sigma * sigma)
    logits = jnp.where(neighbor, logits, -jnp.inf)

    W = jax.nn.softmax(logits, axis=-1)
    W = W / jnp.diagonal(W)[:, None]
    # Zero out positions that were masked (-inf) — softmax + division can
    # leave subnormal numerical residue; explicit zero is cleaner.
    W = jnp.where(neighbor, W, jnp.zeros_like(W))
    return W.astype(dtype)


def blur_means(
    e: Array,
    kernel: Array,
) -> Array:
    """Apply a fixed site-blending matrix to per-site embeddings.

    Args:
        e: shape (B, N, d) — only 1D site_shape supported in phase 1.
            Multi-D site shapes (e.g., 2D images) raise NotImplementedError;
            see the prompt for the ImageNet/CIFAR follow-up.
        kernel: shape (N, N).

    Returns:
        mu_bar of shape (B, N, d), where
            mu_bar[b, i, :] = sum_j kernel[i, j] * e[b, j, :].
    """
    if e.ndim != 3:
        raise NotImplementedError(
            f"blur_means only supports 1D site_shape (e.ndim == 3); "
            f"got e.shape={tuple(e.shape)} (ndim={e.ndim}). "
            f"Multi-D site shapes are deferred."
        )
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(
            f"kernel must be square (N, N); got shape {tuple(kernel.shape)}"
        )
    if kernel.shape[0] != e.shape[1]:
        raise ValueError(
            f"kernel.shape[0] ({kernel.shape[0]}) must equal e.shape[1] "
            f"({e.shape[1]})"
        )
    return jnp.einsum("ij,bjd->bid", kernel, e)
