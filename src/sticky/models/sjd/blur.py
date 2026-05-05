"""Spatially-blurred anchor-mean kernels for non-local SJD un-sticking.

Provides fixed (non-learnable) site-blending matrices W (shape (N, N)) used to
replace the SJD per-anchor center alpha(t)*e(a_i) with alpha(t)*mu_i(X_0),
where mu_i(X_0) = (W @ E(X_0))_i. Convolution closure of the VP-matched
sticky-jump kernel is preserved because the blur only changes the mean, not
the un-sticking variance.

Public API:
    gaussian_position_kernel(seq_len, sigma, include_self=True, dtype=jnp.float32)
    (more added in subsequent tasks)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "gaussian_position_kernel",
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
