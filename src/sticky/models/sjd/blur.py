"""Spatially-blurred anchor-mean kernels for non-local SJD un-sticking.

Provides fixed (non-learnable) site-blending matrices W (shape (N, N)) used to
replace the SJD per-anchor center alpha(t)*e(a_i) with alpha(t)*mu_i(X_0),
where mu_i(X_0) = (W @ E(X_0))_i. Convolution closure of the VP-matched
sticky-jump kernel is preserved because the blur only changes the mean, not
the un-sticking variance.

Public API:
    gaussian_position_kernel(seq_len, sigma, include_self=True, dtype=jnp.float32)
    gaussian_2d_position_kernel(height, width, sigma, normalization="rowstoch", dtype=jnp.float32)
    sudoku_constraint_kernel(sigma, include_row=True, include_col=True, include_box=True, dtype=jnp.float32)
    sudoku_constraint_kernel_convex(sigma, rho, include_row=True, include_col=True, include_box=True, dtype=jnp.float32)
    blur_means(e, kernel)
    blurred_posterior_mean(probs_mean, committed_mask, committed_idx, a_table, kernel)
    kernel_fingerprint(kernel)
    build_blur_kernel(blur_cfg, seq_len=None, grid_shape=None)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "gaussian_position_kernel",
    "gaussian_2d_position_kernel",
    "sudoku_constraint_kernel",
    "sudoku_constraint_kernel_convex",
    "blur_means",
    "blurred_posterior_mean",
    "kernel_fingerprint",
    "build_blur_kernel",
]


def gaussian_position_kernel(
    seq_len: int,
    sigma: float,
    include_self: bool = True,
    dtype=jnp.float32,
) -> Array:
    """Row-rescaled 1D Gaussian kernel.

    Logits L[i, j] = -(i - j)^2 / (2 sigma^2). Behavior:
    - include_self=True (default): W = softmax(L, axis=-1); then
      W = W / W.diagonal()[:, None] so W[i, i] == 1 and rows do NOT sum to 1.
    - include_self=False: the diagonal of L is masked to -inf before softmax,
      so W[i, i] == 0 and rows sum to 1 (no rescale; softmax is row-stochastic).

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


def sudoku_constraint_kernel_convex(
    sigma: float,
    rho: float,
    include_row: bool = True,
    include_col: bool = True,
    include_box: bool = True,
    dtype=jnp.float32,
) -> Array:
    """Convex-blend constraint-graph kernel for the flat-81 Sudoku representation.

    Returns W = (1 - rho) * I + rho * B, where B is a row-stochastic matrix
    with B_ii = 0. B is constructed from the same constraint-neighbor support
    as sudoku_constraint_kernel, but uses softmax (without diagonal rescaling):

      Logits L[i, j] = -((r_i - r_j)^2 + (c_i - c_j)^2) / (2 sigma^2)
      L[i, j] = -inf for non-neighbors AND for i == j (diagonal excluded).
      B = softmax(L, axis=-1)   — row-stochastic, B_ii = 0.

    The convex-blend parameter rho controls the interpolation:
      - rho=0: W == I exactly (in IEEE float; 1*I + 0*B is bitwise identity).
      - rho=1: W == B (pure peer-averaging, zero self-weight).
      - Any rho in [0, 1]: rows of W sum to 1.

    Cell indexing and constraint groups are identical to sudoku_constraint_kernel:
      p = 9 * r + c, with groups row / col / box-exclusive (same 3x3 box AND
      different row AND different column). include_row/_col/_box flags toggle
      each group individually.

    Raises:
        ValueError: if sigma <= 0.
        ValueError: unless 0.0 <= rho <= 1.0.
        ValueError: if none of include_row, include_col, include_box is True
            (an empty-support kernel is meaningless; softmax over all-(-inf)
            logits produces NaN which the cleanup silently zeroes, yielding a
            sub-stochastic W = (1-rho)*I).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0.0, 1.0], got {rho}")
    if not (include_row or include_col or include_box):
        raise ValueError(
            "at least one of include_row, include_col, include_box must be True; "
            "an empty-support convex kernel is meaningless"
        )

    # KEEP-IN-SYNC with sudoku_constraint_kernel: the neighbor-mask construction
    # below (same_row / same_col / box_exclusive, include_* dispatch) is
    # intentionally duplicated from that function.  If the support logic there
    # changes, update this function to match.
    N = 81
    idx = jnp.arange(N)
    rows = idx // 9
    cols = idx % 9
    boxes = (rows // 3) * 3 + (cols // 3)

    same_row = rows[:, None] == rows[None, :]
    same_col = cols[:, None] == cols[None, :]
    same_box = boxes[:, None] == boxes[None, :]
    box_exclusive = same_box & ~same_row & ~same_col

    neighbor = jnp.zeros((N, N), dtype=jnp.bool_)
    if include_row:
        neighbor = neighbor | same_row
    if include_col:
        neighbor = neighbor | same_col
    if include_box:
        neighbor = neighbor | box_exclusive

    # Exclude the diagonal from the neighbor mask — B_ii must be 0.
    eye_mask = jnp.eye(N, dtype=jnp.bool_)
    neighbor = neighbor & ~eye_mask

    dr = (rows[:, None] - rows[None, :]).astype(dtype)
    dc = (cols[:, None] - cols[None, :]).astype(dtype)
    dist_sq = dr * dr + dc * dc
    logits = -dist_sq / (2.0 * sigma * sigma)
    # Mask diagonal and non-neighbors to -inf before softmax.
    logits = jnp.where(neighbor, logits, -jnp.inf)

    B = jax.nn.softmax(logits, axis=-1)
    # Off-support entries after softmax are exactly 0 (they received -inf
    # logits and are never selected); the where-cleanup enforces exact-zero
    # hygiene in case of any platform-specific floating-point edge cases.
    B = jnp.where(neighbor, B, jnp.zeros_like(B))

    W = (1.0 - rho) * jnp.eye(N, dtype=dtype) + rho * B
    return W.astype(dtype)


def gaussian_2d_position_kernel(
    height: int,
    width: int,
    sigma: float,
    normalization: str = "rowstoch",
    dtype=jnp.float32,
) -> Array:
    """Full (H*W, H*W) 2D Gaussian position kernel over a flat row-major pixel grid.

    Mirrors analysis/field_diag/common.gaussian_2d. Flat index p = r * width + c
    with (r, c) in [0, H) x [0, W). Logits L[i, j] = -((r_i-r_j)^2 + (c_i-c_j)^2)
    / (2 sigma^2), full softmax over all positions per row (no truncation):
      - normalization="rowstoch": W = softmax(L, axis=-1); rows sum to 1
        (convex local average over the grid, self included).
      - normalization="additive": W = softmax(L); then W = W / diag(W) so
        W[i, i] == 1 and rows do NOT sum to 1 (2D analogue of the 1D repo mode).

    Raises ValueError on sigma <= 0 or unknown normalization.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    H, W = int(height), int(width)
    idx = jnp.arange(H * W)
    r = (idx // W).astype(dtype)
    c = (idx % W).astype(dtype)
    d2 = (r[:, None] - r[None, :]) ** 2 + (c[:, None] - c[None, :]) ** 2  # (HW, HW)
    logits = -d2 / (2.0 * sigma * sigma)
    K = jax.nn.softmax(logits, axis=-1)  # row-stochastic
    if normalization == "additive":
        K = K / jnp.diagonal(K)[:, None]
    elif normalization != "rowstoch":
        raise ValueError(
            f"Unknown 2D normalization: {normalization!r} (expected 'rowstoch'/'additive')"
        )
    return K.astype(dtype)


def blur_means(
    e: Array,
    kernel: Array,
) -> Array:
    """Apply a fixed site-blending matrix to per-site embeddings.

    Args:
        e: either a 1D-site field of shape (B, N, d), or a 2D image field of
           shape (B, H, W, C, d) — in which case the kernel (N, N) with N = H*W
           is applied over the flat row-major pixel grid PER COLOR CHANNEL.
        kernel: shape (N, N).

    Returns:
        mu_bar with the same shape as `e`, where for the 1D case
            mu_bar[b, i, :] = sum_j kernel[i, j] * e[b, j, :],
        and for the 2D-image case the same contraction is applied over the
        flattened H*W positions independently for each channel c and feature d.
    """
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(
            f"kernel must be square (N, N); got shape {tuple(kernel.shape)}"
        )
    if e.ndim == 3:  # (B, N, d)
        if kernel.shape[0] != e.shape[1]:
            raise ValueError(
                f"kernel.shape[0] ({kernel.shape[0]}) must equal e.shape[1] "
                f"({e.shape[1]})"
            )
        return jnp.einsum("ij,bjd->bid", kernel, e)
    if e.ndim == 5:  # (B, H, W, C, d) image field
        B, H, W, C, d = e.shape
        if kernel.shape[0] != H * W:
            raise ValueError(
                f"kernel.shape[0] ({kernel.shape[0]}) must equal H*W "
                f"({H}*{W}={H * W}) for a 2D image field"
            )
        e2 = e.reshape(B, H * W, C, d)
        out = jnp.einsum("ij,bjcd->bicd", kernel, e2)
        return out.reshape(B, H, W, C, d)
    raise NotImplementedError(
        f"blur_means supports e.ndim in {{3, 5}}; got e.shape={tuple(e.shape)} "
        f"(ndim={e.ndim})."
    )


def blurred_posterior_mean(
    *,
    probs_mean: Array,
    committed_mask: Array,
    committed_idx: Array,
    a_table: Array,
    kernel: Array,
) -> Array:
    """Ehat = committed one-hot anchor embedding at committed sites, else the
    classifier posterior mean; returns blur_means(Ehat, kernel) = W @ Ehat.

    Shape-generic: probs_mean is (..., d) with committed_mask/committed_idx of
    shape probs_mean.shape[:-1]; supports the (B, N, d) and (B, H, W, C, d)
    layouts accepted by blur_means. The -1 uncommitted sentinel in
    committed_idx clips to 0 here; the bogus gathered anchor is discarded by
    the committed_mask select below.

    CALLER INVARIANT (not enforced — jit-compatible code cannot raise on
    traced values): every site with committed_mask=True must carry a VALID
    committed_idx >= 0. An inconsistent (True, -1) site silently blends
    anchor 0 into all its W-neighbors' means. The reverse sampler guarantees
    this invariant (committed sites always receive k_idx on commit /
    clamp_known_state); direct callers must too.
    """
    safe_idx = jnp.clip(committed_idx, 0, a_table.shape[0] - 1)
    committed_vec = a_table[safe_idx]
    e_hat = jnp.where(committed_mask[..., None], committed_vec, probs_mean)
    return blur_means(e_hat, kernel)


def kernel_fingerprint(kernel: Array | None) -> str | None:
    """sha256 hex of a 'float32:<shape>' header + row-major float32 bytes.

    None -> None. NOTE: hashes the kernel bytes as materialized on the current
    JAX backend; softmax/exp bit patterns may differ across backends, so
    fingerprints must only be compared between values produced on the same
    platform (rebuild-and-compare inside the same job).
    """
    if kernel is None:
        return None
    import hashlib

    import numpy as np

    arr = np.ascontiguousarray(
        np.asarray(jax.device_get(kernel), dtype=np.float32)
    )
    h = hashlib.sha256()
    h.update(f"float32:{tuple(arr.shape)}".encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def build_blur_kernel(
    blur_cfg, *, seq_len: int | None = None,
    grid_shape: tuple[int, int] | None = None,
) -> Array | None:
    """Dispatch a blur config to a concrete kernel.

    Args:
        blur_cfg: a dict-like (DictConfig or plain dict). Must expose
            `.get("enabled", False)` and `.get("kind", None)`. When enabled,
            kind-specific keys are read (sigma, include_self, include_row,
            include_col, include_box, normalization, rho).
        seq_len: required for kind == "gaussian_1d"; ignored otherwise.
        grid_shape: (H, W) required for kind == "gaussian_2d"; ignored otherwise.

    Normalization modes (key "normalization"; default "additive" when absent):
        "additive"  — legacy rescaled kernel (W_ii=1, rows do NOT sum to 1).
                      Bit-exact when the key is absent.
        "convex"    — convex blend W=(1-rho)*I + rho*B; rows sum to 1.
                      Only supported for kind="sudoku_constraint".
                      Controlled by "rho" (float, default 0.0).

    Returns:
        None if blur_cfg is None or blur_cfg["enabled"] is False, else an
        (N, N) kernel array.

    Raises:
        ValueError if seq_len is required but None, if kind is unknown, if
        normalization is unknown, or if convex normalization is requested for
        an unsupported kind.
    """
    if blur_cfg is None or not bool(blur_cfg.get("enabled", False)):
        return None

    kind = blur_cfg.get("kind", None)

    if kind is None:
        raise ValueError(
            "blur_cfg has enabled=True but 'kind' is not set (got None)"
        )

    sigma = float(blur_cfg.get("sigma", 1.0))
    raw_norm = blur_cfg.get("normalization", "additive")
    normalization = "additive" if raw_norm is None else str(raw_norm)

    # 2D image grid Gaussian (its own additive/rowstoch normalization semantics).
    if kind == "gaussian_2d":
        if grid_shape is None:
            raise ValueError("gaussian_2d blur requires grid_shape=(H, W), got None")
        H, W = int(grid_shape[0]), int(grid_shape[1])
        if H < 1 or W < 1:
            raise ValueError(f"gaussian_2d blur requires H,W >= 1, got {(H, W)}")
        if normalization not in ("additive", "rowstoch"):
            raise ValueError(
                f"gaussian_2d normalization must be 'additive' or 'rowstoch'; "
                f"got {normalization!r}"
            )
        return gaussian_2d_position_kernel(
            height=H, width=W, sigma=sigma, normalization=normalization,
        )

    if normalization == "additive":
        if kind == "gaussian_1d":
            if seq_len is None:
                raise ValueError("gaussian_1d blur requires seq_len, got None")
            if int(seq_len) < 1:
                raise ValueError(f"gaussian_1d blur requires seq_len >= 1, got {seq_len}")
            return gaussian_position_kernel(
                seq_len=int(seq_len),
                sigma=sigma,
                include_self=bool(blur_cfg.get("include_self", True)),
            )
        if kind == "sudoku_constraint":
            return sudoku_constraint_kernel(
                sigma=sigma,
                include_row=bool(blur_cfg.get("include_row", True)),
                include_col=bool(blur_cfg.get("include_col", True)),
                include_box=bool(blur_cfg.get("include_box", True)),
            )
        raise ValueError(f"Unknown blur kind: {kind!r}")

    if normalization == "convex":
        if kind != "sudoku_constraint":
            raise ValueError(
                f"convex normalization is only implemented for the sudoku_constraint "
                f"kernel; got kind={kind!r}"
            )
        rho = float(blur_cfg.get("rho", 0.0))
        return sudoku_constraint_kernel_convex(
            sigma=sigma,
            rho=rho,
            include_row=bool(blur_cfg.get("include_row", True)),
            include_col=bool(blur_cfg.get("include_col", True)),
            include_box=bool(blur_cfg.get("include_box", True)),
        )

    raise ValueError(f"Unknown blur normalization: {normalization!r}")
