"""No-op lemma for uniform-within-group Sudoku constraint kernels.

Pins the central design claim of src/sticky/models/sjd/blur.py (see the
"NOTE: The non-uniformity within each constraint group is essential..."
docstring of sudoku_constraint_kernel): on VALID Sudoku boards a
uniform-within-group constraint kernel is an information no-op, while the
Gaussian-weighted kernel transmits board-dependent constraint information.

Setup: E[i] = a_table[digit(i)] for flat cell i (a_table is (9, d)), and
mu = B @ E for a row-stochastic peer-averaging matrix B. Let
S = a_table.sum(axis=0). Because every row/col/box of a valid board is a
permutation of {0..8}, the embedding sum over any constraint group is the
board-independent S. Consequences pinned here:

(a) Row+col uniform B (16 peers, weight 1/16 each, no box):
        mu_i = S/8 - E_i/8
    exactly — affine in E_i with board-independent coefficients, so the
    kernel transmits ZERO board information beyond the cell's own digit.
(b) Full uniform B (20 peers, weight 1/20 each): with R_i / C_i the
    embedding sums over (box(i) ∩ row(i)) / (box(i) ∩ col(i)) — both 3-cell
    line segments including cell i itself —
        mu_i = (3S - E_i - R_i - C_i)/20
    exactly: the only board-dependent content beyond E_i is the box-line
    intersection sums R_i + C_i.
(c) Gaussian-weighted B (sudoku_constraint_kernel_convex(sigma=1.5, rho=1.0)):
    mu is NOT reducible to the affine form — the residual of the best
    board-independent affine fit mu ~ alpha*S + beta*E (least squares over
    all cells of all boards jointly) is bounded away from zero, while the
    same fit on the uniform B16 is exact (paired control).

The support masks are built here in numpy directly from rows/cols/boxes,
independently of the repo's kernel-construction code — that independence is
the point. Only the (c) kernel comes from the repo. CPU-only, all RNGs seeded.

Measured magnitudes (float64, d=7, 8 boards): (a) max abs err ~3e-16;
(b) max abs err ~6e-16; affine-fit residual RMS: B16 ~4e-16,
B20 ~6.3e-2, Gaussian ~1.2e-1.
"""
from __future__ import annotations

import numpy as np
import pytest

from sticky.models.sjd.blur import sudoku_constraint_kernel_convex

# ---------------------------------------------------------------------------
# Valid-board generator
# ---------------------------------------------------------------------------

BOARD_SEEDS = list(range(8))
EMBED_DIM = 7
A_TABLE_SEED = 1234


def _base_grid() -> np.ndarray:
    """Cyclic base grid, digits 0..8; valid Sudoku by construction."""
    r = np.arange(9)[:, None]
    c = np.arange(9)[None, :]
    return (3 * (r % 3) + r // 3 + c) % 9


def assert_valid_board(board: np.ndarray) -> None:
    """Assert each row, column and 3x3 box is a permutation of 0..8."""
    assert board.shape == (9, 9)
    full = np.arange(9)
    for r in range(9):
        assert np.array_equal(np.sort(board[r, :]), full), f"row {r} invalid"
    for c in range(9):
        assert np.array_equal(np.sort(board[:, c]), full), f"col {c} invalid"
    for br in range(3):
        for bc in range(3):
            box = board[3 * br : 3 * br + 3, 3 * bc : 3 * bc + 3].ravel()
            assert np.array_equal(np.sort(box), full), f"box ({br},{bc}) invalid"


def random_valid_board(seed: int) -> np.ndarray:
    """Seeded random valid Sudoku board (digits 0..8).

    Starts from the cyclic base grid and composes validity-preserving
    transforms: digit-label relabeling, row-band + within-band row
    permutations, column-stack + within-stack column permutations, and an
    optional transpose.
    """
    rng = np.random.default_rng(seed)
    board = _base_grid()

    # Relabel the 9 digit labels.
    board = rng.permutation(9)[board]

    # Permute the 3 row-bands, and the rows within each band.
    band_order = rng.permutation(3)
    row_order = np.concatenate([3 * b + rng.permutation(3) for b in band_order])
    board = board[row_order, :]

    # Permute the 3 column-stacks, and the columns within each stack.
    stack_order = rng.permutation(3)
    col_order = np.concatenate([3 * s + rng.permutation(3) for s in stack_order])
    board = board[:, col_order]

    # Optional transpose.
    if rng.integers(2):
        board = board.T

    assert_valid_board(board)
    return board


# ---------------------------------------------------------------------------
# Support masks (built independently of the repo's kernel code)
# ---------------------------------------------------------------------------

def _support_masks():
    """(row_peers, col_peers, box_exclusive_peers) as (81, 81) bool,
    diagonal False everywhere."""
    idx = np.arange(81)
    rows = idx // 9
    cols = idx % 9
    boxes = (rows // 3) * 3 + (cols // 3)

    same_row = rows[:, None] == rows[None, :]
    same_col = cols[:, None] == cols[None, :]
    same_box = boxes[:, None] == boxes[None, :]
    eye = np.eye(81, dtype=bool)

    row_peers = same_row & ~eye
    col_peers = same_col & ~eye
    box_exclusive = same_box & ~same_row & ~same_col
    return row_peers, col_peers, box_exclusive


ROW_PEERS, COL_PEERS, BOX_EXCL_PEERS = _support_masks()

# Uniform-within-group kernels.
B16 = (ROW_PEERS | COL_PEERS).astype(np.float64) / 16.0
B20 = (ROW_PEERS | COL_PEERS | BOX_EXCL_PEERS).astype(np.float64) / 20.0


def _a_table() -> np.ndarray:
    rng = np.random.default_rng(A_TABLE_SEED)
    return rng.standard_normal((9, EMBED_DIM))


def _embeddings(board: np.ndarray, a_table: np.ndarray) -> np.ndarray:
    """E of shape (81, d): E[i] = a_table[digit(i)] in flat order p = 9r + c."""
    return a_table[board.ravel()]


def _box_line_sums(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, C of shape (81, d): per-cell embedding sums over box∩row / box∩col,
    each a 3-cell line segment including the cell itself."""
    idx = np.arange(81)
    rows = idx // 9
    cols = idx % 9
    boxes = (rows // 3) * 3 + (cols // 3)
    same_row = rows[:, None] == rows[None, :]
    same_col = cols[:, None] == cols[None, :]
    same_box = boxes[:, None] == boxes[None, :]
    # Boolean masks act as 0/1 summation matrices.
    R = (same_box & same_row).astype(np.float64) @ E
    C = (same_box & same_col).astype(np.float64) @ E
    return R, C


def _affine_fit_residual_rms(B: np.ndarray, boards, a_table: np.ndarray) -> float:
    """RMS residual of the best board-independent affine fit
    mu ~ alpha*S + beta*E, with scalars (alpha, beta) solved by least
    squares jointly over all cells of all boards (and all embedding dims)."""
    S = a_table.sum(axis=0)
    mu_chunks, e_chunks = [], []
    for board in boards:
        E = _embeddings(board, a_table)
        mu_chunks.append(np.asarray(B, dtype=np.float64) @ E)
        e_chunks.append(E)
    mu_all = np.concatenate(mu_chunks, axis=0)  # (n_boards*81, d)
    E_all = np.concatenate(e_chunks, axis=0)
    S_all = np.broadcast_to(S, E_all.shape)
    X = np.stack([S_all.ravel(), E_all.ravel()], axis=1)
    y = mu_all.ravel()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float(np.sqrt(np.mean(resid**2)))


# ---------------------------------------------------------------------------
# Generator sanity
# ---------------------------------------------------------------------------

def test_board_generator_produces_valid_distinct_boards():
    boards = [random_valid_board(s) for s in BOARD_SEEDS]
    for board in boards:
        assert_valid_board(board)
    # The randomization must actually do something: boards pairwise distinct.
    flat = {tuple(b.ravel().tolist()) for b in boards}
    assert len(flat) == len(boards), "seeded boards are not pairwise distinct"


def test_support_mask_peer_counts():
    """Sanity on the independently-built masks: 8 row + 8 col + 4 box-exclusive
    peers per cell, diagonal False, groups pairwise disjoint."""
    assert np.all(ROW_PEERS.sum(axis=1) == 8)
    assert np.all(COL_PEERS.sum(axis=1) == 8)
    assert np.all(BOX_EXCL_PEERS.sum(axis=1) == 4)
    eye = np.eye(81, dtype=bool)
    for mask in (ROW_PEERS, COL_PEERS, BOX_EXCL_PEERS):
        assert not np.any(mask & eye)
    assert not np.any(ROW_PEERS & COL_PEERS)
    assert not np.any(ROW_PEERS & BOX_EXCL_PEERS)
    assert not np.any(COL_PEERS & BOX_EXCL_PEERS)
    assert np.all((ROW_PEERS | COL_PEERS).sum(axis=1) == 16)
    assert np.all((ROW_PEERS | COL_PEERS | BOX_EXCL_PEERS).sum(axis=1) == 20)


# ---------------------------------------------------------------------------
# (a) Row+col uniform kernel is an exact board-independent affine map
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", BOARD_SEEDS)
def test_uniform_rowcol_kernel_is_affine_noop(seed):
    """B16 (16 peers, weight 1/16): mu_i = S/8 - E_i/8 exactly on every valid
    board — zero residual, zero board information beyond the cell's digit."""
    a_table = _a_table()
    S = a_table.sum(axis=0)
    board = random_valid_board(seed)
    E = _embeddings(board, a_table)
    mu = B16 @ E
    assert np.allclose(mu, S / 8.0 - E / 8.0, atol=1e-5)


# ---------------------------------------------------------------------------
# (b) Full uniform kernel transmits only the box-line intersection sums
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", BOARD_SEEDS)
def test_uniform_full_kernel_closed_form(seed):
    """B20 (20 peers, weight 1/20): mu_i = (3S - E_i - R_i - C_i)/20 exactly,
    where R_i / C_i are the box∩row / box∩col 3-cell line sums (incl. i)."""
    a_table = _a_table()
    S = a_table.sum(axis=0)
    board = random_valid_board(seed)
    E = _embeddings(board, a_table)
    R, C = _box_line_sums(E)
    mu = B20 @ E
    assert np.allclose(mu, (3.0 * S - E - R - C) / 20.0, atol=1e-5)


# ---------------------------------------------------------------------------
# (c) Gaussian-weighted kernel transmits board-dependent information
# ---------------------------------------------------------------------------

def test_gaussian_kernel_not_reducible_to_affine():
    """The actual convex kernel (sigma=1.5, rho=1.0) is NOT a board-independent
    affine map of E: the joint least-squares affine fit mu ~ alpha*S + beta*E
    leaves a residual bounded away from zero. Paired control inside the same
    test: the identical fit on B16 is exact (RMS < 1e-6).

    Measured (float64, d=7, 8 boards): Gaussian RMS ~1.2e-1, B16 RMS ~4e-16.
    """
    a_table = _a_table()
    boards = [random_valid_board(s) for s in BOARD_SEEDS]

    B_gauss = np.asarray(sudoku_constraint_kernel_convex(sigma=1.5, rho=1.0))
    rms_gauss = _affine_fit_residual_rms(B_gauss, boards, a_table)
    rms_b16 = _affine_fit_residual_rms(B16, boards, a_table)

    assert rms_gauss > 1e-3, (
        f"Gaussian kernel affine-fit residual RMS {rms_gauss:.3e} should be "
        f"bounded away from zero (> 1e-3)"
    )
    assert rms_b16 < 1e-6, (
        f"control: uniform B16 affine-fit residual RMS {rms_b16:.3e} should "
        f"be ~0 (< 1e-6)"
    )


def test_b20_transmits_exactly_the_box_line_sums():
    """B20 demonstrates 'transmits only the box-line sums': the affine-only fit
    (alpha*S + beta*E) leaves a residual > 1e-3, while the closed form
    including R and C is exact (~0) on every board.

    Measured (float64, d=7, 8 boards): affine-only RMS ~6.3e-2.
    """
    a_table = _a_table()
    S = a_table.sum(axis=0)
    boards = [random_valid_board(s) for s in BOARD_SEEDS]

    # Affine-only fit cannot explain B20's output...
    rms_affine = _affine_fit_residual_rms(B20, boards, a_table)
    assert rms_affine > 1e-3, (
        f"B20 affine-only fit residual RMS {rms_affine:.3e} should be > 1e-3"
    )

    # ...but the closed form including the box-line sums R, C is exact.
    for board in boards:
        E = _embeddings(board, a_table)
        R, C = _box_line_sums(E)
        mu = B20 @ E
        closed = (3.0 * S - E - R - C) / 20.0
        resid_rms = float(np.sqrt(np.mean((mu - closed) ** 2)))
        assert resid_rms < 1e-10, (
            f"B20 closed-form residual RMS {resid_rms:.3e} should be ~0"
        )
