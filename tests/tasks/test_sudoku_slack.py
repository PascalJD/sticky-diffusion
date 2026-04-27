from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from sticky.data.sudoku import compute_slack_vectors
from sticky.data.sudoku.slack import GROUPS, compute_slack_from_cells


def _valid_solution() -> np.ndarray:
    """Build a valid 9x9 Sudoku solution via the standard cyclic construction."""
    board = np.zeros((9, 9), dtype=np.int32)
    for r in range(9):
        for c in range(9):
            board[r, c] = ((r * 3 + r // 3 + c) % 9) + 1
    return board.reshape(1, 81)


def test_groups_constant_has_expected_shape_and_partition():
    assert GROUPS.shape == (27, 9)
    # Rows partition the 81 cells.
    rows = GROUPS[:9].reshape(-1)
    np.testing.assert_array_equal(np.sort(rows), np.arange(81))
    # Columns partition the 81 cells.
    cols = GROUPS[9:18].reshape(-1)
    np.testing.assert_array_equal(np.sort(cols), np.arange(81))
    # Boxes partition the 81 cells.
    boxes = GROUPS[18:27].reshape(-1)
    np.testing.assert_array_equal(np.sort(boxes), np.arange(81))


def test_compute_slack_vectors_on_valid_solution_is_all_ones():
    solution = _valid_solution()
    slack = compute_slack_vectors(solution)
    assert slack.shape == (1, 27, 9)
    assert slack.dtype == np.float32
    np.testing.assert_array_equal(slack, np.ones((1, 27, 9), dtype=np.float32))


def test_compute_slack_vectors_batch_independence():
    sol_a = _valid_solution()
    # A constant board: all cells == 5. Then every group has 9 fives and zero
    # of every other digit.
    sol_b = np.full((1, 81), 5, dtype=np.int32)
    batch = np.concatenate([sol_a, sol_b], axis=0)
    slack = compute_slack_vectors(batch)
    np.testing.assert_array_equal(slack[0], np.ones((27, 9), dtype=np.float32))
    expected_b = np.zeros((27, 9), dtype=np.float32)
    expected_b[:, 4] = 9.0  # digit 5 -> column 4
    np.testing.assert_array_equal(slack[1], expected_b)


def test_compute_slack_vectors_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(B, 81\)"):
        compute_slack_vectors(np.zeros((3, 80), dtype=np.int32))


def test_compute_slack_from_cells_simplex_vertices_returns_ones():
    """Cells set to one-hot of a valid Sudoku solution must project back to
    the all-ones slack tensor (each group has exactly one of each digit)."""
    solution = _valid_solution()  # (1, 81)
    digits = solution[0] - 1  # (81,) in 0..8
    one_hot = np.eye(9, dtype=np.float32)[digits]  # (81, 9)
    cell_state = one_hot[None, :, :]  # (1, 81, 9)
    slack = compute_slack_from_cells(cell_state)
    assert slack.shape == (1, 27, 9)
    np.testing.assert_array_equal(
        np.asarray(slack), np.ones((1, 27, 9), dtype=np.float32)
    )


def test_compute_slack_from_cells_works_on_jax_arrays():
    solution = _valid_solution()
    digits = solution[0] - 1
    one_hot = np.eye(9, dtype=np.float32)[digits]
    cell_state = jnp.asarray(one_hot[None, :, :])
    slack = compute_slack_from_cells(cell_state)
    np.testing.assert_array_equal(
        np.asarray(slack), np.ones((1, 27, 9), dtype=np.float32)
    )


def test_compute_slack_from_cells_is_linear_in_cell_state():
    """sum over a group is linear: T(2x) = 2 T(x)."""
    rng = np.random.default_rng(0)
    cell_state = rng.standard_normal((3, 81, 9)).astype(np.float32)
    s_x = compute_slack_from_cells(cell_state)
    s_2x = compute_slack_from_cells(2.0 * cell_state)
    np.testing.assert_allclose(np.asarray(s_2x), 2.0 * np.asarray(s_x), rtol=1e-5)


def test_compute_slack_from_cells_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(B, 81, 9\)"):
        compute_slack_from_cells(np.zeros((1, 81, 8), dtype=np.float32))


def test_iterator_emits_slack_x0(monkeypatch):
    """SudokuBoardBatchIterator returns a per-batch slack_x0 of shape (B,27,9)."""
    from sticky.data.sudoku import boards as boards_mod

    rng = np.random.default_rng(0)
    rows = np.zeros((4, boards_mod.SUDOKU_TABLE_WIDTH), dtype=np.int32)
    sol = _valid_solution()[0]  # (81,)
    for b in range(4):
        rows[b, 0] = 30  # any valid start_index
        for j in range(81):
            cell_id = j
            rows[b, 1 + 4 * j + 0] = cell_id // 9
            rows[b, 1 + 4 * j + 1] = cell_id % 9
            rows[b, 1 + 4 * j + 2] = int(sol[cell_id])
            rows[b, 1 + 4 * j + 3] = 0

    batch = boards_mod.build_board_batch(rows)
    slack = compute_slack_vectors(batch["solution_board"])
    assert slack.shape == (4, 27, 9)
    np.testing.assert_array_equal(slack, np.ones((4, 27, 9), dtype=np.float32))
