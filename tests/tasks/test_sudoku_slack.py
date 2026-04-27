from __future__ import annotations

import numpy as np
import pytest

from sticky.data.sudoku import compute_slack_vectors
from sticky.data.sudoku.slack import GROUPS


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
