from __future__ import annotations

import numpy as np

from sticky.data.sudoku.boards import (
    SUDOKU_NUM_CELLS,
    SUDOKU_TABLE_WIDTH,
    build_board_batch,
)


def test_build_board_batch_includes_sjd_fields():
    # Build a valid Shah-table row: column 0 = start_index, then 81 * 4 cells.
    row = np.zeros((1, SUDOKU_TABLE_WIDTH), dtype=np.int32)
    start = 25
    row[0, 0] = start
    # Fill 81 quadruples: (r, c, v, extra), using digit 5 for values.
    for j in range(SUDOKU_NUM_CELLS):
        cell_id = j
        row[0, 1 + 4 * j + 0] = cell_id // 9
        row[0, 1 + 4 * j + 1] = cell_id % 9
        row[0, 1 + 4 * j + 2] = 5  # any valid digit in 1..9
        row[0, 1 + 4 * j + 3] = 0

    batch = build_board_batch(row)
    assert set(batch) == {"solution_board", "clue_board", "clue_mask", "start_index"}
    np.testing.assert_array_equal(
        batch["solution_board"],
        np.full((1, SUDOKU_NUM_CELLS), 5, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch["clue_board"][0, :start],
        np.full(start, 5, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch["clue_board"][0, start:],
        np.zeros(SUDOKU_NUM_CELLS - start, dtype=np.int32),
    )
