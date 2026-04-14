from __future__ import annotations

import numpy as np

from sticky.data.sudoku import build_solver_rank
from sticky.data.sudoku.boards import (
    SUDOKU_NUM_CELLS,
    SUDOKU_TABLE_WIDTH,
    build_board_batch,
)


def _make_triples(batch_size: int, permutations: list[np.ndarray]) -> np.ndarray:
    assert len(permutations) == batch_size
    triples = np.zeros((batch_size, SUDOKU_NUM_CELLS, 3), dtype=np.int32)
    for i, perm in enumerate(permutations):
        assert perm.shape == (SUDOKU_NUM_CELLS,)
        # perm[j] is the cell id (0..80) at solver position j.
        rows = perm // 9
        cols = perm % 9
        vals = np.full(SUDOKU_NUM_CELLS, 5, dtype=np.int32)  # any legal digit 1..9
        triples[i, :, 0] = rows
        triples[i, :, 1] = cols
        triples[i, :, 2] = vals
    return triples


def test_clue_cells_have_zero_rank():
    perm = np.arange(SUDOKU_NUM_CELLS, dtype=np.int32)
    triples = _make_triples(1, [perm])
    start_index = np.asarray([30], dtype=np.int32)
    rank = build_solver_rank(triples, start_index=start_index)
    # Cells listed in positions 0..29 of the triples are clues -> rank must be 0.
    clue_cell_ids = perm[:30]
    np.testing.assert_array_equal(rank[0, clue_cell_ids], np.zeros(30, dtype=np.float32))


def test_unknown_ranks_span_zero_to_one_in_solver_order():
    perm = np.arange(SUDOKU_NUM_CELLS, dtype=np.int32)
    triples = _make_triples(1, [perm])
    start_index = np.asarray([20], dtype=np.int32)
    rank = build_solver_rank(triples, start_index=start_index)

    # Unknowns are at triple positions 20..80 -> 61 cells -> ranks [0, 1/60, ..., 1].
    unknown_cells = perm[20:]
    got = rank[0, unknown_cells]
    expected = np.linspace(0.0, 1.0, num=SUDOKU_NUM_CELLS - 20, dtype=np.float32)
    np.testing.assert_allclose(got, expected, atol=1e-6)
    # First unknown is exactly 0, last is exactly 1.
    assert float(got[0]) == 0.0
    assert float(got[-1]) == 1.0


def test_rank_follows_permuted_solver_order():
    # Non-trivial permutation: verify rank tracks the triples array slot, not the cell id.
    rng = np.random.default_rng(0)
    perm = rng.permutation(SUDOKU_NUM_CELLS).astype(np.int32)
    triples = _make_triples(1, [perm])
    start = 10
    start_index = np.asarray([start], dtype=np.int32)
    rank = build_solver_rank(triples, start_index=start_index)

    n_unknown = SUDOKU_NUM_CELLS - start
    for pos in range(n_unknown):
        cell_id = int(perm[start + pos])
        expected = pos / (n_unknown - 1)
        assert abs(float(rank[0, cell_id]) - expected) < 1e-6


def test_single_unknown_rank_is_zero():
    perm = np.arange(SUDOKU_NUM_CELLS, dtype=np.int32)
    triples = _make_triples(1, [perm])
    start_index = np.asarray([SUDOKU_NUM_CELLS - 1], dtype=np.int32)
    rank = build_solver_rank(triples, start_index=start_index)
    # Exactly one unknown cell: rank clamps to 0 (no span to normalize across).
    assert float(rank[0, int(perm[-1])]) == 0.0


def test_build_board_batch_includes_solver_rank():
    # Build a valid Shah-table row: column 0 = start_index, then 81 * 4 = 324 columns.
    row = np.zeros((1, SUDOKU_TABLE_WIDTH), dtype=np.int32)
    start = 25
    row[0, 0] = start
    # Fill 81 quadruples: (r, c, v, extra). Use the identity permutation and digit 5 for vals.
    for j in range(SUDOKU_NUM_CELLS):
        cell_id = j
        row[0, 1 + 4 * j + 0] = cell_id // 9
        row[0, 1 + 4 * j + 1] = cell_id % 9
        row[0, 1 + 4 * j + 2] = 5  # any valid digit in 1..9
        row[0, 1 + 4 * j + 3] = 0

    batch = build_board_batch(row)
    assert "solver_rank" in batch
    rank = np.asarray(batch["solver_rank"])
    assert rank.shape == (1, SUDOKU_NUM_CELLS)
    assert rank.dtype == np.float32
    # Clue cells (ids 0..24) get rank 0; unknown cells (ids 25..80) span 0..1.
    np.testing.assert_array_equal(rank[0, :start], np.zeros(start, dtype=np.float32))
    np.testing.assert_allclose(
        rank[0, start:],
        np.linspace(0.0, 1.0, num=SUDOKU_NUM_CELLS - start, dtype=np.float32),
        atol=1e-6,
    )
