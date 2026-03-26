from __future__ import annotations

import numpy as np

from sticky.eval.sudoku import _board_from_sequence, _evaluate_batch_counts, valid_solution


def _solved_board_triples() -> np.ndarray:
    triples = np.zeros((81, 3), dtype=np.int32)
    idx = 0
    for row in range(9):
        for col in range(9):
            triples[idx] = (row, col, ((row * 3 + row // 3 + col) % 9) + 1)
            idx += 1
    return triples


def test_strict_solve_metric_requires_exact_reconstructed_board():
    triples = _solved_board_triples()
    pred_triples = triples.copy()
    pred_triples[:, 2] = (pred_triples[:, 2] % 9) + 1

    pred_seq = pred_triples.reshape(1, -1)
    puzzle_sol = _board_from_sequence(triples.reshape(-1))[None, :]

    assert valid_solution(pred_seq[0]) is True
    assert np.array_equal(_board_from_sequence(pred_seq[0]), puzzle_sol[0]) is False

    counts = _evaluate_batch_counts(
        pred_seq=pred_seq,
        puzzle_sol=puzzle_sol,
        start_index=np.zeros((1, 1), dtype=np.int32),
        input_seq=pred_seq,
    )

    assert counts["valid_complete"] == 1
    assert counts["board_exact"] == 0
    assert counts["strict_complete"] == 0


def test_batch_counts_report_token_and_duplicate_coordinate_diagnostics():
    triples = _solved_board_triples()
    pred_triples = triples.copy()
    pred_triples[79, 1] = pred_triples[78, 1]
    pred_triples[80, 0] = (pred_triples[80, 0] + 1) % 9
    pred_triples[80, 2] = (pred_triples[80, 2] % 9) + 1

    pred_seq = pred_triples.reshape(1, -1)
    input_seq = triples.reshape(1, -1)
    puzzle_sol = _board_from_sequence(triples.reshape(-1))[None, :]

    counts = _evaluate_batch_counts(
        pred_seq=pred_seq,
        puzzle_sol=puzzle_sol,
        start_index=np.asarray([[78]], dtype=np.int32),
        input_seq=input_seq,
    )

    assert counts["total_pred"] == 3
    assert counts["row_token_correct"] == 2
    assert counts["col_token_correct"] == 2
    assert counts["value_token_correct"] == 2
    assert counts["rowcol_correct_total"] == 1
    assert counts["value_given_correct_rowcol"] == 1
    assert counts["duplicate_coordinate_total"] == 1
