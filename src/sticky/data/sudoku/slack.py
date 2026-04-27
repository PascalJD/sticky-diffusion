"""Constraint-slack site builder for Sudoku SJD.

Each slack site is a vector in R^9 whose v-th coordinate is the count of digit
v in a constraint group (row / column / 3x3 box). For valid solutions every
slack vector equals (1, 1, ..., 1). The 27 groups are ordered:

    0..8   : rows 0..8
    9..17  : columns 0..8
    18..26 : 3x3 boxes (row-major within the 3x3 box grid)
"""

from __future__ import annotations

import numpy as np


def _row_groups() -> np.ndarray:
    return np.arange(81, dtype=np.int32).reshape(9, 9)


def _col_groups() -> np.ndarray:
    return np.arange(81, dtype=np.int32).reshape(9, 9).T


def _box_groups() -> np.ndarray:
    cells = np.arange(81, dtype=np.int32).reshape(9, 9)
    groups = []
    for br in range(3):
        for bc in range(3):
            groups.append(
                cells[3 * br : 3 * br + 3, 3 * bc : 3 * bc + 3].reshape(-1)
            )
    return np.stack(groups, axis=0)


GROUPS = np.concatenate([_row_groups(), _col_groups(), _box_groups()], axis=0)
assert GROUPS.shape == (27, 9)


def compute_slack_vectors(solution_board: np.ndarray) -> np.ndarray:
    """Build slack count vectors from a batch of Sudoku solutions.

    Parameters
    ----------
    solution_board : (B, 81) int with digits in 1..9, row-major.

    Returns
    -------
    (B, 27, 9) float32 with slack[b, g, v-1] = count of digit v in group g.
    For a valid solution every entry is exactly 1.0.
    """
    board = np.asarray(solution_board, dtype=np.int32)
    if board.ndim != 2 or board.shape[1] != 81:
        raise ValueError(
            f"solution_board must have shape (B, 81), got {tuple(board.shape)}."
        )
    group_cells = board[:, GROUPS]  # (B, 27, 9) digits 1..9
    B = int(board.shape[0])
    slack = np.zeros((B, 27, 9), dtype=np.float32)
    for v in range(1, 10):
        slack[..., v - 1] = (group_cells == v).sum(axis=-1).astype(np.float32)
    return slack
