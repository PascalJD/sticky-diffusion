from __future__ import annotations

import numpy as np


VALID_SEQ_ORDERS = {"dataset", "fixed", "random"}


def _sort_row_col_val(triples: np.ndarray) -> np.ndarray:
    if triples.shape[0] <= 1:
        return triples
    keys = (triples[:, 2], triples[:, 1], triples[:, 0])
    return triples[np.lexsort(keys)]


def to_inputs(
    *,
    triples: np.ndarray,
    start_index: np.ndarray,
    seq_order: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if seq_order == "dataset":
        return triples.reshape(triples.shape[0], -1)

    transformed = np.empty_like(triples)
    for i in range(triples.shape[0]):
        k = int(start_index[i])
        if (k < 0) or (k > 81):
            raise ValueError(f"Invalid Sudoku start_index={k}; expected 0..81.")

        prefix = triples[i, :k]
        suffix = triples[i, k:]

        if seq_order == "fixed":
            prefix = _sort_row_col_val(prefix)
            suffix = _sort_row_col_val(suffix)
        elif seq_order == "random":
            if prefix.shape[0] > 1:
                prefix = prefix[rng.permutation(prefix.shape[0])]
            if suffix.shape[0] > 1:
                suffix = suffix[rng.permutation(suffix.shape[0])]
        else:
            raise ValueError(
                f"Unknown seq_order={seq_order!r}. "
                f"Expected one of {sorted(VALID_SEQ_ORDERS)}."
            )

        transformed[i, :k] = prefix
        transformed[i, k:] = suffix

    return transformed.reshape(transformed.shape[0], -1)


def build_solution_board(triples: np.ndarray) -> np.ndarray:
    rows = triples[:, :, 0]
    cols = triples[:, :, 1]
    vals = triples[:, :, 2]

    if rows.min() < 0 or rows.max() > 8:
        raise ValueError("Sudoku rows must be in [0, 8].")
    if cols.min() < 0 or cols.max() > 8:
        raise ValueError("Sudoku cols must be in [0, 8].")
    if vals.min() < 1 or vals.max() > 9:
        raise ValueError("Sudoku values must be in [1, 9].")

    cell_id = rows * 9 + cols
    puzzle = np.zeros((triples.shape[0], 81), dtype=np.int32)
    puzzle[np.arange(triples.shape[0])[:, None], cell_id] = vals
    return puzzle
