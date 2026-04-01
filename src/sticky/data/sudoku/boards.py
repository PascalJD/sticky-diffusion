from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping, Optional

import numpy as np

from .download import ensure_sudoku_data_available
from .paths import SUDOKU_DATA_URL, resolve_data_file


SUDOKU_NUM_CELLS = 81
SUDOKU_TABLE_WIDTH = 325


def _load_sudoku_table(file_path: Path, *, mmap: bool) -> np.ndarray:
    mmap_mode = "r" if bool(mmap) else None
    table = np.load(str(file_path), mmap_mode=mmap_mode)
    if table.ndim != 2 or table.shape[1] != SUDOKU_TABLE_WIDTH:
        raise ValueError(
            "Expected Sudoku array with shape (N, 325): "
            f"got {table.shape} from {file_path}."
        )
    return table


def _normalize_shah_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int32)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != SUDOKU_TABLE_WIDTH:
        raise ValueError(
            "Expected Shah Sudoku rows with shape (batch, 325) or (325,), "
            f"got {rows.shape}."
        )
    return rows


def _normalize_triples(triples: np.ndarray) -> np.ndarray:
    triples = np.asarray(triples, dtype=np.int32)
    if triples.ndim == 2:
        triples = triples[None, :, :]
    if triples.ndim != 3 or triples.shape[1:] != (SUDOKU_NUM_CELLS, 3):
        raise ValueError(
            "Expected Sudoku triples with shape (batch, 81, 3) or (81, 3), "
            f"got {triples.shape}."
        )
    rows = triples[:, :, 0]
    cols = triples[:, :, 1]
    vals = triples[:, :, 2]
    if rows.min() < 0 or rows.max() > 8:
        raise ValueError("Sudoku rows must be in [0, 8].")
    if cols.min() < 0 or cols.max() > 8:
        raise ValueError("Sudoku cols must be in [0, 8].")
    if vals.min() < 1 or vals.max() > 9:
        raise ValueError("Sudoku values must be in [1, 9].")
    return triples


def _normalize_start_index(start_index: np.ndarray, *, batch_size: int) -> np.ndarray:
    start = np.asarray(start_index, dtype=np.int32).reshape(-1)
    if start.shape[0] != int(batch_size):
        raise ValueError(
            "start_index must have one entry per Sudoku example: "
            f"got {start.shape[0]} for batch size {batch_size}."
        )
    if np.any(start < 0) or np.any(start > SUDOKU_NUM_CELLS):
        raise ValueError(
            f"Invalid Sudoku start_index; expected values in 0..{SUDOKU_NUM_CELLS}."
        )
    return start


def _cell_ids_from_triples(triples: np.ndarray) -> np.ndarray:
    triples = _normalize_triples(triples)
    return (triples[:, :, 0] * 9) + triples[:, :, 1]


def extract_shah_triples(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = _normalize_shah_rows(rows)
    start_index = _normalize_start_index(rows[:, 0], batch_size=rows.shape[0])
    triples = rows[:, 1:].reshape(-1, SUDOKU_NUM_CELLS, 4)[:, :, :3]
    return triples.astype(np.int32), start_index


def build_solution_board(triples: np.ndarray) -> np.ndarray:
    triples = _normalize_triples(triples)
    batch_size = int(triples.shape[0])
    cell_ids = _cell_ids_from_triples(triples)
    board = np.zeros((batch_size, SUDOKU_NUM_CELLS), dtype=np.int32)
    board[np.arange(batch_size)[:, None], cell_ids] = triples[:, :, 2]

    if np.any(board == 0):
        raise ValueError(
            "Sudoku solution board reconstruction left blank cells. "
            "Each Shah example should specify all 81 cells exactly once."
        )
    return board


def build_clue_mask(
    triples: np.ndarray,
    *,
    start_index: np.ndarray,
) -> np.ndarray:
    triples = _normalize_triples(triples)
    start = _normalize_start_index(start_index, batch_size=triples.shape[0])
    batch_size = int(triples.shape[0])
    clue_mask = np.zeros((batch_size, SUDOKU_NUM_CELLS), dtype=np.bool_)
    cell_ids = _cell_ids_from_triples(triples)
    for i in range(batch_size):
        clue_mask[i, cell_ids[i, : int(start[i])]] = True
    return clue_mask


def build_clue_board(
    triples: np.ndarray,
    *,
    start_index: np.ndarray,
) -> np.ndarray:
    triples = _normalize_triples(triples)
    start = _normalize_start_index(start_index, batch_size=triples.shape[0])
    batch_size = int(triples.shape[0])
    clue_board = np.zeros((batch_size, SUDOKU_NUM_CELLS), dtype=np.int32)
    cell_ids = _cell_ids_from_triples(triples)
    for i in range(batch_size):
        clue_count = int(start[i])
        clue_board[i, cell_ids[i, :clue_count]] = triples[i, :clue_count, 2]
    return clue_board


def build_board_batch(rows: np.ndarray) -> Mapping[str, np.ndarray]:
    triples, start_index = extract_shah_triples(rows)
    solution_board = build_solution_board(triples)
    clue_board = build_clue_board(triples, start_index=start_index)
    clue_mask = build_clue_mask(triples, start_index=start_index)
    return {
        "solution_board": solution_board.astype(np.int32),
        "clue_board": clue_board.astype(np.int32),
        "clue_mask": clue_mask.astype(np.bool_),
        "start_index": start_index.reshape(-1, 1).astype(np.int32),
    }


class SudokuBoardBatchIterator:
    """Board-level Shah Sudoku iterator for the 81-cell inpainting benchmark.

    This deliberately reconstructs row-major clue/solution boards from the Shah
    triples and ignores the stored suffix ordering after board construction.
    """

    def __init__(
        self,
        *,
        file_path: Path,
        batch_size: int,
        seed: int,
        shuffle: bool,
        repeat: bool,
        drop_remainder: bool,
        mmap: bool,
        max_examples: int,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if not file_path.exists():
            raise FileNotFoundError(
                f"Sudoku file not found: {file_path}. "
                "Download `Sudoku-train-data.npy` and `Sudoku-test-data.npy` from "
                f"{SUDOKU_DATA_URL}."
            )

        table = _load_sudoku_table(file_path, mmap=bool(mmap))

        n_total = int(table.shape[0])
        if max_examples > 0:
            n_total = min(n_total, int(max_examples))
        if n_total <= 0:
            raise ValueError("Sudoku dataset is empty after applying max_examples.")
        if bool(repeat) and bool(drop_remainder) and (n_total < int(batch_size)):
            raise ValueError(
                "No full batches available: repeat=true and drop_remainder=true "
                f"requires dataset size >= batch_size, got {n_total} < {batch_size}."
            )

        self._table = table
        self._n = n_total
        self._batch_size = int(batch_size)
        self._repeat = bool(repeat)
        self._drop_remainder = bool(drop_remainder)
        self._shuffle = bool(shuffle)
        self._rng = np.random.default_rng(int(seed))
        self._idx = np.arange(self._n, dtype=np.int64)
        self._cursor = 0
        if self._shuffle:
            self._rng.shuffle(self._idx)

    def __iter__(self):
        return self

    def _next_indices(self) -> np.ndarray:
        while True:
            if self._cursor >= self._n:
                if not self._repeat:
                    raise StopIteration
                self._cursor = 0
                if self._shuffle:
                    self._rng.shuffle(self._idx)

            remaining = self._n - self._cursor
            if remaining < self._batch_size and self._drop_remainder:
                if not self._repeat:
                    raise StopIteration
                self._cursor = self._n
                continue

            take = self._batch_size if remaining >= self._batch_size else remaining
            out = self._idx[self._cursor : self._cursor + take]
            self._cursor += take
            return out

    def __next__(self) -> Mapping[str, np.ndarray]:
        batch_idx = self._next_indices()
        rows = np.asarray(self._table[batch_idx], dtype=np.int32)
        board_batch = build_board_batch(rows)
        solution_board = np.asarray(board_batch["solution_board"], dtype=np.int32)
        return {
            **dict(board_batch),
            "image": solution_board,
        }


def make_sudoku_board_iterator(
    *,
    split: str,
    batch_size: int,
    seed: int = 0,
    data_dir: Optional[str] = None,
    train_file: str = "Sudoku-train-data.npy",
    test_file: str = "Sudoku-test-data.npy",
    shuffle: bool = True,
    repeat: bool = False,
    drop_remainder: bool = True,
    mmap: bool = True,
    max_examples: int = -1,
    auto_download: bool = True,
    download_timeout_sec: int = 120,
    download_retries: int = 8,
) -> Iterator[Mapping[str, np.ndarray]]:
    split_key = str(split).lower()
    if split_key not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")

    if bool(auto_download):
        ensure_sudoku_data_available(
            data_dir=data_dir,
            train_file=train_file,
            test_file=test_file,
            timeout_sec=int(download_timeout_sec),
            retries=int(download_retries),
        )

    filename = train_file if split_key == "train" else test_file
    file_path = resolve_data_file(data_dir=data_dir, filename=filename)

    return iter(
        SudokuBoardBatchIterator(
            file_path=file_path,
            batch_size=int(batch_size),
            seed=int(seed),
            shuffle=bool(shuffle),
            repeat=bool(repeat),
            drop_remainder=bool(drop_remainder),
            mmap=bool(mmap),
            max_examples=int(max_examples),
        )
    )


def get_sudoku_board_num_examples(
    *,
    split: str,
    data_dir: Optional[str] = None,
    train_file: str = "Sudoku-train-data.npy",
    test_file: str = "Sudoku-test-data.npy",
    mmap: bool = True,
    max_examples: int = -1,
    auto_download: bool = True,
    download_timeout_sec: int = 120,
    download_retries: int = 8,
) -> int:
    split_key = str(split).lower()
    if split_key not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")

    if bool(auto_download):
        ensure_sudoku_data_available(
            data_dir=data_dir,
            train_file=train_file,
            test_file=test_file,
            timeout_sec=int(download_timeout_sec),
            retries=int(download_retries),
        )

    filename = train_file if split_key == "train" else test_file
    file_path = resolve_data_file(data_dir=data_dir, filename=filename)
    table = _load_sudoku_table(file_path, mmap=bool(mmap))

    n_total = int(table.shape[0])
    if int(max_examples) > 0:
        n_total = min(n_total, int(max_examples))
    if n_total <= 0:
        raise ValueError("Sudoku dataset is empty after applying max_examples.")
    return n_total
