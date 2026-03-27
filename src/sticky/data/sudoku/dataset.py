from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping, Optional

import numpy as np

from .download import ensure_sudoku_data_available
from .paths import SUDOKU_DATA_URL, resolve_data_file
from .transforms import VALID_SEQ_ORDERS, build_solution_board, to_inputs


def _load_sudoku_table(file_path: Path, *, mmap: bool) -> np.ndarray:
    mmap_mode = "r" if bool(mmap) else None
    table = np.load(str(file_path), mmap_mode=mmap_mode)
    if table.ndim != 2 or table.shape[1] != 325:
        raise ValueError(
            "Expected Sudoku array with shape (N, 325): "
            f"got {table.shape} from {file_path}."
        )
    return table


class SudokuBatchIterator:
    def __init__(
        self,
        *,
        file_path: Path,
        batch_size: int,
        seed: int,
        shuffle: bool,
        repeat: bool,
        drop_remainder: bool,
        seq_order: str,
        mmap: bool,
        max_examples: int,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if seq_order not in VALID_SEQ_ORDERS:
            raise ValueError(
                f"Unknown seq_order={seq_order!r}. "
                f"Expected one of {sorted(VALID_SEQ_ORDERS)}."
            )
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
        self._seq_order = str(seq_order)
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

        start_index = rows[:, 0]
        triples = rows[:, 1:].reshape(-1, 81, 4)[:, :, :3]

        inputs = to_inputs(
            triples=triples,
            start_index=start_index,
            seq_order=self._seq_order,
            rng=self._rng,
        ).astype(np.int32)
        puzzle = build_solution_board(triples).astype(np.int32)

        return {
            "image": inputs,
            "puzzle": puzzle,
            "start_index": start_index.reshape(-1, 1).astype(np.int32),
        }


def make_sudoku_iterator(
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
    seq_order: str = "dataset",
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
        SudokuBatchIterator(
            file_path=file_path,
            batch_size=int(batch_size),
            seed=int(seed),
            shuffle=bool(shuffle),
            repeat=bool(repeat),
            drop_remainder=bool(drop_remainder),
            seq_order=str(seq_order),
            mmap=bool(mmap),
            max_examples=int(max_examples),
        )
    )


def get_sudoku_num_examples(
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
