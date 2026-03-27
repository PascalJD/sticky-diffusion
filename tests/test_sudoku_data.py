from __future__ import annotations

import numpy as np

from sticky.data.sudoku import ensure_sudoku_data_available, get_sudoku_num_examples


def _synthetic_sudoku_table(num_examples: int) -> np.ndarray:
    table = np.zeros((num_examples, 325), dtype=np.int32)
    for i in range(num_examples):
        table[i, 0] = i % 81
    return table


def test_ensure_sudoku_data_available_returns_existing_files_without_download(monkeypatch, tmp_path):
    train_path = tmp_path / "Sudoku-train-data.npy"
    test_path = tmp_path / "Sudoku-test-data.npy"
    np.save(train_path, _synthetic_sudoku_table(2))
    np.save(test_path, _synthetic_sudoku_table(3))

    def _unexpected_download(**kwargs):
        raise AssertionError("download should not run when Sudoku files already exist")

    monkeypatch.setattr(
        "sticky.data.sudoku_download._download_google_drive_file",
        _unexpected_download,
    )

    resolved_train, resolved_test = ensure_sudoku_data_available(data_dir=str(tmp_path))

    assert resolved_train == train_path
    assert resolved_test == test_path


def test_get_sudoku_num_examples_respects_max_examples(tmp_path):
    train_path = tmp_path / "Sudoku-train-data.npy"
    np.save(train_path, _synthetic_sudoku_table(5))

    assert (
        get_sudoku_num_examples(
            split="train",
            data_dir=str(tmp_path),
            mmap=False,
            max_examples=-1,
            auto_download=False,
        )
        == 5
    )
    assert (
        get_sudoku_num_examples(
            split="train",
            data_dir=str(tmp_path),
            mmap=False,
            max_examples=3,
            auto_download=False,
        )
        == 3
    )
