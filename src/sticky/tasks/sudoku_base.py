from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import jax.numpy as jnp

from sticky.data.sudoku import get_sudoku_num_examples, make_sudoku_iterator
from sticky.tasks.base import Batch, Task


@dataclass
class SudokuTaskBase(Task):
    data_dir: Optional[str]
    train_file: str
    test_file: str
    batch_size: int
    eval_batch_size: int
    data_shape: Tuple[int, ...]
    vocab_size: int
    num_classes: int
    drop_remainder: bool = True
    shuffle: bool = True
    seq_order: str = "dataset"
    mmap: bool = True
    max_train_examples: int = -1
    max_test_examples: int = -1
    auto_download: bool = True
    download_timeout_sec: int = 120
    download_retries: int = 8

    def _make_sudoku_iterator(
        self,
        *,
        split: str,
        batch_size: int,
        seed: int,
        shuffle: bool,
        repeat: bool,
        drop_remainder: bool,
        max_examples: int,
    ):
        return make_sudoku_iterator(
            split=str(split),
            batch_size=int(batch_size),
            seed=int(seed),
            data_dir=self.data_dir,
            train_file=self.train_file,
            test_file=self.test_file,
            shuffle=bool(shuffle),
            repeat=bool(repeat),
            drop_remainder=bool(drop_remainder),
            seq_order=str(self.seq_order),
            mmap=bool(self.mmap),
            max_examples=int(max_examples),
            auto_download=bool(self.auto_download),
            download_timeout_sec=int(self.download_timeout_sec),
            download_retries=int(self.download_retries),
        )

    def make_dataloaders(
        self, *, seed: int
    ) -> Tuple[Iterable[Batch], Optional[Iterable[Batch]]]:
        train_it = self._make_sudoku_iterator(
            split="train",
            batch_size=int(self.batch_size),
            seed=int(seed),
            shuffle=bool(self.shuffle),
            repeat=True,
            drop_remainder=bool(self.drop_remainder),
            max_examples=int(self.max_train_examples),
        )
        eval_it = self._make_sudoku_iterator(
            split="test",
            batch_size=int(self.eval_batch_size),
            seed=int(seed) + 1,
            shuffle=False,
            repeat=False,
            drop_remainder=False,
            max_examples=int(self.max_test_examples),
        )
        return train_it, eval_it

    def train_num_examples(self) -> int | None:
        return get_sudoku_num_examples(
            split="train",
            data_dir=self.data_dir,
            train_file=self.train_file,
            test_file=self.test_file,
            mmap=bool(self.mmap),
            max_examples=int(self.max_train_examples),
            auto_download=bool(self.auto_download),
            download_timeout_sec=int(self.download_timeout_sec),
            download_retries=int(self.download_retries),
        )

    def _known_prefix_mask(self, *, start_index: jnp.ndarray, seq_len: int) -> jnp.ndarray:
        token_pos = jnp.arange(int(seq_len), dtype=jnp.int32)[None, :]
        return token_pos < (3 * start_index)
