from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.data.sudoku import get_sudoku_num_examples, make_sudoku_iterator
from sticky.rng import PRNGKey
from sticky.tasks.base import Batch, Metrics, Task, TaskSpec


@dataclass
class SudokuMDLMTask(Task):
    """Conditional Sudoku sequence task for vanilla MDLM.

    The first `3 * start_index` tokens are fixed clue-prefix conditioning.
    The remaining suffix tokens are the unknown solution tokens and are the
    only positions that contribute to the loss.
    """

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

    def __post_init__(self):
        self.spec = TaskSpec(
            name="mdlm_sudoku",
            task_type="text",
            data_shape=tuple(self.data_shape),
            vocab_size=int(self.vocab_size),
            num_classes=int(self.num_classes),
        )

    def make_dataloaders(
        self, *, seed: int
    ) -> Tuple[Iterable[Batch], Optional[Iterable[Batch]]]:
        train_it = make_sudoku_iterator(
            split="train",
            batch_size=int(self.batch_size),
            seed=int(seed),
            data_dir=self.data_dir,
            train_file=self.train_file,
            test_file=self.test_file,
            shuffle=bool(self.shuffle),
            repeat=True,
            drop_remainder=bool(self.drop_remainder),
            seq_order=str(self.seq_order),
            mmap=bool(self.mmap),
            max_examples=int(self.max_train_examples),
            auto_download=bool(self.auto_download),
            download_timeout_sec=int(self.download_timeout_sec),
            download_retries=int(self.download_retries),
        )
        eval_it = make_sudoku_iterator(
            split="test",
            batch_size=int(self.eval_batch_size),
            seed=int(seed) + 1,
            data_dir=self.data_dir,
            train_file=self.train_file,
            test_file=self.test_file,
            shuffle=False,
            repeat=False,
            drop_remainder=False,
            seq_order=str(self.seq_order),
            mmap=bool(self.mmap),
            max_examples=int(self.max_test_examples),
            auto_download=bool(self.auto_download),
            download_timeout_sec=int(self.download_timeout_sec),
            download_retries=int(self.download_retries),
        )
        return train_it, eval_it

    def loss_fn(
        self,
        *,
        rng: PRNGKey,
        model,
        params,
        batch: Batch,
        train: bool,
    ) -> tuple[jnp.ndarray, Metrics]:
        key_sample, key_dropout = jax.random.split(rng)
        x = batch["image"].astype(jnp.int32)
        start_index = batch["start_index"].astype(jnp.int32)

        token_pos = jnp.arange(x.shape[-1], dtype=jnp.int32)[None, :]
        # Original Sudoku task semantics: the clue prefix is fixed conditioning
        # and only the suffix contributes to the MDLM objective.
        known_token_mask = token_pos < (3 * start_index)
        loss_mask = ~known_token_mask

        rngs = {"sample": key_sample}
        if train:
            rngs["dropout"] = key_dropout

        stats = model.apply(
            {"params": params},
            x,
            train=train,
            known_token_mask=known_token_mask,
            loss_mask=loss_mask,
            rngs=rngs,
        )
        return stats["loss"], stats

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(x, dtype=jnp.int32)

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
