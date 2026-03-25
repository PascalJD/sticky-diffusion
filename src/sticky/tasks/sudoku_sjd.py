from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.data.sudoku import get_sudoku_num_examples, make_sudoku_iterator
from sticky.models.sjd.losses import ce_allocation_loss
from sticky.tasks.base import Task, TaskSpec

Array = jnp.ndarray


@dataclass
class SudokuSJDTask(Task):
    """Sudoku sequence task for SJD.

    Each example is a sequence of length 243 where tokens are grouped as
    `(row, col, value)` triplets for the 81 cells. The model sees tokens in
    `[0, 9]` with:
      - row/col in `[0, 8]`
      - value in `[1, 9]`

    The original Google Sudoku task is conditional sequence modeling:
      - the first `3 * start_index` tokens are the given clue prefix
      - the remaining tokens are the solution suffix
      - training loss is applied only on the suffix
    """

    # dataset
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
    seq_order: str = "dataset"  # dataset | fixed | random
    mmap: bool = True
    max_train_examples: int = -1
    max_test_examples: int = -1
    auto_download: bool = True
    download_timeout_sec: int = 120
    download_retries: int = 8

    # VP forward schedule
    beta: Callable[[Array], Array] = None
    hazard: Optional[Any] = None
    jump: Optional[Any] = None
    T: float = 1.0
    log_state_dependency: bool = True
    state_dep_log_ratio_clip: float = 10.0

    def __post_init__(self):
        if self.beta is None:
            raise ValueError("SudokuSJDTask requires a beta schedule.")
        self._spec = TaskSpec(
            name="sjd_sudoku",
            task_type="text",
            data_shape=tuple(self.data_shape),
            vocab_size=int(self.vocab_size),
            num_classes=int(self.num_classes),
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def make_dataloaders(self, seed: int):
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
        rng: Array,
        model,
        params: Any,
        batch: Dict[str, Array],
        train: bool,
    ):
        key_loss, key_dropout = jax.random.split(rng)

        # Discrete sequence tokens [B, 243].
        x0_idx = batch["image"].astype(jnp.int32)
        start_index = batch["start_index"].astype(jnp.int32)

        # Match the original Sudoku task definition: the clue prefix is always
        # conditioned on and never contributes to the loss.
        token_pos = jnp.arange(x0_idx.shape[-1], dtype=jnp.int32)[None, :]
        given_mask = token_pos < (3 * start_index)

        # Anchor vectors [B, 243, d_anchor].
        x0_anchor = model.apply({"params": params}, x0_idx, method=model.embed)
        anchor_table = None
        if self.log_state_dependency:
            try:
                anchor_table = params["anchors"]["table"]
            except Exception:
                anchor_table = model.apply({"params": params}, method=model.anchor_table)

        def apply_fn(p, xt, t_img):
            if train:
                return model.apply(
                    {"params": p},
                    xt,
                    t_img,
                    train=True,
                    rngs={"dropout": key_dropout},
                )
            return model.apply({"params": p}, xt, t_img, train=False)

        loss, metrics = ce_allocation_loss(
            key=key_loss,
            params=params,
            apply_fn=apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=self.beta,
            hazard=self.hazard,
            jump=self.jump if self.log_state_dependency else None,
            anchor_table=anchor_table,
            state_dep_log_ratio_clip=float(self.state_dep_log_ratio_clip),
            T=float(self.T),
            given_mask=given_mask,
        )

        return loss, metrics

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
