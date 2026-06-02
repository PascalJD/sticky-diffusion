from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sticky.data.text8 import decode_ids, make_text8_iterator
from sticky.tasks.base import Batch, TaskSpec
from sticky.tasks.sjd_base import SJDTaskBase


Array = jnp.ndarray


@dataclass
class Text8SJDTask(SJDTaskBase):
    """Text8 SJD task wiring char-level token IDs through the SJD anchor path."""

    task_name: str
    train_tokens_path: str
    eval_tokens_path: Optional[str]
    batch_size: int
    eval_batch_size: int
    seq_len: int
    vocab_size: int
    num_classes: int = -1
    drop_remainder: bool = True
    shuffle: bool = True
    mmap: bool = True
    max_train_examples: int = -1
    max_eval_examples: int = -1

    def __post_init__(self):
        super().__post_init__()
        self.spec = TaskSpec(
            name=str(self.task_name),
            task_type="text",
            data_shape=(int(self.seq_len),),
            vocab_size=int(self.vocab_size),
            num_classes=int(self.num_classes),
        )

    def make_dataloaders(
        self, *, seed: int
    ) -> Tuple[Iterable[Batch], Optional[Iterable[Batch]]]:
        train_iter = make_text8_iterator(
            split="train",
            batch_size=int(self.batch_size),
            seq_len=int(self.seq_len),
            train_tokens_path=str(self.train_tokens_path),
            eval_tokens_path=self.eval_tokens_path,
            seed=int(seed),
            shuffle=bool(self.shuffle),
            repeat=True,
            drop_remainder=bool(self.drop_remainder),
            mmap=bool(self.mmap),
            max_examples=int(self.max_train_examples),
        )
        eval_iter = make_text8_iterator(
            split="eval",
            batch_size=int(self.eval_batch_size),
            seq_len=int(self.seq_len),
            train_tokens_path=str(self.train_tokens_path),
            eval_tokens_path=self.eval_tokens_path,
            seed=int(seed) + 1,
            shuffle=False,
            repeat=False,
            drop_remainder=False,
            mmap=bool(self.mmap),
            max_examples=int(self.max_eval_examples),
        )
        return train_iter, eval_iter

    def _extract_x0_idx(self, batch: Batch) -> Array:
        return jnp.asarray(batch["image"], dtype=jnp.int32)

    def _extra_metrics(
        self, metrics: dict, batch: Batch, x0_idx: Array
    ) -> dict:
        metrics["clean_index_min"] = jnp.min(x0_idx).astype(jnp.float32)
        metrics["clean_index_max"] = jnp.max(x0_idx).astype(jnp.float32)
        return metrics

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(x, dtype=jnp.int32)

    def format_samples_for_logging(self, x: jnp.ndarray) -> list[str] | None:
        samples = np.asarray(jax.device_get(x))
        if samples.ndim == 1:
            samples = samples[None, :]
        return [decode_ids(row) for row in samples]
