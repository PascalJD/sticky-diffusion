# src/sticky/tasks/cifar10_md4.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.data.cifar10_discrete import make_cifar10_iterator
from sticky.tasks.base import Batch, Metrics, Task, TaskSpec


@dataclass
class CIFAR10MD4Task(Task):
    data_dir: Optional[str]
    batch_size: int
    eval_batch_size: int
    vocab_size: int = 256
    num_classes: int = -1  # -1 => unconditional
    drop_remainder: bool = True
    augment_enabled: bool = True
    augment_prob: float = 0.15
    augment_rotate: bool = True
    augment_hflip: bool = True
    augment_eval: bool = False

    def __post_init__(self):
        self.spec = TaskSpec(
            name="md4_cifar10",
            task_type="image",
            data_shape=(32, 32, 3),
            vocab_size=self.vocab_size,
            num_classes=self.num_classes,
        )

    def make_dataloaders(
        self, *, seed: int
    ) -> Tuple[Iterable[Batch], Optional[Iterable[Batch]]]:
        train_iter = make_cifar10_iterator(
            split="train",
            batch_size=self.batch_size,
            seed=seed,
            data_dir=self.data_dir,
            shuffle=True,
            repeat=True,
            drop_remainder=self.drop_remainder,
            augment=self.augment_enabled,
            augment_prob=self.augment_prob,
            augment_rotate=self.augment_rotate,
            augment_hflip=self.augment_hflip,
        )
        eval_iter = make_cifar10_iterator(
            split="test",
            batch_size=self.eval_batch_size,
            seed=seed + 1,
            data_dir=self.data_dir,
            shuffle=False,
            repeat=False,
            drop_remainder=False,
            augment=self.augment_eval,
            augment_prob=self.augment_prob,
            augment_rotate=self.augment_rotate,
            augment_hflip=self.augment_hflip,
        )
        return train_iter, eval_iter

    def loss_fn(
        self,
        *,
        rng: jax.random.PRNGKey,
        model,
        params,
        batch: Batch,
        train: bool,
    ) -> Tuple[jnp.ndarray, Metrics]:
        key_sample, key_dropout = jax.random.split(rng)
        x = batch["image"].astype(jnp.int32)  # [B,32,32,3] in [0..255]
        if self.num_classes > 0:
            cond = batch["label"].astype(jnp.int32)
        else:
            cond = None

        rngs = {"sample": key_sample}
        if train:
            rngs["dropout"] = key_dropout

        stats = model.apply(
            {"params": params},
            x,
            cond=cond,
            train=train,
            rngs=rngs,
        )
        return stats["loss"], stats

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is int32 tokens in [0..255]. Convert to uint8 for saving/visualization.
        return jnp.clip(x, 0, self.vocab_size - 1).astype(jnp.uint8)
