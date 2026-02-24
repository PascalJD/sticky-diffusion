from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.data.cifar10_discrete import make_cifar10_iterator
from sticky.tasks.base import Batch, Metrics, Task, TaskSpec


@dataclass
class CIFAR10CADDTask(Task):
    """CIFAR-10 task for CADD.

    The model API matches the MD4 task (the model returns a dict with a "loss" key),
    but we register a distinct task/spec name for experiment bookkeeping.
    """

    data_dir: Optional[str]
    batch_size: int
    eval_batch_size: int
    vocab_size: int = 256
    num_classes: int = -1  # -1 => unconditional
    drop_remainder: bool = True

    # Pixel-preserving augmentation (rotation + hflip).
    augment_enabled: bool = True
    augment_prob: float = 0.15
    augment_rotate: bool = True
    augment_hflip: bool = True
    augment_eval: bool = False

    def __post_init__(self):
        self.spec = TaskSpec(
            name="cadd_cifar10",
            task_type="image",
            data_shape=(32, 32, 3),
            vocab_size=int(self.vocab_size),
            num_classes=int(self.num_classes),
        )

    def make_dataloaders(
        self, *, seed: int
    ) -> Tuple[Iterable[Batch], Optional[Iterable[Batch]]]:
        train_iter = make_cifar10_iterator(
            split="train",
            batch_size=int(self.batch_size),
            seed=int(seed),
            data_dir=self.data_dir,
            shuffle=True,
            repeat=True,
            drop_remainder=bool(self.drop_remainder),
            augment=bool(self.augment_enabled),
            augment_prob=float(self.augment_prob),
            augment_rotate=bool(self.augment_rotate),
            augment_hflip=bool(self.augment_hflip),
        )
        eval_iter = make_cifar10_iterator(
            split="test",
            batch_size=int(self.eval_batch_size),
            seed=int(seed) + 1,
            data_dir=self.data_dir,
            shuffle=False,
            repeat=False,
            drop_remainder=False,
            augment=bool(self.augment_eval),
            augment_prob=float(self.augment_prob),
            augment_rotate=bool(self.augment_rotate),
            augment_hflip=bool(self.augment_hflip),
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
        return jnp.clip(x, 0, self.vocab_size - 1).astype(jnp.uint8)
