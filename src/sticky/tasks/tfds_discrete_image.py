from __future__ import annotations

from typing import Iterable, Optional, Tuple

import jax.numpy as jnp

from sticky.data.tfds_discrete_image import make_tfds_discrete_image_iterator
from sticky.tasks.base import Batch, Task, TaskSpec


class TFDSDiscreteImageTaskBase(Task):
    """Shared TFDS-backed image task plumbing for discrete uint8 image models."""

    dataset_name: str
    train_split: str = "train"
    eval_split: str = "test"
    data_dir: Optional[str]
    batch_size: int
    eval_batch_size: int
    data_shape: Tuple[int, ...]
    vocab_size: int
    num_classes: int
    drop_remainder: bool = True
    augment_enabled: bool = True
    augment_prob: float = 0.15
    augment_rotate: bool = True
    augment_hflip: bool = True
    augment_eval: bool = False
    include_label: bool | str = "auto"
    dummy_label_value: int = -1
    shuffle_buffer_size: Optional[int] = None

    def _build_image_task_spec(self, *, name: str) -> TaskSpec:
        return TaskSpec(
            name=str(name),
            task_type="image",
            data_shape=tuple(self.data_shape),
            vocab_size=int(self.vocab_size),
            num_classes=int(self.num_classes),
        )

    def _make_tfds_iterator(
        self,
        *,
        split: str,
        batch_size: int,
        seed: int,
        shuffle: bool,
        repeat: bool,
        drop_remainder: bool,
        augment: bool,
    ):
        return make_tfds_discrete_image_iterator(
            dataset_name=str(self.dataset_name),
            split=str(split),
            batch_size=int(batch_size),
            seed=int(seed),
            data_dir=self.data_dir,
            shuffle=bool(shuffle),
            repeat=bool(repeat),
            drop_remainder=bool(drop_remainder),
            augment=bool(augment),
            augment_prob=float(self.augment_prob),
            augment_rotate=bool(self.augment_rotate),
            augment_hflip=bool(self.augment_hflip),
            include_label=self.include_label,
            dummy_label_value=int(self.dummy_label_value),
            shuffle_buffer_size=self.shuffle_buffer_size,
        )

    def make_dataloaders(
        self, *, seed: int
    ) -> Tuple[Iterable[Batch], Optional[Iterable[Batch]]]:
        train_it = self._make_tfds_iterator(
            split=str(self.train_split),
            batch_size=int(self.batch_size),
            seed=int(seed),
            shuffle=True,
            repeat=True,
            drop_remainder=bool(self.drop_remainder),
            augment=bool(self.augment_enabled),
        )
        eval_it = self._make_tfds_iterator(
            split=str(self.eval_split),
            batch_size=int(self.eval_batch_size),
            seed=int(seed) + 1,
            shuffle=False,
            repeat=False,
            drop_remainder=False,
            augment=bool(self.augment_eval),
        )
        return train_it, eval_it

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(x, 0, self.vocab_size - 1).astype(jnp.uint8)
