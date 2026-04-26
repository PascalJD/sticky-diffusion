from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.rng import PRNGKey
from sticky.tasks.base import Batch, Metrics
from sticky.tasks.tfds_discrete_image import TFDSDiscreteImageTaskBase


@dataclass
class CIFAR10DiscreteTask(TFDSDiscreteImageTaskBase):
    """Shared CIFAR-10 discrete-token image task for MD4/CADD-like models.

    Assumes model API:
      model.apply({"params": params}, x, cond=cond, train=train, rngs=rngs)
    returning a dict containing at least {"loss": ...}.
    """

    task_name: str
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

    dataset_name: str = "cifar10"
    train_split: str = "train"
    eval_split: str = "test"
    include_label: bool | str = "auto"
    dummy_label_value: int = -1
    data_shape: Tuple[int, ...] = (32, 32, 3)
    shuffle_buffer_size: Optional[int] = None

    def __post_init__(self):
        self.spec = self._build_image_task_spec(name=str(self.task_name))

    def loss_fn(
        self,
        *,
        rng: PRNGKey,
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
