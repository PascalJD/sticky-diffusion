from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Tuple

import jax.numpy as jnp

from sticky.tasks.base import TaskSpec
from sticky.tasks.sjd_base import SJDTaskBase
from sticky.tasks.tfds_discrete_image import TFDSDiscreteImageTaskBase

Array = jnp.ndarray


@dataclass
class CIFAR10SJDTask(SJDTaskBase, TFDSDiscreteImageTaskBase):
    """CIFAR-10 task for training the SJD ContinuousClassifier.

    The dataset provides discrete tokens in `[0, vocab_size)` with shape
    `(B, H, W, C)`.

    The model provides:
      - `embed(token_ids) -> anchor vectors (B, H, W, C, d)`
      - `__call__(y_t, t_img) -> logits (B, H, W, C, vocab_size)`

    We train with `ce_allocation_loss` in continuous VP space.
    """

    # CIFAR-10 passes the time argument positionally to model.apply.
    _t_passes_positionally: ClassVar[bool] = True

    # dataset
    data_dir: str
    batch_size: int
    eval_batch_size: int
    data_shape: Tuple[int, ...]
    vocab_size: int
    num_classes: int

    task_name: str = "sjd_cifar10"
    dataset_name: str = "cifar10"
    train_split: str = "train"
    eval_split: str = "test"
    include_label: bool | str = "auto"
    dummy_label_value: int = -1

    # data augmentation
    augment_enabled: bool = True
    augment_prob: float = 0.15
    augment_rotate: bool = True
    augment_hflip: bool = True
    augment_eval: bool = False

    def __post_init__(self):
        super().__post_init__()
        self._spec = self._build_image_task_spec(name=str(self.task_name))

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def make_dataloaders(self, seed: int):
        return super().make_dataloaders(seed=seed)

    def _extract_x0_idx(self, batch) -> Array:
        return batch["image"].astype(jnp.int32)
