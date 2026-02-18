from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.data.cifar10_discrete import make_cifar10_iterator
from sticky.models.sjd.losses import ce_allocation_loss
from sticky.tasks.base import Task, TaskSpec

Array = jnp.ndarray


@dataclass
class CIFAR10SJDTask(Task):
    """CIFAR-10 task for training the SJD ContinuousClassifier.

    The dataset provides discrete tokens in `[0, vocab_size)` with shape
    `(B, H, W, C)`.

    The model provides:
      - `embed(token_ids) -> anchor vectors (B, H, W, C, d)`
      - `__call__(y_t, t_img) -> logits (B, H, W, C, vocab_size)`

    We train with `ce_allocation_loss` in continuous VP space.
    """

    # dataset
    data_dir: str
    batch_size: int
    eval_batch_size: int
    data_shape: Tuple[int, ...]
    vocab_size: int
    num_classes: int

    # VP forward schedule
    beta: Callable[[Array], Array]
    hazard: Optional[Any] = None
    jump: Optional[Any] = None
    T: float = 1.0
    log_state_dependency: bool = True
    state_dep_log_ratio_clip: float = 10.0

    # data augmentation
    augment_enabled: bool = True
    augment_prob: float = 0.15
    augment_rotate: bool = True
    augment_hflip: bool = True
    augment_eval: bool = False

    def __post_init__(self):
        self._spec = TaskSpec(
            name="sjd_cifar10",
            task_type="image",
            data_shape=tuple(self.data_shape),
            vocab_size=int(self.vocab_size),
            num_classes=int(self.num_classes),
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def make_dataloaders(self, seed: int):
        train_it = make_cifar10_iterator(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            split="train",
            shuffle=True,
            seed=seed,
            repeat=True,
            augment=self.augment_enabled,
            augment_prob=self.augment_prob,
            augment_rotate=self.augment_rotate,
            augment_hflip=self.augment_hflip,
        )
        eval_it = make_cifar10_iterator(
            data_dir=self.data_dir,
            batch_size=self.eval_batch_size,
            split="test",
            shuffle=False,
            seed=seed,
            repeat=False,
            augment=self.augment_eval,
            augment_prob=self.augment_prob,
            augment_rotate=self.augment_rotate,
            augment_hflip=self.augment_hflip,
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
        # Discrete indices (tokens).
        x0_idx = batch["image"].astype(jnp.int32)

        # Anchor vectors (continuous) from the model's anchor table.
        x0_anchor = model.apply({"params": params}, x0_idx, method=model.embed)
        anchor_table = None
        if self.log_state_dependency:
            try:
                anchor_table = params["anchors"]["table"]
            except Exception:
                anchor_table = model.apply({"params": params}, method=model.anchor_table)

        def apply_fn(p, xt, t_img):
            return model.apply({"params": p}, xt, t_img, train=train)

        loss, metrics = ce_allocation_loss(
            key=rng,
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
        )

        return loss, metrics
