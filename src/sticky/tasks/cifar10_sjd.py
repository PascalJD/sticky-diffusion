from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.models.sjd.losses import ce_allocation_loss
from sticky.tasks.base import TaskSpec
from sticky.tasks.tfds_discrete_image import TFDSDiscreteImageTaskBase

Array = jnp.ndarray


@dataclass
class CIFAR10SJDTask(TFDSDiscreteImageTaskBase):
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
    task_name: str = "sjd_cifar10"
    dataset_name: str = "cifar10"
    train_split: str = "train"
    eval_split: str = "test"
    include_label: bool | str = "auto"
    dummy_label_value: int = -1
    hazard: Optional[Any] = None
    jump: Optional[Any] = None
    T: float = 1.0
    log_state_dependency: bool = True
    state_dep_log_ratio_clip: float = 10.0
    time_sampling: str = "uniform"
    loss_weighting: str = "uniform"
    anchor_log_w: Optional[Array] = None
    pass_noisy_mask_to_model: bool = False

    # data augmentation
    augment_enabled: bool = True
    augment_prob: float = 0.15
    augment_rotate: bool = True
    augment_hflip: bool = True
    augment_eval: bool = False

    def __post_init__(self):
        self._spec = self._build_image_task_spec(name=str(self.task_name))

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def make_dataloaders(self, seed: int):
        return super().make_dataloaders(seed=seed)

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

        def apply_fn(p, xt, t_img, noisy_position_mask=None):
            extra_kwargs = {}
            if noisy_position_mask is not None:
                extra_kwargs["noisy_position_mask"] = noisy_position_mask
            if train:
                return model.apply(
                    {"params": p},
                    xt,
                    t_img,
                    train=True,
                    **extra_kwargs,
                    rngs={"dropout": key_dropout},
                )
            return model.apply({"params": p}, xt, t_img, train=False, **extra_kwargs)

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
            time_sampling=str(self.time_sampling),
            loss_weighting=str(self.loss_weighting),
            anchor_log_w=self.anchor_log_w,
            pass_noisy_mask_to_model=bool(self.pass_noisy_mask_to_model),
        )

        return loss, metrics
