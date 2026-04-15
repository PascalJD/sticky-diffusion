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
    forward_site_hazard_mode: str = "none"
    teacher_confidence_metric: str = "margin"
    teacher_w_min: float = 0.5
    teacher_w_max: float = 2.0
    teacher_neutral_weight: float = 1.0
    teacher_log_metrics: bool = True

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

    @property
    def requires_teacher_params(self) -> bool:
        return str(self.forward_site_hazard_mode) == "teacher_margin"

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
        teacher_params: Any = None,
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

        teacher_mode = str(self.forward_site_hazard_mode) == "teacher_margin"
        x0_anchor_teacher = None
        teacher_apply_fn = None
        # Eval fallback: when teacher mode is declared but no teacher params
        # were threaded, drop to the baseline scalar hazard for this call.
        effective_mode = str(self.forward_site_hazard_mode)
        if teacher_mode and teacher_params is None:
            effective_mode = "none"
        if teacher_mode and teacher_params is not None:
            x0_anchor_teacher = model.apply(
                {"params": teacher_params}, x0_idx, method=model.embed
            )

            def teacher_apply_fn(p, xt, t_img):  # noqa: E306
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
            time_sampling=str(self.time_sampling),
            loss_weighting=str(self.loss_weighting),
            forward_site_hazard_mode=effective_mode,
            x0_anchor_teacher=x0_anchor_teacher,
            teacher_params=teacher_params,
            teacher_apply_fn=teacher_apply_fn,
            teacher_confidence_metric=str(self.teacher_confidence_metric),
            teacher_w_min=float(self.teacher_w_min),
            teacher_w_max=float(self.teacher_w_max),
            teacher_neutral_weight=float(self.teacher_neutral_weight),
            teacher_log_metrics=bool(self.teacher_log_metrics),
        )

        return loss, metrics
