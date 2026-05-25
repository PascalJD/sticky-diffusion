from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp

from sticky.models.sjd.losses import ce_allocation_loss
from sticky.rng import PRNGKey
from sticky.tasks.base import Batch, Metrics, Task

Array = jnp.ndarray


@dataclass(kw_only=True)
class SJDTaskBase(Task):
    """Base class for SJD tasks: shared loss_fn and forward-process knobs.

    Subclasses provide dataset-specific:
      - _extract_x0_idx(batch) -> Array (clean token indices)
      - _given_mask(batch) -> Optional[Array]
      - _extra_metrics(metrics, batch, x0_idx) -> dict (defaults to {})
      - data loading (make_dataloaders)
      - dataset-specific dataclass fields

    NOTE: ``beta`` has a ``None`` default to avoid dataclass field-ordering
    errors when subclasses define required (no-default) fields. It MUST be
    supplied at construction time; passing ``beta=None`` will raise at the
    first training step. Use ``__post_init__`` in a subclass or the factory
    to enforce this if desired.
    """

    # Forward dynamics (populated by the task factory)
    beta: Callable[[Array], Array] = None  # required; default avoids ordering issues
    hazard: Optional[Any] = None
    jump: Optional[Any] = None
    T: float = 1.0
    # DHM knobs
    log_state_dependency: bool = True
    state_dep_log_ratio_clip: float = 10.0
    time_sampling: str = "uniform"
    loss_weighting: str = "uniform"
    anchor_log_w: Optional[Array] = None
    learn_log_w: bool = False
    t_floor: float = 1e-3
    log_anchor_log_w_stats: bool = True
    pass_noisy_mask_to_model: bool = False

    # Whether the model.apply signature takes the time arg positionally (CIFAR-10)
    # vs as a keyword ``t=`` (OpenWebText, Sudoku).
    _t_passes_positionally: bool = False

    def _extract_x0_idx(self, batch: Batch) -> Array:
        raise NotImplementedError("Subclasses must implement _extract_x0_idx")

    def _given_mask(self, batch: Batch) -> Optional[Array]:
        return None

    def _extra_metrics(
        self, metrics: Dict[str, Any], batch: Batch, x0_idx: Array
    ) -> Dict[str, Any]:
        return metrics

    def loss_fn(
        self,
        *,
        rng: PRNGKey,
        model,
        params,
        batch: Batch,
        train: bool,
    ):
        key_loss, key_dropout = jax.random.split(rng)
        x0_idx = self._extract_x0_idx(batch)

        x0_anchor = model.apply({"params": params}, x0_idx, method=model.embed)
        anchor_table = None
        if bool(self.log_state_dependency):
            try:
                anchor_table = params["anchors"]["table"]
            except Exception:
                anchor_table = model.apply({"params": params}, method=model.anchor_table)

        def apply_fn(p, xt, t_img, noisy_position_mask=None):
            extra_kwargs = {}
            if noisy_position_mask is not None:
                extra_kwargs["noisy_position_mask"] = noisy_position_mask
            rngs = {"dropout": key_dropout} if train else None
            if self._t_passes_positionally:
                if train:
                    return model.apply(
                        {"params": p}, xt, t_img, train=True, **extra_kwargs, rngs=rngs
                    )
                return model.apply({"params": p}, xt, t_img, train=False, **extra_kwargs)
            if train:
                return model.apply(
                    {"params": p}, xt, t=t_img, train=True, **extra_kwargs, rngs=rngs
                )
            return model.apply({"params": p}, xt, t=t_img, train=False, **extra_kwargs)

        if self.learn_log_w:
            anchor_log_w = model.apply({"params": params}, method=model.anchor_log_w)
        else:
            anchor_log_w = self.anchor_log_w

        loss, metrics = ce_allocation_loss(
            key=key_loss,
            params=params,
            apply_fn=apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=self.beta,
            hazard=self.hazard,
            jump=self.jump if bool(self.log_state_dependency) else None,
            anchor_table=anchor_table,
            state_dep_log_ratio_clip=float(self.state_dep_log_ratio_clip),
            T=float(self.T),
            given_mask=self._given_mask(batch),
            time_sampling=str(self.time_sampling),
            loss_weighting=str(self.loss_weighting),
            anchor_log_w=anchor_log_w,
            t_floor=float(self.t_floor),
            log_anchor_log_w_stats=bool(self.log_anchor_log_w_stats),
            pass_noisy_mask_to_model=bool(self.pass_noisy_mask_to_model),
        )

        metrics = dict(metrics)
        metrics["loss"] = loss
        return self._extra_metrics(metrics, batch, x0_idx)
