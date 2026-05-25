from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import jax
import jax.numpy as jnp

from sticky.models.sjd.losses import ce_allocation_loss
from sticky.models.sjd.schedule import ForwardSchedule
from sticky.models.sjd.sjd_elbo_loss import elbo_eta1_loss
from sticky.rng import PRNGKey
from sticky.tasks.base import Batch, Metrics, Task

_VALID_OBJECTIVES = ("ce", "elbo_eta1")

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

    NOTE: ``forward`` has a ``None`` default to avoid dataclass field-ordering
    errors when subclasses define required (no-default) fields. It MUST be
    supplied at construction time; passing ``forward=None`` will raise in
    ``__post_init__``.
    """

    # Forward dynamics (populated by the task factory)
    forward: Optional[ForwardSchedule] = None  # required at construction time
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
    # End-to-end ELBO mode (opt-in). Default 'ce' preserves the legacy path.
    # 'elbo_eta1' uses the SJD negative-ELBO at eta=1 (L_CE + rb_weight * L_RB
    # + prior_strength * Omega), giving unbiased gradients on log_w. Requires
    # learn_log_w=True, jump.eta == 1.0, jump.blur_kernel is None.
    objective: str = "ce"
    rb_weight: float = 1.0
    rb_share_sample: bool = True
    prior_strength: float = 0.0

    # Whether the model.apply signature takes the time arg positionally (CIFAR-10)
    # vs as a keyword ``t=`` (OpenWebText, Sudoku).
    # Declared as ClassVar so @dataclass does not treat it as an instance field.
    _t_passes_positionally: ClassVar[bool] = False

    def __post_init__(self):
        if self.forward is None:
            raise ValueError(
                f"{type(self).__name__} requires a ForwardSchedule; got forward=None"
            )
        obj = str(self.objective)
        if obj not in _VALID_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {_VALID_OBJECTIVES}, got {obj!r}"
            )
        if obj == "elbo_eta1":
            if not bool(self.learn_log_w):
                raise ValueError(
                    "objective='elbo_eta1' requires learn_log_w=True; the whole "
                    "point of this mode is to learn the per-anchor hazard w(a)."
                )
            jump = self.forward.jump
            eta_val = float(getattr(jump, "eta", 1.0))
            if abs(eta_val - 1.0) > 1e-6:
                raise ValueError(
                    f"objective='elbo_eta1' is written for eta=1 only; got "
                    f"jump.eta={eta_val}. The lam_hat closed form is no longer "
                    f"y-free at eta<1 (see appendix `rem:eta-lt-1`)."
                )
            blur_kernel = getattr(jump, "blur_kernel", None)
            if blur_kernel is not None:
                raise ValueError(
                    "objective='elbo_eta1' requires W=I (jump.blur_kernel=None); "
                    f"got blur_kernel={type(blur_kernel).__name__}."
                )
        super_post = getattr(super(), "__post_init__", None)
        if super_post is not None:
            super_post()

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

        if str(self.objective) == "elbo_eta1":
            loss, metrics = elbo_eta1_loss(
                key=key_loss,
                params=params,
                apply_fn=apply_fn,
                x0_anchor=x0_anchor,
                x0_idx=x0_idx,
                beta=self.forward.beta,
                hazard=self.forward.hazard,
                jump=self.forward.jump,
                T=float(self.forward.T),
                anchor_log_w=anchor_log_w,
                given_mask=self._given_mask(batch),
                time_sampling=str(self.time_sampling),
                t_floor=float(self.t_floor),
                rb_share_sample=bool(self.rb_share_sample),
                rb_weight=float(self.rb_weight),
                prior_strength=float(self.prior_strength),
            )
        else:
            loss, metrics = ce_allocation_loss(
                key=key_loss,
                params=params,
                apply_fn=apply_fn,
                x0_anchor=x0_anchor,
                x0_idx=x0_idx,
                beta=self.forward.beta,
                hazard=self.forward.hazard,
                jump=self.forward.jump if bool(self.log_state_dependency) else None,
                anchor_table=anchor_table,
                state_dep_log_ratio_clip=float(self.state_dep_log_ratio_clip),
                T=float(self.forward.T),
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
        metrics = self._extra_metrics(metrics, batch, x0_idx)
        return loss, metrics
