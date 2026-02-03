# src/sticky/trainers/train_step.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import struct

from sticky.core.anchors import AnalogBitsAnchors
from sticky.core.hazard import HazardSchedule
from sticky.core.jump import GaussianJump
from sticky.trainers.losses import ce_allocation_loss, dhm_loss

Array = jnp.ndarray


@dataclass
class ForwardProcess:
    beta: object
    hazard: HazardSchedule
    jump: GaussianJump
    T: float = 1.0

@struct.dataclass
class DualTrainState:
    cls: train_state.TrainState
    haz: train_state.TrainState

@struct.dataclass
class StickyTrainState(train_state.TrainState):
    """Single shared model with logits + hazard."""
    pass
    
def train_step(
    rng: jax.random.PRNGKey,
    state: StickyTrainState,
    batch_x: Array,
    anchors: AnalogBitsAnchors,
    fwd: ForwardProcess,
    L: int,
    ce_weight: float = 1.0,
    dhm_weight: float = 0.05,
    use_dhm: bool = True,
) -> Tuple[StickyTrainState, Dict[str, Array]]:

    B = batch_x.shape[0]
    H = W = 28

    # Discretize to anchors
    x0_idx, x0_anc = anchors.encode_from_pixels(batch_x.reshape(B, H, W))

    def dhm_weight_fn(t_flat: Array) -> Array:
        lam_fwd = fwd.hazard.lam(t_flat)
        surv = fwd.hazard.surv(t_flat)
        w = lam_fwd * surv
        w = jnp.clip(w, 1e-8, None)
        return w

    do_dhm = bool(use_dhm) and float(dhm_weight) > 0.0

    def _loss(params, rng):
        k_ce, k_dhm = jax.random.split(rng, 2)

        # CE (allocator)
        loss_ce, ce_metrics = ce_allocation_loss(
            key=k_ce,
            params=params,
            apply_fn=state.apply_fn,
            x0_anchor=x0_anc,
            x0_idx=x0_idx,
            beta=fwd.beta,
            T=fwd.T,
        )

        # DHM (hazard)
        if do_dhm:
            loss_dhm, dhm_metrics = dhm_loss(
                key=k_dhm,
                params=params,
                apply_fn=state.apply_fn,
                anchors=anchors,
                x0_anchor=x0_anc,
                beta=fwd.beta,
                hazard=fwd.hazard,
                jump=fwd.jump,
                T=fwd.T,
                weight_fn=dhm_weight_fn,
            )

        total = ce_weight * loss_ce + (dhm_weight * loss_dhm if do_dhm else 0)

        metrics = dict(ce_metrics)
        if do_dhm:
            metrics.update(dhm_metrics)
            metrics["loss/DHM"] = loss_dhm
        else:
            metrics["loss/DHM"] = jnp.array(0.0, jnp.float32)

        metrics.update({
            "loss/total": total,
            "loss/CE": loss_ce,
        })

        return total, metrics

    (loss_val, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(state.params, rng)
    new_state = state.apply_gradients(grads=grads)

    metrics = {
        **metrics,
        "optim/grad_norm": optax.global_norm(grads),
        "optim/param_norm": optax.global_norm(state.params),
    }
    return new_state, metrics