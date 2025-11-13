# src/sticky/trainers/train_step.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import jax, jax.numpy as jnp
import optax
from flax.training import train_state
import flax.linen as nn
from flax import struct 

from sticky.core.anchors import AnalogBitsAnchors
from sticky.core.sde_vp import make_beta, vp_perturb, vp_logpdf
from sticky.core.hazard import make_hazard_early, HazardSchedule
from sticky.core.jump import GaussianJump

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

def _safe_corr(x: Array, y: Array, eps: float = 1e-8) -> Array:
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)
    mx = jnp.mean(x)
    my = jnp.mean(y)
    vx = jnp.mean((x - mx) ** 2)
    vy = jnp.mean((y - my) ** 2)
    cov = jnp.mean((x - mx) * (y - my))
    return cov / jnp.maximum(jnp.sqrt(vx * vy), eps)

def train_step(
    rng: jax.random.PRNGKey,
    state: DualTrainState,
    batch_x: Array,
    anchors: AnalogBitsAnchors,
    fwd: ForwardProcess,
    L: int,
    ce_weight: float = 1.0,
    dhm_weight: float = 1.0,
) -> Tuple[DualTrainState, Dict[str, Array]]:

    B = batch_x.shape[0]
    H = W = 28
    d = anchors.d

    x0_idx, x0_anc = anchors.encode_from_pixels(batch_x.reshape(B, H, W))  # (B,H,W), (B,H,W,d)

    def _loss(params_cls, params_haz, rng):
        k_ce_t, k_ce_y, k_dhm_t, k_dhm_vp = jax.random.split(rng, 4)

        # CE (allocator)
        t_event_flat = fwd.hazard.first_event_time(k_ce_t, shape=(B * H * W,))
        t_event = t_event_flat.reshape(B, H, W)
        has_event = jnp.isfinite(t_event)  # (B,H,W)
        y_event = fwd.jump.sample(k_ce_y, x0_anc)
        t_img_ce = jnp.mean(jnp.where(has_event, t_event, 0.0), axis=(1, 2))

        logits_full = state.cls.apply_fn(params_cls, y_event, t_img_ce)
        logp_full = jax.nn.log_softmax(logits_full, axis=-1)  # (B,H,W,L)
        nll = -jnp.take_along_axis(
            logp_full, x0_idx[..., None], axis=-1
        ).squeeze(-1)  # (B,H,W)
        nll = jnp.where(has_event, nll, 0.0)
        denom_ce = jnp.maximum(jnp.sum(has_event), 1.0)
        loss_ce = jnp.sum(nll) / denom_ce

        # Diagnostics on allocator
        probs_full = jax.nn.softmax(logits_full, axis=-1)
        pred_idx = jnp.argmax(probs_full, axis=-1)
        correct = (pred_idx == x0_idx) & has_event
        acc_top1_event = jnp.sum(correct) / denom_ce

        ent = -jnp.sum(probs_full * jnp.log(jnp.maximum(probs_full, 1e-12)), axis=-1)
        alloc_entropy = jnp.sum(jnp.where(has_event, ent, 0.0)) / denom_ce
        ce_perplexity = jnp.exp(loss_ce)
        ce_nll_bits = loss_ce / jnp.log(2.0)

        # DHM (hazard)
        t_img = jax.random.uniform(k_dhm_t, shape=(B,), minval=0.0, maxval=fwd.T)
        t_flat = jnp.repeat(t_img, H * W, axis=0)
        x0_flat = x0_anc.reshape(B * H * W, d)
        xt_flat, _ = vp_perturb(k_dhm_vp, x0_flat, t_flat, fwd.beta)

        log_r = fwd.jump.logpdf(xt_flat, x0_flat)
        log_q = vp_logpdf(xt_flat, x0_flat, t_flat, fwd.beta)
        lam_fwd = fwd.hazard.lam(t_flat)
        surv = fwd.hazard.surv(t_flat)
        lam_hat_flat = lam_fwd * surv * jnp.exp(
            jnp.clip(log_r - log_q, -50.0, 50.0)
        )
        xt_img = xt_flat.reshape(B, H, W, d)
        lam_pred_img = state.haz.apply_fn(params_haz, xt_img, t_img)  # (B,H,W)
        lam_pred_flat = lam_pred_img.reshape(B * H * W)
        loss_dhm = jnp.mean(jnp.square(lam_pred_flat - lam_hat_flat))

        # Diagnostic
        lam_mae = jnp.mean(jnp.abs(lam_pred_flat - lam_hat_flat))
        lam_corr = _safe_corr(lam_pred_flat, lam_hat_flat)
        lam_pred_mean = jnp.mean(lam_pred_flat)
        lam_hat_mean = jnp.mean(lam_hat_flat)

        total = ce_weight * loss_ce + dhm_weight * loss_dhm
        metrics = dict(
            loss_total=total,
            loss_ce=loss_ce,
            loss_dhm=loss_dhm,
            frac_event=denom_ce / (B * H * W),
            acc_top1_event=acc_top1_event,
            alloc_entropy=alloc_entropy,
            ce_perplexity=ce_perplexity,
            ce_nll_bits=ce_nll_bits,
            lam_mae=lam_mae,
            lam_corr=lam_corr,
            lam_pred_mean=lam_pred_mean,
            lam_hat_mean=lam_hat_mean,
        )
        return total, metrics

    # Joint grad wrt both param trees
    params_all = (state.cls.params, state.haz.params)
    (loss_val, metrics), grads = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(*params_all, rng)
    grads_cls, grads_haz = grads

    # Apply updates
    new_cls = state.cls.apply_gradients(grads=grads_cls)
    new_haz = state.haz.apply_gradients(grads=grads_haz)
    new_state = DualTrainState(cls=new_cls, haz=new_haz)

    # Add grad/param norms after update (still on device; fine for jit)
    metrics = {
        **metrics,
        "grad_norm_cls": optax.global_norm(grads_cls),
        "grad_norm_haz": optax.global_norm(grads_haz),
        "param_norm_cls": optax.global_norm(state.cls.params),
        "param_norm_haz": optax.global_norm(state.haz.params),
    }
    return new_state, metrics