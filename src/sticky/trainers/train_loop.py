# src/sticky/trainers/train_loop.py
from __future__ import annotations
import functools, time
import jax, jax.numpy as jnp
import optax
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple

Array = jnp.ndarray

@dataclass
class TrainState:
    params_score: Any
    params_cls: Any
    params_int: Any
    opt: optax.GradientTransformation
    opt_state: optax.OptState

def make_optimizer(lr: float):
    return optax.adamw(lr)

def train_step_coord(
    state: TrainState,
    rng: jax.random.PRNGKey,
    batch_x0: Array, 
    batch_t: Array,
    beta_fn: Callable[[Array], Array],
    score_apply: Callable, 
    cls_apply: Callable, 
    int_apply: Callable,
    event_y: Array, 
    event_t: Array, 
    marks: Array,  # marks:(M,2)=(i,ell)
    T: float, 
    alpha_s: float, 
    alpha_c: float, 
    alpha_l: float,
):
    from .losses import dsm_loss, cls_loss_coord, intensity_nll
    def loss_fn(params_score, params_cls, params_int):
        rng1, _ = jax.random.split(rng)
        # DSM
        score_fun = lambda x,t,train: score_apply(
            {'params': params_score}, x, t, train
        )
        Ldsm, (xt, _, _) = dsm_loss(
            rng1, batch_x0, batch_t, beta_fn, score_fun
        )
        # Classifier (per-pixel logits)
        logits = cls_apply({'params': params_cls}, event_y, event_t, True)  # (M,d,L)
        Lcls = cls_loss_coord(logits, marks)
        # Intensity
        lam_events = int_apply({'params': params_int}, event_y, event_t, True)  # (M,1)
        lam_expo = int_apply({'params': params_int}, xt, batch_t, True)  # (B,1)
        Lnll_evt, Lexp = intensity_nll(lam_events, lam_expo)
        Lint = Lnll_evt + (T / xt.shape[0]) * Lexp
        loss = alpha_s * Ldsm + alpha_c * Lcls + alpha_l * Lint
        return loss, {'L_dsm': Ldsm, 'L_cls': Lcls, 'L_int': Lint}

    grads_fn = jax.value_and_grad(loss_fn, argnums=(0,1,2), has_aux=True)
    (loss, logs), grads = grads_fn(
        state.params_score, state.params_cls, state.params_int
    )
    updates, opt_state = state.opt.update(
        grads, 
        state.opt_state, 
        (state.params_score, state.params_cls, state.params_int)
    )
    params_score, params_cls, params_int = optax.apply_updates(
        (state.params_score, state.params_cls, state.params_int), updates
    )
    new_state = TrainState(
        params_score, params_cls, params_int, state.opt, opt_state
    )
    return new_state, loss, logs