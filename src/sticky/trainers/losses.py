# src/sticky/trainers/losses.py
from __future__ import annotations
import jax, jax.numpy as jnp
from typing import Dict, Tuple
from ..core.sde_vp import vp_perturb

def dsm_loss(rng, x0, t, beta_fn, score_fn):
    xt, target = vp_perturb(rng, x0, t, beta_fn)
    pred = score_fn(xt, t, True)
    w = jnp.ones_like(t) 
    pred   = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    err = jnp.mean((pred - target)**2, axis=tuple(range(1, pred.ndim)))
    return jnp.mean(w * err), (xt, target, pred)

def cls_loss_coord(logits, marks):
    # logits: (B,d,L); marks: (B,2) with (i, ell) indices per event
    B, d, L = logits.shape
    i = marks[:, 0]
    ell = marks[:, 1]
    logits_i = logits[jnp.arange(B), i]  # (B,L)
    logp = logits_i - jax.scipy.special.logsumexp(
        logits_i, axis=-1, keepdims=True
    )
    nll = -jnp.take_along_axis(logp, ell[:, None], axis=-1).squeeze(-1)
    return jnp.mean(nll)

def intensity_nll(events_lambda, exposure_lambdas):
    # events_lambda: (M,1); exposure_lambdas: (B,1) used with factor T/B outside
    return (
        -jnp.mean(jnp.log(events_lambda + 1e-12)), 
        jnp.mean(exposure_lambdas)
    )