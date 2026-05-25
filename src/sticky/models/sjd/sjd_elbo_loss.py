"""End-to-end SJD ELBO training loss at eta=1 (appendix `alg:sjd-e2e-eta1`).

Implements
    L = L_CE + rb_weight * L_RB + prior_strength * Omega(w)
where
    L_CE  = E_{X_0, t, eps} [ lam(t) * w(X_0) * S_{X_0}(t) * (-log P_theta(X_0 | Y, t)) ]
    L_RB  = E_{X_0, t, eps'} [ (1 - S_{X_0}(t)) *
                              Sum_a lam_hat_t(a) * (P_theta(a | X_t, t) - 1{a = X_0}) ]
    Omega = Sum_a (w(a) - 1)^2

with Y = alpha(t) * E(X_0) + sigma(t) * eps drawn from the VP perturbation
kernel and X_t drawn from the same kernel (independently or shared via
`rb_share_sample=True`).

This is opt-in via task field `objective="elbo_eta1"` and exists alongside
the legacy `ce_allocation_loss` (whose gradient on log_w is biased).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.rng import PRNGKey

from .sdes import alpha_sigma
from .plugin_intensity import log_lambda_hat_per_anchor

Array = jnp.ndarray
Metrics = Dict[str, Array]


def q_t_sample(
    *,
    key: PRNGKey,
    x0_anchor: Array,
    t_img: Array,
    beta,
    jump,
) -> Array:
    """Draw ``Y = alpha(t) * E(X_0) + sigma(t) * eps`` from the VP perturbation
    kernel ``q_t(.|X_0)``.

    Always Gaussian; no Bernoulli filter and no tau-sampling. This is the
    eta=1 closed-form collapse of the SJD off-anchor conditional ``p_t^ac``
    (appendix eq:c1-collapse), with the ``(1 - S_a)`` prefactor that the
    ELBO weight absorbs.
    """
    x0_anchor = jnp.asarray(x0_anchor, dtype=jnp.float32)
    alpha_t, sigma_t = alpha_sigma(beta, t_img)  # (B,), (B,)
    site_axes = (1,) * (x0_anchor.ndim - 1)
    alpha_b = alpha_t.reshape((-1,) + site_axes)
    sigma_b = sigma_t.reshape((-1,) + site_axes)
    anchor_blurred = jump.apply_blur(x0_anchor)  # identity at W=I (enforced upstream)
    eps = jax.random.normal(key, shape=x0_anchor.shape, dtype=jnp.float32)
    return alpha_b * anchor_blurred + sigma_b * eps


def elbo_eta1_loss(
    *,
    key: PRNGKey,
    params,
    apply_fn: Callable[[object, Array, Array], Tuple[Array, dict]],
    x0_anchor: Array,
    x0_idx: Array,
    beta,
    hazard,
    jump,
    T: float,
    anchor_log_w: Array,
    given_mask: Optional[Array] = None,
    time_sampling: str = "uniform",
    t_floor: float = 1e-3,
    rb_share_sample: bool = True,
    rb_weight: float = 1.0,
    prior_strength: float = 0.0,
    eps: float = 1e-8,
) -> Tuple[Array, Metrics]:
    if anchor_log_w is None:
        raise ValueError("elbo_eta1_loss requires anchor_log_w (the learned hazard).")

    x0_idx = jnp.asarray(x0_idx, dtype=jnp.int32)
    if given_mask is None:
        given_mask = jnp.zeros_like(x0_idx, dtype=jnp.bool_)
    else:
        given_mask = jnp.asarray(given_mask, dtype=jnp.bool_)
        if given_mask.shape != x0_idx.shape:
            raise ValueError(
                f"given_mask shape {given_mask.shape} != x0_idx shape {x0_idx.shape}"
            )
    suffix_mask = ~given_mask

    B = int(x0_anchor.shape[0])
    key_t, key_ce, key_rb_eps = jax.random.split(key, 3)

    # Sample t ~ Unif(0, T); clamp away from endpoints to keep lam_hat finite.
    if time_sampling == "antithetic" and B >= 2:
        half_B = (B + 1) // 2
        t_half = jax.random.uniform(
            key_t, shape=(half_B,), minval=0.0, maxval=float(T)
        )
        t_img = jnp.concatenate([t_half, float(T) - t_half], axis=0)[:B]
    else:
        t_img = jax.random.uniform(key_t, shape=(B,), minval=0.0, maxval=float(T))
    t_floor_f = float(t_floor)
    t_img = jnp.clip(t_img, t_floor_f, float(T) - t_floor_f)

    log_w_table = jnp.asarray(anchor_log_w, dtype=jnp.float32)  # (K,)
    K = int(log_w_table.shape[0])

    # Per-site log w(X_0,i).
    log_w_per_site = jnp.take(log_w_table, x0_idx, axis=0)  # (B, *site)
    w_per_site = jnp.exp(log_w_per_site)

    # Per-site survival and hazard.
    site_axes = (1,) * (x0_idx.ndim - 1)
    H_t = jnp.asarray(hazard.cum(t_img), dtype=jnp.float32).reshape((B,) + site_axes)
    lam_t = jnp.asarray(hazard.lam(t_img), dtype=jnp.float32).reshape((B,) + site_axes)
    log_S_a = -w_per_site * H_t              # (B, *site)
    S_a = jnp.exp(log_S_a)

    # CE weight = lam(t) * w(X_0) * S_{X_0}(t), the appendix's -m_dot_t(a)/p_0(a).
    # NOT lam_hat = lam*w*S/(1-S) — the (1-S) factor is NOT in the denominator,
    # because we draw Y ~ q_t directly (no Bernoulli filter), so the per-site
    # never-unstuck probability S_a is absorbed into the deterministic weight.
    w_ce = lam_t * w_per_site * S_a          # (B, *site)

    # ---- L_CE ----
    Y = q_t_sample(
        key=key_ce, x0_anchor=x0_anchor, t_img=t_img, beta=beta, jump=jump,
    )
    logits, _ = apply_fn(params, Y, t_img)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, x0_idx[..., None], axis=-1).squeeze(-1)

    mask_f = suffix_mask.astype(jnp.float32)
    ce_num = jnp.sum(mask_f * w_ce * nll)
    # stop_gradient'd per-batch normalizer: a pure scalar rescaling, so the
    # gradient direction in (theta, log_w) is unchanged. This is the
    # *intentional* stop-grad. Contrast the *unintentional* stop-grad on the
    # Bernoulli `never_unstuck_mask` at corruption.py:88-91, which the
    # elbo_eta1 mode exists to remove — there the (1-S_a) factor lived in a
    # non-differentiable filter, biasing the w-gradient.
    ce_den = jax.lax.stop_gradient(jnp.maximum(jnp.sum(mask_f * w_ce), eps))
    loss_ce = ce_num / ce_den

    # ---- L_RB ----
    # Reparametrized form: ell_RB_i = (1 - S_a) * Sum_a lam_hat(a) * (P_theta(a) - 1{a=x0_i}).
    # The deterministic (1 - S_a) weight replaces appendix Algorithm 1's
    # `1{u >= S_a}` Bernoulli filter so the w-gradient flows analytically
    # through S_a as well as lam_hat. Same trick as L_CE: absorb the per-site
    # never-unstuck probability into a closed-form factor instead of into a
    # non-differentiable threshold. Same expectation; clean reparametrized
    # gradient. (Algorithm 1's wording implies this without spelling it out.)
    if rb_share_sample:
        X_t_rb = Y
        logits_rb = logits
    else:
        X_t_rb = q_t_sample(
            key=key_rb_eps, x0_anchor=x0_anchor, t_img=t_img,
            beta=beta, jump=jump,
        )
        logits_rb, _ = apply_fn(params, X_t_rb, t_img)
    p_theta_rb = jax.nn.softmax(logits_rb, axis=-1)  # (B, *site, K)

    one_minus_S = 1.0 - S_a  # (B, *site); deterministic, w-differentiable.

    # Per-(B, anchor) lam_hat (eta=1 closed form), broadcast to per-site.
    log_lam_hat_BK = log_lambda_hat_per_anchor(
        t_img=t_img, hazard=hazard, log_w_table=log_w_table,
    )  # (B, K)
    lam_hat_BK = jnp.exp(log_lam_hat_BK)
    lam_hat_per_site = lam_hat_BK.reshape((B,) + site_axes + (K,))
    lam_hat_per_site = jnp.broadcast_to(lam_hat_per_site, p_theta_rb.shape)

    one_hot_x0 = jax.nn.one_hot(x0_idx, K, dtype=jnp.float32)
    rb_inner = jnp.sum(
        lam_hat_per_site * (p_theta_rb - one_hot_x0), axis=-1
    )  # (B, *site)
    rb_per_site = mask_f * one_minus_S * rb_inner
    rb_num = jnp.sum(rb_per_site)
    # Normalizer = count of active sites, w-independent (no stop_gradient needed).
    rb_count = jnp.maximum(jnp.sum(mask_f), 1.0)
    loss_rb = rb_num / rb_count

    # ---- Prior Omega(w) = lambda_Omega * sum_a (w(a) - 1)^2 ----
    w_table = jnp.exp(log_w_table)
    loss_prior = jnp.asarray(prior_strength, dtype=jnp.float32) * jnp.sum(
        (w_table - 1.0) ** 2
    )

    loss = loss_ce + jnp.asarray(rb_weight, dtype=jnp.float32) * loss_rb + loss_prior

    metrics: Metrics = {
        "loss": loss,
        "loss/ce": loss_ce,
        "loss/rb": loss_rb,
        "loss/prior": loss_prior,
        "loss/ce_nll_bits": loss_ce / jnp.log(2.0),
        # ce_num is the un-normalized weighted sum; ce_den is the stop-gradiented
        # per-batch normalizer. Useful for gradient-correctness tests: FD does
        # not respect stop_gradient, so verify gradient parity on ce_num.
        "loss/ce_num": ce_num,
        "loss/ce_den": ce_den,
        "loss/rb_num": rb_num,
        "t/mean": jnp.mean(t_img),
        "t/std": jnp.std(t_img),
        "log_w/mean": jnp.mean(log_w_table),
        "log_w/std": jnp.std(log_w_table),
        "log_w/range": jnp.max(log_w_table) - jnp.min(log_w_table),
    }
    return loss, metrics
