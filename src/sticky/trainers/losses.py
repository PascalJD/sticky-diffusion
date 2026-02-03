# src/sticky/trainers/losses.py
from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from sticky.core.hazard import HazardSchedule, lam_off_star
from sticky.core.jump import GaussianJump, VPMatchedGaussianJump
from sticky.core.sde_vp import vp_perturb, vp_logpdf

Array = jnp.ndarray
Metrics = Dict[str, Array]


def ce_allocation_loss(
    key: jax.random.PRNGKey,
    params,
    apply_fn: Callable[[object, Array, Array], Array],
    x0_anchor: Array,
    x0_idx: Array,
    beta,
    T: float,
) -> Tuple[Array, Metrics]:
    """
    Cross-entropy loss for the allocator p_theta(a | x_t, t).

    We sample one global time per image (t_img: (B,)) and apply VP corruption
    to every pixel at that shared time.
    """
    B, H, W, d = x0_anchor.shape
    HW = H * W

    key_t, key_vp = jax.random.split(key, 2)

    # One global time per image
    t_img = jax.random.uniform(key_t, shape=(B,), minval=0.0, maxval=T)

    # Corrupt all pixels at that time (flatten for vp_perturb)
    t_flat = jnp.repeat(t_img, HW, axis=0)
    x0_flat = x0_anchor.reshape(B * HW, d) 
    xt_flat, _ = vp_perturb(key_vp, x0_flat, t_flat, beta)
    xt = xt_flat.reshape(B, H, W, d)

    logits, _ = apply_fn(params, xt, t_img)
    logp = jax.nn.log_softmax(logits, axis=-1)

    # NLL against true anchor index
    nll = -jnp.take_along_axis(logp, x0_idx[..., None], axis=-1).squeeze(-1)
    loss = jnp.mean(nll)

    # Diagnostics
    probs = jnp.exp(logp)
    pred_idx = jnp.argmax(probs, axis=-1)
    acc_top1 = jnp.mean(pred_idx == x0_idx)
    ent = -jnp.sum(probs * logp, axis=-1)
    alloc_entropy = jnp.mean(ent)

    metrics: Metrics = {
        "CE/acc_top1_event": acc_top1,
        "CE/alloc_entropy": alloc_entropy,
        "CE/ce_perplexity": jnp.exp(loss),
        "CE/ce_nll_bits": loss / jnp.log(2.0),
        "CE/frac_event": jnp.array(1.0, jnp.float32),  # we use all pixels now
    }
    return loss, metrics


def _safe_corr(x: Array, y: Array, eps: float = 1e-8) -> Array:
    """Numerically safe Pearson correlation for 1D-ish arrays."""
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)
    mx = jnp.mean(x)
    my = jnp.mean(y)
    vx = jnp.mean((x - mx) ** 2)
    vy = jnp.mean((y - my) ** 2)
    cov = jnp.mean((x - mx) * (y - my))
    return cov / jnp.maximum(jnp.sqrt(vx * vy), eps)


def dhm_loss(
    key: jax.random.PRNGKey,
    params,
    apply_fn: Callable[[object, Array, Array], Array],
    anchors,
    x0_anchor: Array,
    beta,
    hazard: HazardSchedule,
    jump: GaussianJump | VPMatchedGaussianJump,
    T: float,
    log_ratio_clip: float = 10.0,
    weight_fn: Callable[[Array], Array] | None = None,
    plugin_topk: int = 8,
    compute_plugin_metrics: bool = True,
    eps: float = 1e-12,
) -> Tuple[Array, Metrics]:
    """
    Discrete Hazard Matching (DHM) loss.

    We regress log rho(y,t) where:
      lam_hat(y,t) = lam_fwd(t) * S(t) * (r_t(y|a) / q_t(y|a))
      rho_hat(y,t) = lam_hat(y,t) / lam_off_star(t)

    We normalize rho_hat per image to have mean 1 across spatial locations.
    """
    B, H, W, d = x0_anchor.shape
    HW = H * W
    log_HW = jnp.log(jnp.asarray(HW, dtype=jnp.float32))

    key_t, key_vp = jax.random.split(key, 2)

    # Sample one global time per image
    t_img = jax.random.uniform(key_t, shape=(B,), minval=0.0, maxval=T)
    t_flat = jnp.repeat(t_img, HW, axis=0)

    # VP corruption X_t | X_0
    x0_flat = x0_anchor.reshape(B * HW, d)
    xt_flat, _ = vp_perturb(key_vp, x0_flat, t_flat, beta)

    # log r/q (forward densities)
    log_r = jump.logpdf(xt_flat, x0_flat, t_flat)
    log_q = vp_logpdf(xt_flat, x0_flat, t_flat, beta)

    log_ratio_raw = log_r - log_q
    clip_mask = jnp.abs(log_ratio_raw) > log_ratio_clip
    log_ratio_clip_frac = jnp.mean(clip_mask.astype(jnp.float32))
    log_ratio = jnp.clip(log_ratio_raw, -log_ratio_clip, log_ratio_clip)

    lam_fwd = hazard.lam(t_flat)
    surv = hazard.surv(t_flat)
    lam_hat_flat = lam_fwd * surv * jnp.exp(log_ratio)

    lam_off_flat = lam_off_star(hazard, t_flat) 
    lam_off_flat = jnp.maximum(lam_off_flat, eps)

    # rho_hat (unnormalized), then normalize per image so E_spatial[rho_hat]=1
    rho_hat_flat = lam_hat_flat / lam_off_flat
    rho_hat_img = rho_hat_flat.reshape(B, H, W)

    log_rho_hat_img = jnp.log(rho_hat_img + eps)
    log_mean_rho_hat = logsumexp(log_rho_hat_img, axis=(1, 2), keepdims=True) - log_HW
    log_rho_hat_img = log_rho_hat_img - log_mean_rho_hat
    log_rho_hat_flat = log_rho_hat_img.reshape(B * HW)

    # raw rho target scale diagnostics (before normalization)
    rho_hat_raw_mean_img = jnp.mean(rho_hat_img, axis=(1, 2))

    # Predict rho from intensity head (StickyHead already normalizes per image)
    xt_img = xt_flat.reshape(B, H, W, d)
    logits, rho_pred_img = apply_fn(params, xt_img, t_img)
    rho_pred_flat = rho_pred_img.reshape(B * HW)
    log_rho_pred_flat = jnp.log(rho_pred_flat + eps)

    # Weights, normalized to mean 1 for scale stability
    if weight_fn is None:
        w = jnp.ones_like(log_rho_hat_flat)
    else:
        w = weight_fn(t_flat)
        w = w / (jnp.mean(w) + eps)

    # Losses in log space
    loss_const = jnp.mean(w * jnp.square(log_rho_hat_flat))  # predictor=0 i.e. rho=1
    loss = jnp.mean(w * jnp.square(log_rho_pred_flat - log_rho_hat_flat))

    # Spatial diagnostics (normalized targets)
    log_rho_pred_img = log_rho_pred_flat.reshape(B, H, W)
    rho_hat_log_std_spatial = jnp.mean(jnp.std(log_rho_hat_img, axis=(1, 2)))
    rho_pred_log_std_spatial = jnp.mean(jnp.std(log_rho_pred_img, axis=(1, 2)))

    # Compare implied lambda_pred to lambda_hat
    lam_pred_flat = lam_off_flat * rho_pred_flat
    lam_mae = jnp.mean(jnp.abs(lam_pred_flat - lam_hat_flat))
    lam_pred_mean = jnp.mean(lam_pred_flat)
    lam_hat_mean = jnp.mean(lam_hat_flat)
    lam_corr = _safe_corr(lam_pred_flat, lam_hat_flat)

    # Rho mean diagnostics (rho_pred is already normalized by the model)
    rho_mean_img = jnp.mean(rho_pred_img, axis=(1, 2))
    rho_mean = jnp.mean(rho_mean_img)
    rho_std = jnp.std(rho_mean_img)
    rho_mean_abs_err = jnp.mean(jnp.abs(rho_mean_img - 1.0))

    metrics: Metrics = {
        "DHM/loss_rho": loss,
        "DHM/log_ratio_clip_frac": log_ratio_clip_frac,
        "DHM/loss_const_rho1": loss_const,
        "DHM/improvement_over_const": loss_const - loss,
        "DHM/log_rho_corr": _safe_corr(log_rho_pred_flat, log_rho_hat_flat),
        "DHM/rho_hat_log_std_spatial": rho_hat_log_std_spatial,
        "DHM/rho_pred_log_std_spatial": rho_pred_log_std_spatial,
        "DHM/rho_mean": rho_mean,
        "DHM/rho_std": rho_std,
        "DHM/rho_mean_abs_err": rho_mean_abs_err,
        "DHM/rho_hat_raw_mean": jnp.mean(rho_hat_raw_mean_img),
        "DHM/rho_hat_raw_mean_std": jnp.std(rho_hat_raw_mean_img),
        "DHM/lam_mae": lam_mae,
        "DHM/lam_corr": lam_corr,
        "DHM/lam_pred_mean": lam_pred_mean,
        "DHM/lam_hat_mean": lam_hat_mean,
    }

    # Plugin rho diagnostics (optional; costs extra compute)
    if compute_plugin_metrics:
        probs = jax.nn.softmax(jax.lax.stop_gradient(logits), axis=-1)
        L = probs.shape[-1]
        K = int(min(max(1, plugin_topk), L))

        topv, topi = jax.lax.top_k(probs, K)
        a_top = anchors.table_float[topi]

        # Broadcast time and y for logpdf calls
        t_top = t_img[:, None, None, None]
        y_top = xt_img[..., None, :]

        log_r_top = jump.logpdf(y_top, a_top, t_top)
        log_q_top = vp_logpdf(y_top, a_top, t_top, beta)
        ratio_top = jnp.exp(jnp.clip(log_r_top - log_q_top, -log_ratio_clip, log_ratio_clip))

        rho_plugin = jnp.sum(topv * ratio_top, axis=-1)
        log_rho_plugin_flat = jnp.log(rho_plugin.reshape(B * HW) + eps)

        metrics.update({
            "DHM/rho_plugin_log_std_spatial": jnp.mean(jnp.std(jnp.log(rho_plugin + eps), axis=(1, 2))),
            "DHM/rho_plugin_corr_with_target": _safe_corr(log_rho_plugin_flat, log_rho_hat_flat),
            "DHM/loss_plugin": jnp.mean(w * jnp.square(log_rho_plugin_flat - log_rho_hat_flat)),
        })

    return loss, metrics
