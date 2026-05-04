from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.rng import PRNGKey

from .sdes import _expand_like, alpha_sigma
from .corruption import sample_pair
from .state_dependency import state_dependency_metrics

Array = jnp.ndarray
Metrics = Dict[str, Array]


def _hazard_deriv_weights(
    *,
    t_img: Array,
    beta,
    hazard,
    anchor_log_w: Optional[Array],
    x0_idx: Array,
    target_like: Array,
) -> Array:
    """Pre-normalization SJD per-site loss weighting.

    With factored hazard λ_t(a) = β(t)·w(a) and S_a(t) = exp(−w(a)·H(t)),
    returns w_t(a) = β(t)·w(a)·S_a(t) / (1 − S_a(t)) per token. Uses
    ``jnp.expm1`` for numerical stability when w(a)·H(t) is small. When
    ``anchor_log_w is None``, w(a) ≡ 1 and the formula collapses to
    β(t)·exp(−H)/(1−exp(−H)).
    """
    H_t = hazard.cum(t_img)
    beta_t = beta(t_img)
    if anchor_log_w is not None:
        log_w_site = jnp.take(
            jnp.asarray(anchor_log_w, dtype=jnp.float32),
            jnp.asarray(x0_idx, dtype=jnp.int32),
            axis=0,
        )
        w_a = jnp.exp(log_w_site)
        H_b = _expand_like(H_t, w_a)
        b_b = _expand_like(beta_t, w_a)
        log_Sa = -w_a * H_b
        one_minus_Sa = -jnp.expm1(log_Sa)
        w_t = b_b * w_a * jnp.exp(log_Sa) / jnp.maximum(one_minus_Sa, 1e-8)
    else:
        one_minus_S = -jnp.expm1(-H_t)
        w_t = beta_t * jnp.exp(-H_t) / jnp.maximum(one_minus_S, 1e-8)
    return _expand_like(w_t, target_like)


def ce_allocation_loss(
    key: PRNGKey,
    params,
    apply_fn: Callable[[object, Array, Array], Tuple[Array, dict]],
    x0_anchor: Array,
    x0_idx: Array,
    beta,
    hazard: Optional[object],
    T: float,
    jump: Optional[object] = None,
    anchor_table: Optional[Array] = None,
    state_dep_log_ratio_clip: float = 10.0,
    given_mask: Optional[Array] = None,
    time_sampling: str = "uniform",
    loss_weighting: str = "uniform",
    anchor_log_w: Optional[Array] = None,
    pass_noisy_mask_to_model: bool = False,
) -> Tuple[Array, Metrics]:
    if hazard is None:
        raise ValueError("ce_allocation_loss requires a hazard schedule.")
    if jump is None:
        raise ValueError("ce_allocation_loss requires a VPMatchedGaussianJump.")

    x0_idx = x0_idx.astype(jnp.int32)
    if given_mask is None:
        given_mask = jnp.zeros_like(x0_idx, dtype=jnp.bool_)
    else:
        given_mask = jnp.asarray(given_mask, dtype=jnp.bool_)
        if given_mask.shape != x0_idx.shape:
            raise ValueError(
                "given_mask must match x0_idx shape, got "
                f"{given_mask.shape} vs {x0_idx.shape}."
            )

    B = int(x0_anchor.shape[0])

    key_t, key_vp = jax.random.split(key, 2)

    # One global time per example. For odd B, ceil(B/2) base samples and
    # their complements give >=B elements; truncate to exactly B.
    if time_sampling == "antithetic" and B >= 2:
        half_B = (B + 1) // 2
        t_half = jax.random.uniform(
            key_t, shape=(half_B,), minval=0.0, maxval=float(T)
        )
        t_complement = float(T) - t_half
        t_img = jnp.concatenate([t_half, t_complement], axis=0)[:B]
    else:
        t_img = jax.random.uniform(key_t, shape=(B,), minval=0.0, maxval=float(T))

    # SJD-paired corruption: x_t ~ N(alpha(t) a, v_t(tau) I) with tau drawn
    # from the truncated lam(tau)*S(tau) density on (0, t); never_unstuck_mask
    # marks sites where the particle had not unstuck by t (x_t == x0_anchor).
    if anchor_log_w is not None:
        log_w_per_site = jnp.take(
            jnp.asarray(anchor_log_w, dtype=jnp.float32),
            jnp.asarray(x0_idx, dtype=jnp.int32),
            axis=0,
        )
    else:
        log_w_per_site = None
    x_t, never_unstuck_mask = sample_pair(
        key_vp, x0_anchor, t_img, beta, hazard, jump,
        log_w_per_site=log_w_per_site,
    )

    # never_unstuck_mask is the canonical commit mask: an independent
    # Bernoulli(S(t)) draw would double-mask the loss to (1-S(t))^2 rather
    # than (1-S(t)). given_mask reserves conditional context tokens.
    committed = jnp.logical_or(never_unstuck_mask, given_mask)

    # At never-unstuck sites x_t == x0_anchor by construction, so this where
    # is a no-op at those sites.
    x_in = jnp.where(committed[..., None], x0_anchor, x_t)

    if pass_noisy_mask_to_model:
        logits, _ = apply_fn(params, x_in, t_img, ~committed)
    else:
        logits, _ = apply_fn(params, x_in, t_img)
    logp = jax.nn.log_softmax(logits, axis=-1)

    # NLL against the true token/anchor index.
    nll = -jnp.take_along_axis(logp, x0_idx[..., None], axis=-1).squeeze(-1)
    suffix_mask = ~given_mask
    # committed = never_unstuck_mask | given_mask, so
    #   ~committed & suffix = ~(never_unstuck | given) & ~given
    #                       = (~never_unstuck) & suffix_mask
    # i.e. masking via never_unstuck alone is exact and avoids the prior
    # double-mask bug.
    effective_loss_mask = suffix_mask & (~never_unstuck_mask)
    effective_loss_weight = effective_loss_mask.astype(jnp.float32)
    w_t = None
    if loss_weighting == "alpha_deriv":
        alpha_t, _ = alpha_sigma(beta, t_img)
        beta_t = beta(t_img)
        w_t = 0.5 * beta_t * alpha_t / jnp.maximum(1.0 - alpha_t, 1e-8)
        w_t = w_t / jnp.maximum(jnp.mean(w_t), 1e-8)
        w_t = _expand_like(w_t, effective_loss_weight)
        effective_loss_weight = effective_loss_weight * w_t
    elif loss_weighting == "hazard_deriv":
        # SJD ELBO weighting: makes log_w enter the loss differentiably (the
        # only other use of log_w is the non-differentiable sample_pair draw).
        w_t = _hazard_deriv_weights(
            t_img=t_img,
            beta=beta,
            hazard=hazard,
            anchor_log_w=anchor_log_w,
            x0_idx=x0_idx,
            target_like=effective_loss_weight,
        )
        mask_f = effective_loss_mask.astype(jnp.float32)
        mean_w = jnp.sum(w_t * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)
        w_t = w_t / jnp.maximum(mean_w, 1e-8)
        effective_loss_weight = effective_loss_weight * w_t
    effective_loss_count = jnp.sum(effective_loss_weight)
    denom = jnp.maximum(effective_loss_count, 1.0)
    loss = jnp.sum(nll * effective_loss_weight) / denom

    # Diagnostics: top-1 accuracy on active sites and fraction of sites that
    # actually contributed to the loss (= 1 - frac_committed when no given_mask).
    pred_idx = jnp.argmax(logp, axis=-1).astype(jnp.int32)
    correct = (pred_idx == x0_idx).astype(jnp.float32)
    acc_top1 = jnp.sum(correct * effective_loss_weight) / denom

    suffix_count = jnp.maximum(jnp.sum(suffix_mask.astype(jnp.float32)), 1.0)
    frac_uncommitted = jnp.sum(effective_loss_weight) / suffix_count

    # Whitelisted training metrics (Prompt B). The dict order below is the
    # canonical order used by the W&B / console logging. Anything not in this
    # set is computed for use elsewhere but not emitted as a logged metric.
    metrics: Metrics = {
        "loss": loss,
        "loss/ce_nll_bits": loss / jnp.log(2.0),
        "loss/acc_top1": acc_top1,
        "loss/frac_active": frac_uncommitted,
        "loss/frac_never_unstuck": jnp.mean(
            never_unstuck_mask.astype(jnp.float32)
        ),
        "t/mean": jnp.mean(t_img),
        "t/std": jnp.std(t_img),
    }

    # state_dep/log_ratio_{mean,std} are always emitted when both jump and
    # anchor_table are available (the active-mask diagnostic on the DHM
    # log-ratio). If either is missing we still emit the keys with NaN so the
    # whitelist is invariant.
    if (jump is not None) and (anchor_table is not None):
        sd = state_dependency_metrics(
            y=x_t,
            t_img=t_img,
            logits=logits,
            uncommitted_mask=effective_loss_mask,
            anchor_table=anchor_table,
            beta=beta,
            jump=jump,
            hazard=hazard,
            log_ratio_clip=float(state_dep_log_ratio_clip),
            anchor_log_w=anchor_log_w,
        )
        metrics["state_dep/log_ratio_mean"] = sd["state_dep/log_ratio_mean"]
        metrics["state_dep/log_ratio_std"] = sd["state_dep/log_ratio_std"]
    else:
        nan = jnp.asarray(jnp.nan, dtype=jnp.float32)
        metrics["state_dep/log_ratio_mean"] = nan
        metrics["state_dep/log_ratio_std"] = nan

    return loss, metrics
