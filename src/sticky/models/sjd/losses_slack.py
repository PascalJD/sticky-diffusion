"""Slack-aware CE allocation loss for the Sudoku SJD task.

Mirrors `ce_allocation_loss` but additionally noises slack sites via a pure
VP-SDE forward kernel and concatenates them into the classifier input. The
NLL is computed on the cell sites only — slack sites have no anchor set and
contribute zero NLL by construction. `state_dependency_metrics` is computed
on cell logits + cell x_t against the simplex-vertex anchor table.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.rng import PRNGKey

from .corruption import sample_pair
from .sdes import _expand_like, alpha_sigma
from .slack_corruption import sample_slack_pair
from .state_dependency import state_dependency_metrics

Array = jnp.ndarray
Metrics = Dict[str, Array]


def ce_allocation_loss_with_slack(
    key: PRNGKey,
    params,
    apply_fn: Callable[[object, Array, Array, Array], Tuple[Array, dict]],
    x0_anchor: Array,
    x0_idx: Array,
    slack_x0: Array,
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
) -> Tuple[Array, Metrics]:
    if hazard is None:
        raise ValueError("ce_allocation_loss_with_slack requires a hazard schedule.")
    if jump is None:
        raise ValueError(
            "ce_allocation_loss_with_slack requires a VPMatchedGaussianJump."
        )

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

    slack_x0 = jnp.asarray(slack_x0, dtype=jnp.float32)
    if slack_x0.ndim != 3 or slack_x0.shape[1:] != (27, 9):
        raise ValueError(
            f"slack_x0 must have shape (B, 27, 9); got {tuple(slack_x0.shape)}."
        )
    if slack_x0.shape[0] != x0_anchor.shape[0]:
        raise ValueError(
            "slack_x0 batch dim must match x0_anchor; got "
            f"{slack_x0.shape[0]} vs {x0_anchor.shape[0]}."
        )

    B = int(x0_anchor.shape[0])

    key_t, key_vp_cell, key_vp_slack = jax.random.split(key, 3)

    if time_sampling == "antithetic" and B >= 2:
        half_B = (B + 1) // 2
        t_half = jax.random.uniform(
            key_t, shape=(half_B,), minval=0.0, maxval=float(T)
        )
        t_complement = float(T) - t_half
        t_img = jnp.concatenate([t_half, t_complement], axis=0)[:B]
    else:
        t_img = jax.random.uniform(key_t, shape=(B,), minval=0.0, maxval=float(T))

    if anchor_log_w is not None:
        log_w_per_site = jnp.take(
            jnp.asarray(anchor_log_w, dtype=jnp.float32),
            jnp.asarray(x0_idx, dtype=jnp.int32),
            axis=0,
        )
    else:
        log_w_per_site = None
    cell_x_t, never_unstuck_mask = sample_pair(
        key_vp_cell, x0_anchor, t_img, beta, hazard, jump,
        log_w_per_site=log_w_per_site,
    )
    slack_x_t = sample_slack_pair(key_vp_slack, slack_x0, t_img, beta)

    committed = jnp.logical_or(never_unstuck_mask, given_mask)
    cell_x_in = jnp.where(committed[..., None], x0_anchor, cell_x_t)

    logits, _ = apply_fn(params, cell_x_in, slack_x_t, t_img)
    cell_logits = logits[:, : x0_anchor.shape[1], :]
    logp = jax.nn.log_softmax(cell_logits, axis=-1)

    nll = -jnp.take_along_axis(logp, x0_idx[..., None], axis=-1).squeeze(-1)
    suffix_mask = ~given_mask
    effective_loss_mask = suffix_mask & (~never_unstuck_mask)
    effective_loss_weight = effective_loss_mask.astype(jnp.float32)
    if loss_weighting == "alpha_deriv":
        alpha_t, _ = alpha_sigma(beta, t_img)
        beta_t = beta(t_img)
        w_t = 0.5 * beta_t * alpha_t / jnp.maximum(1.0 - alpha_t, 1e-8)
        w_t = w_t / jnp.maximum(jnp.mean(w_t), 1e-8)
        w_t = _expand_like(w_t, effective_loss_weight)
        effective_loss_weight = effective_loss_weight * w_t
    effective_loss_count = jnp.sum(effective_loss_weight)
    denom = jnp.maximum(effective_loss_count, 1.0)
    loss = jnp.sum(nll * effective_loss_weight) / denom

    pred_idx = jnp.argmax(logp, axis=-1).astype(jnp.int32)
    correct = (pred_idx == x0_idx).astype(jnp.float32)
    acc_top1 = jnp.sum(correct * effective_loss_weight) / denom

    suffix_count = jnp.maximum(jnp.sum(suffix_mask.astype(jnp.float32)), 1.0)
    frac_uncommitted = jnp.sum(effective_loss_weight) / suffix_count

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

    if (jump is not None) and (anchor_table is not None):
        sd = state_dependency_metrics(
            y=cell_x_t,
            t_img=t_img,
            logits=cell_logits,
            uncommitted_mask=effective_loss_mask,
            anchor_table=anchor_table,
            beta=beta,
            jump=jump,
            hazard=hazard,
            log_ratio_clip=float(state_dep_log_ratio_clip),
        )
        metrics["state_dep/log_ratio_mean"] = sd["state_dep/log_ratio_mean"]
        metrics["state_dep/log_ratio_std"] = sd["state_dep/log_ratio_std"]
    else:
        nan = jnp.asarray(jnp.nan, dtype=jnp.float32)
        metrics["state_dep/log_ratio_mean"] = nan
        metrics["state_dep/log_ratio_std"] = nan

    # Per-coordinate residual against the VP-marginal mean alpha(t) * slack_x0.
    # We report the per-coordinate root-mean-square — an unbiased estimator
    # of sqrt(E[r^2]) = sigma(t) for a Gaussian residual, regardless of t.
    # This replaces the earlier `slack_l2_to_ones` metric, which was
    # uninformative (it collapsed toward alpha(t) * sqrt(d) at large t purely
    # because the VP mean shrinks toward 0, even though nothing was wrong).
    alpha_t_slack, sigma_t_slack = alpha_sigma(beta, t_img)
    alpha_t_b = alpha_t_slack[:, None, None]
    slack_residual = slack_x_t - alpha_t_b * slack_x0
    # Per-batch-element RMS over (group, dim) ≈ sigma(t_b) for a VP residual,
    # then average over the batch ≈ E_t[sigma(t)] = `loss/slack_sigma_t_mean`.
    # Scalar-RMS-across-the-whole-batch would converge to sqrt(E[sigma^2(t)])
    # (by Jensen), which is uniformly higher and harder to interpret.
    per_b_mse = jnp.mean(jnp.square(slack_residual), axis=(1, 2))  # (B,)
    metrics["loss/slack_residual_l2"] = jnp.mean(jnp.sqrt(per_b_mse))
    metrics["loss/slack_sigma_t_mean"] = jnp.mean(sigma_t_slack)

    return loss, metrics
