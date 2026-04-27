"""Multi-axis CE allocation loss.

Generalizes `ce_allocation_loss` over a `StateLayout`: each anchored axis
contributes an SJD paired-corruption forward law and an NLL term; each
unanchored axis follows its declared `dynamics` and emits per-axis
diagnostics but never enters the loss.

Single-anchored-axis layouts (e.g., the cell-only Sudoku task) reduce
to the scalar code path of `ce_allocation_loss` modulo bookkeeping.
The Sudoku slack-augmented layout has one anchored axis (`cells`) and
three unanchored axes (`row_slacks`, `col_slacks`, `box_slacks`).

Dynamics dial (B5):
  * "sjd"           : `sample_pair` (paired SJD corruption). The only
                       legal choice for axes that contribute to NLL.
  * "vp"            : `sample_slack_pair` (pure VP-SDE perturbation),
                       no anchor stickiness.
  * "deterministic" : value at t > 0 is computed from other axes via
                       a per-layout callback. Stubbed in this PR; the
                       full design lives in slack_corruption.py's
                       Phase D TODO.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.rng import PRNGKey

from .corruption import sample_pair
from .sdes import _expand_like, alpha_sigma
from .slack_corruption import sample_slack_pair
from .state_dependency import state_dependency_metrics
from .state_layout import StateLayout

Array = jnp.ndarray
Metrics = Dict[str, Array]


def _sample_t(key, T: float, B: int, time_sampling: str) -> Array:
    if time_sampling == "antithetic" and B >= 2:
        half_B = (B + 1) // 2
        t_half = jax.random.uniform(
            key, shape=(half_B,), minval=0.0, maxval=float(T)
        )
        t_complement = float(T) - t_half
        return jnp.concatenate([t_half, t_complement], axis=0)[:B]
    return jax.random.uniform(key, shape=(B,), minval=0.0, maxval=float(T))


def ce_allocation_loss_multi_axis(
    key: PRNGKey,
    params,
    apply_fn: Callable[[object, Mapping[str, Array], Array], Tuple[Mapping[str, Array], dict]],
    layout: StateLayout,
    x0_anchors: Mapping[str, Array],
    x0_indices: Mapping[str, Array],
    x0_unanchored: Mapping[str, Array],
    beta,
    hazard,
    T: float,
    *,
    jump=None,
    anchor_tables: Optional[Mapping[str, Array]] = None,
    state_dep_log_ratio_clip: float = 10.0,
    given_masks: Optional[Mapping[str, Array]] = None,
    time_sampling: str = "uniform",
    loss_weighting: str = "uniform",
    anchor_log_w: Optional[Mapping[str, Array]] = None,
    axis_loss_weights: Optional[Mapping[str, float]] = None,
) -> Tuple[Array, Metrics]:
    """Compute the multi-axis CE allocation loss.

    Parameters
    ----------
    apply_fn : callable
        `apply_fn(params, state_dict, t_img)` -> (per_axis_logits, aux). The
        per-axis logits dict must have an entry for each anchored axis with
        shape (B, site_count, vocab_size).
    x0_anchors : per anchored-axis embeddings of shape (B, site_count, dim).
    x0_indices : per anchored-axis token indices of shape (B, site_count).
    x0_unanchored : per unanchored-axis values of shape (B, site_count, dim).
    given_masks : optional per anchored-axis bool masks of shape
        (B, site_count); True = clue / context (do not contribute to NLL).
    anchor_tables : optional per anchored-axis (vocab_size, dim) table for
        DHM / state-dependence diagnostics.
    anchor_log_w : optional per anchored-axis (vocab_size,) log frequency
        weights, mirroring the cell-only `ce_allocation_loss`.
    axis_loss_weights : optional per anchored-axis scalar coefficients
        applied to that axis's mean NLL before summing. Defaults to 1.0.
    """
    if hazard is None:
        raise ValueError(
            "ce_allocation_loss_multi_axis requires a hazard schedule."
        )
    if jump is None:
        raise ValueError(
            "ce_allocation_loss_multi_axis requires a VPMatchedGaussianJump."
        )

    anchored_axes = layout.anchored_axes
    if not anchored_axes:
        raise ValueError("layout must have at least one anchored axis.")

    given_masks = dict(given_masks or {})
    anchor_tables = dict(anchor_tables or {})
    anchor_log_w = dict(anchor_log_w or {})
    axis_loss_weights = dict(axis_loss_weights or {})

    any_x0 = next(iter(x0_anchors.values()))
    B = int(any_x0.shape[0])

    # Split the RNG: one for time, one per axis (anchored or unanchored).
    n_axis_keys = len(layout.axes)
    keys = jax.random.split(key, 1 + n_axis_keys)
    key_t = keys[0]
    axis_keys = {axis.name: keys[1 + i] for i, axis in enumerate(layout.axes)}

    t_img = _sample_t(key_t, T=float(T), B=B, time_sampling=str(time_sampling))

    # Forward sampling per axis (dynamics dial).
    state_t: Dict[str, Array] = {}
    never_unstuck_masks: Dict[str, Array] = {}
    for axis in layout.axes:
        akey = axis_keys[axis.name]
        if axis.dynamics == "sjd":
            x0 = x0_anchors[axis.name]
            log_w_v = anchor_log_w.get(axis.name, None)
            if log_w_v is not None:
                log_w_per_site = jnp.take(
                    jnp.asarray(log_w_v, dtype=jnp.float32),
                    jnp.asarray(x0_indices[axis.name], dtype=jnp.int32),
                    axis=0,
                )
            else:
                log_w_per_site = None
            x_t, never_unstuck = sample_pair(
                akey, x0, t_img, beta, hazard, jump,
                log_w_per_site=log_w_per_site,
            )
            state_t[axis.name] = x_t
            never_unstuck_masks[axis.name] = never_unstuck
        elif axis.dynamics == "vp":
            x0 = x0_unanchored[axis.name]
            x_t = sample_slack_pair(akey, x0, t_img, beta)
            state_t[axis.name] = x_t
        elif axis.dynamics == "deterministic":
            raise NotImplementedError(
                f"AxisSpec.dynamics='deterministic' is stubbed in this PR. "
                f"See slack_corruption.py's Phase D TODO for the joint "
                f"un-sticking design that this dial point would enable."
            )
        else:  # pragma: no cover — guarded by AxisSpec validation
            raise AssertionError(f"unknown dynamics: {axis.dynamics}")

    # Build the classifier input by overwriting committed cells with their
    # x0 anchor (per anchored axis, mirroring ce_allocation_loss).
    classifier_input: Dict[str, Array] = dict(state_t)
    effective_loss_masks: Dict[str, Array] = {}
    for axis in anchored_axes:
        x0 = x0_anchors[axis.name]
        gmask = given_masks.get(
            axis.name, jnp.zeros_like(x0_indices[axis.name], dtype=jnp.bool_)
        )
        gmask = jnp.asarray(gmask, dtype=jnp.bool_)
        if gmask.shape != x0_indices[axis.name].shape:
            raise ValueError(
                f"given_masks[{axis.name!r}] expected shape "
                f"{tuple(x0_indices[axis.name].shape)}; got {tuple(gmask.shape)}."
            )
        committed = jnp.logical_or(never_unstuck_masks[axis.name], gmask)
        classifier_input[axis.name] = jnp.where(
            committed[..., None], x0, state_t[axis.name]
        )
        effective_loss_masks[axis.name] = (~gmask) & (~never_unstuck_masks[axis.name])
        given_masks[axis.name] = gmask

    per_axis_logits, _ = apply_fn(params, classifier_input, t_img)

    # Per-axis NLL; sum with weights.
    metrics: Metrics = {
        "t/mean": jnp.mean(t_img),
        "t/std": jnp.std(t_img),
    }
    total_loss = jnp.asarray(0.0, dtype=jnp.float32)
    for axis in anchored_axes:
        x0_idx = jnp.asarray(x0_indices[axis.name], dtype=jnp.int32)
        logits = per_axis_logits[axis.name]
        if logits.shape[:2] != x0_idx.shape:
            raise ValueError(
                f"axis {axis.name!r}: logits shape {tuple(logits.shape[:2])} "
                f"does not match x0_idx shape {tuple(x0_idx.shape)}."
            )
        logp = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(logp, x0_idx[..., None], axis=-1).squeeze(-1)
        loss_mask = effective_loss_masks[axis.name]
        loss_weight = loss_mask.astype(jnp.float32)
        if loss_weighting == "alpha_deriv":
            alpha_t, _ = alpha_sigma(beta, t_img)
            beta_t = beta(t_img)
            w_t = 0.5 * beta_t * alpha_t / jnp.maximum(1.0 - alpha_t, 1e-8)
            w_t = w_t / jnp.maximum(jnp.mean(w_t), 1e-8)
            w_t = _expand_like(w_t, loss_weight)
            loss_weight = loss_weight * w_t
        loss_count = jnp.sum(loss_weight)
        denom = jnp.maximum(loss_count, 1.0)
        per_axis_loss = jnp.sum(nll * loss_weight) / denom

        pred_idx = jnp.argmax(logp, axis=-1).astype(jnp.int32)
        correct = (pred_idx == x0_idx).astype(jnp.float32)
        per_axis_acc = jnp.sum(correct * loss_weight) / denom

        suffix_mask = ~given_masks[axis.name]
        suffix_count = jnp.maximum(jnp.sum(suffix_mask.astype(jnp.float32)), 1.0)
        per_axis_frac_active = jnp.sum(loss_weight) / suffix_count

        prefix = f"loss/{axis.name}"
        metrics[f"{prefix}/ce_nll_bits"] = per_axis_loss / jnp.log(2.0)
        metrics[f"{prefix}/acc_top1"] = per_axis_acc
        metrics[f"{prefix}/frac_active"] = per_axis_frac_active
        metrics[f"{prefix}/frac_never_unstuck"] = jnp.mean(
            never_unstuck_masks[axis.name].astype(jnp.float32)
        )

        anchor_table = anchor_tables.get(axis.name, None)
        if anchor_table is not None:
            sd = state_dependency_metrics(
                y=state_t[axis.name],
                t_img=t_img,
                logits=logits,
                uncommitted_mask=loss_mask,
                anchor_table=anchor_table,
                beta=beta,
                jump=jump,
                hazard=hazard,
                log_ratio_clip=float(state_dep_log_ratio_clip),
            )
            metrics[f"state_dep/{axis.name}/log_ratio_mean"] = sd[
                "state_dep/log_ratio_mean"
            ]
            metrics[f"state_dep/{axis.name}/log_ratio_std"] = sd[
                "state_dep/log_ratio_std"
            ]
        else:
            nan = jnp.asarray(jnp.nan, dtype=jnp.float32)
            metrics[f"state_dep/{axis.name}/log_ratio_mean"] = nan
            metrics[f"state_dep/{axis.name}/log_ratio_std"] = nan

        coeff = float(axis_loss_weights.get(axis.name, 1.0))
        total_loss = total_loss + coeff * per_axis_loss

    # Per-unanchored-axis residual diagnostics (no NLL contribution).
    for axis in layout.unanchored_axes:
        if axis.dynamics == "vp":
            x0 = x0_unanchored[axis.name]
            x_t = state_t[axis.name]
            alpha_t_a, sigma_t_a = alpha_sigma(beta, t_img)
            alpha_t_b = alpha_t_a[:, None, None]
            residual = x_t - alpha_t_b * x0
            per_b_mse = jnp.mean(jnp.square(residual), axis=(1, 2))
            metrics[f"loss/{axis.name}/residual_l2"] = jnp.mean(jnp.sqrt(per_b_mse))
            metrics[f"loss/{axis.name}/sigma_t_mean"] = jnp.mean(sigma_t_a)

    metrics["loss"] = total_loss
    return total_loss, metrics
