from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from sticky.models.common.discrete_mixture import categorical_sample_from_probs, normalize_probs
from sticky.rng import PRNGKey

from .hazard import HazardSchedule, lambda_off_star
from .jump import VPMatchedGaussianJump
from .sdes import _expand_like, alpha_sigma
from .corruption import mixture_logpdf


Array = jnp.ndarray


def _log1mexp(x: Array, eps: float = 1e-7) -> Array:
    """Numerically stable log(1 - exp(x)) for x <= 0 (Maechler 2012).

    `eps` clamps `x` away from 0 so the result stays finite when the per-anchor
    survival is essentially 1 (e.g. evaluating the plug-in at very small t).
    1e-7 is roughly one float32 epsilon below 0, the smallest clamp that the
    float32 representation can actually distinguish from 0.
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    log2 = jnp.log(jnp.asarray(2.0, dtype=jnp.float32))
    safe_a = jnp.minimum(x, -jnp.asarray(eps, dtype=jnp.float32))  # ensure x < 0
    return jnp.where(
        safe_a < -log2,
        jnp.log1p(-jnp.exp(safe_a)),
        jnp.log(-jnp.expm1(safe_a)),
    )


def _sample_anchor_from_probs(
    *,
    key: PRNGKey,
    choice_probs: Array,
    alloc_mode: str,
    categorical_sampling_policy: str = "legacy_low",
) -> Array:
    if alloc_mode == "argmax":
        return jnp.argmax(choice_probs, axis=-1).astype(jnp.int32)
    if alloc_mode == "sample":
        return categorical_sample_from_probs(
            key,
            choice_probs,
            policy=categorical_sampling_policy,
        )
    raise ValueError(f"Unknown alloc_mode={alloc_mode!r}")


def dhm_log_ratio(
    *,
    y: Array,
    t_img: Array,
    anchors: Any,
    beta: Any,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    log_ratio_clip: float | None = 10.0,
    tau_grid_size: int = 32,
) -> Array:
    """log r_a(y) - log p_t(y | a) per anchor a — the DHM target log-ratio.

    Numerator:   log r_a(y) = log N(y; alpha(t) a, eta^2 sigma^2(t) I), the
                 VP-matched un-sticking kernel of the SJD forward process.
    Denominator: log p_t(y | a), the SJD off-anchor conditional (mixture over
                 unstick times) computed by `mixture_logpdf` via tau-quadrature
                 with `tau_grid_size` midpoint nodes. Under a frequency-weighted
                 hazard ``lambda_tau(a) = beta(tau) * w(a)``, the integrand
                 carries the per-anchor ``log w(a)`` and ``S(tau)^{w(a)}`` factors;
                 ``anchors.effective_log_w`` is forwarded as ``log_w_anchor``.

    Output shape: (B, *site_shape, L).
    """
    y = jnp.asarray(y, dtype=jnp.float32)
    a_table = jnp.asarray(anchors.table_float, dtype=jnp.float32)
    # Tolerate duck-typed namespaces that lack effective_log_w.
    _eff_log_w = getattr(anchors, "effective_log_w", None)
    if _eff_log_w is None:
        _eff_log_w = jnp.zeros((int(a_table.shape[0]),), dtype=jnp.float32)
    log_w_table = jnp.asarray(_eff_log_w, dtype=jnp.float32)  # (L,)

    def per_anchor(a_l: Array, log_w_l: Array) -> Array:
        a_b = jnp.broadcast_to(a_l, y.shape)
        log_r = jump.logpdf(y, a_b, t_img)
        log_p = mixture_logpdf(
            y=y,
            anchor=a_b,
            t=t_img,
            beta=beta,
            hazard=hazard,
            jump=jump,
            tau_grid_size=int(tau_grid_size),
            log_w_anchor=log_w_l,
        )
        return log_r - log_p

    log_ratio = jax.vmap(per_anchor, in_axes=(0, 0))(a_table, log_w_table)
    log_ratio = jnp.moveaxis(log_ratio, 0, -1)
    if log_ratio_clip is not None:
        log_ratio = jnp.clip(
            log_ratio, -float(log_ratio_clip), float(log_ratio_clip)
        )
    return log_ratio


def _full_hazard_and_allocation(
    *,
    logits: Array,
    y: Array,
    t_img: Array,
    anchors: Any,
    beta: Any,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    logit_temperature: float,
    log_ratio_clip: float | None,
    eps: float,
    sitewise_lam_off: Array | None = None,
    tau_grid_size: int = 32,
) -> Tuple[Array, Array]:
    """Dense plug-in implementation that materializes [B, S, L]-scale tensors.

    When ``sitewise_lam_off`` is None, ``lambda_off_star(hazard, t_img)`` is
    used as a scalar per-example baseline broadcast to all sites. When
    provided (shape (B, *site_shape) or broadcastable), it replaces the
    scalar `lam_off` per-site. Because the same log_lam_base value is added
    to every class/anchor entry at a site, the softmax-normalized
    ``choice_probs`` is invariant under this substitution — only ``lam_total``
    (and hence commit probability) changes.
    """
    if logit_temperature <= 0.0:
        raise ValueError(f"logit_temperature must be > 0, got {logit_temperature}")

    logits = logits.astype(jnp.float32)
    B = logits.shape[0]
    site_shape = logits.shape[1:-1]
    L = int(logits.shape[-1])
    logp = jax.nn.log_softmax(logits, axis=-1).astype(jnp.float32)
    logp_flat = logp.reshape((B, -1, L))

    log_ratio = dhm_log_ratio(
        y=y,
        t_img=t_img,
        anchors=anchors,
        beta=beta,
        jump=jump,
        log_ratio_clip=log_ratio_clip,
        hazard=hazard,
        tau_grid_size=tau_grid_size,
    ).reshape((B, -1, L))

    has_freq_weighting = getattr(anchors, "log_w", None) is not None
    if has_freq_weighting and sitewise_lam_off is not None:
        raise ValueError(
            "frequency_weighted_hazard (anchors.log_w) and state-dependent "
            "sitewise_lam_off are mutually exclusive."
        )

    if sitewise_lam_off is None and not has_freq_weighting:
        lam_base = lambda_off_star(hazard, t_img).astype(jnp.float32)[:, None, None]
        log_lam_base = jnp.log(
            jnp.maximum(lam_base, jnp.asarray(eps, dtype=jnp.float32))
        )
    elif sitewise_lam_off is not None:
        sw = jnp.asarray(sitewise_lam_off, dtype=jnp.float32)
        if sw.shape[0] != B:
            raise ValueError(
                f"sitewise_lam_off batch dim {sw.shape[0]} does not match logits {B}."
            )
        # Flatten site axes and add a trailing anchor broadcast dim.
        lam_base = sw.reshape((B, -1, 1))
        log_lam_base = jnp.log(
            jnp.maximum(lam_base, jnp.asarray(eps, dtype=jnp.float32))
        )
    else:
        # Per-anchor log_lam_base[a] = log lam(t) + log w(a) + w(a)*log S(t)
        #                              - log(1 - S(t)^{w(a)}),
        # where lam(t) = hazard.lam(t) is the base (anchor-agnostic) hazard
        # rate. Collapses to log lambda_off_star(t) when w(a) = 1 for all a.
        log_w_table = jnp.asarray(anchors.effective_log_w, dtype=jnp.float32)  # (L,)
        w_table = jnp.exp(log_w_table)  # (L,)
        lam_t = jnp.asarray(hazard.lam(t_img), dtype=jnp.float32)  # (B,)
        log_lam_t = jnp.log(
            jnp.maximum(lam_t, jnp.asarray(eps, dtype=jnp.float32))
        )[:, None, None]  # (B, 1, 1)
        log_S_t = -jnp.asarray(hazard.cum(t_img), dtype=jnp.float32)[:, None, None]
        log_S_a_t = w_table[None, None, :] * log_S_t  # (B, 1, L)
        # _log1mexp's eps is a "x < 0" guard against float-resolution issues;
        # leave it at the function default (1e-7) rather than the much
        # smaller `eps` (1e-20) used as a max-floor on the lam_t / sum_a path.
        log_one_minus_S_a_t = _log1mexp(log_S_a_t)
        log_lam_base = (
            log_lam_t
            + log_w_table[None, None, :]
            + log_S_a_t
            - log_one_minus_S_a_t
        )  # (B, 1, L)
    logw_raw = log_lam_base + logp_flat + log_ratio
    lam_total_flat = jnp.maximum(
        jnp.sum(jnp.exp(logw_raw), axis=-1),
        jnp.asarray(eps, dtype=jnp.float32),
    )

    if logit_temperature == 1.0:
        logp_choice_flat = logp_flat
    else:
        logp_choice = jax.nn.log_softmax(
            logits / jnp.asarray(logit_temperature, dtype=jnp.float32),
            axis=-1,
        ).astype(jnp.float32)
        logp_choice_flat = logp_choice.reshape((B, -1, L))

    logw_choice = log_lam_base + logp_choice_flat + log_ratio
    choice_probs_flat = jax.nn.softmax(logw_choice, axis=-1).astype(jnp.float32)

    lam_total = lam_total_flat.reshape((B,) + site_shape)
    choice_probs = choice_probs_flat.reshape((B,) + site_shape + (L,))
    return lam_total, normalize_probs(choice_probs)


def dhm_target(
    *,
    y: Array,
    t_img: Array,
    true_anchor_idx: Array,
    anchors: Any,
    beta: Any,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    log_ratio_clip: float = 10.0,
    eps: float = 1e-20,
    tau_grid_size: int = 32,
) -> Array:
    """The DHM target intensity hat-lambda_t(y, x_0): a per-site scalar that
    plug-in training matches in expectation. Uses the SJD off-anchor mixture
    in the denominator (see `dhm_log_ratio`)."""
    log_ratio = dhm_log_ratio(
        y=y,
        t_img=t_img,
        anchors=anchors,
        beta=beta,
        jump=jump,
        log_ratio_clip=log_ratio_clip,
        hazard=hazard,
        tau_grid_size=tau_grid_size,
    )
    idx = jnp.asarray(true_anchor_idx, dtype=jnp.int32)
    gathered = jnp.take_along_axis(log_ratio, idx[..., None], axis=-1)[..., 0]
    if getattr(anchors, "log_w", None) is None:
        lam_base = _expand_like(
            lambda_off_star(hazard, t_img).astype(jnp.float32), gathered
        )
    else:
        # Per-anchor base rate evaluated at the true anchor:
        #   lam_base[a] = lam(t) * w(a) * S(t)^{w(a)} / (1 - S(t)^{w(a)}),
        # where lam(t) = hazard.lam(t) is the base (anchor-agnostic) rate.
        log_w_table = jnp.asarray(anchors.effective_log_w, dtype=jnp.float32)  # (L,)
        log_w_at = jnp.take(log_w_table, idx, axis=0)  # (B, *site_shape)
        w_at = jnp.exp(log_w_at)
        log_S_t = -jnp.asarray(hazard.cum(t_img), dtype=jnp.float32)
        log_S_t_b = _expand_like(log_S_t, gathered)
        log_S_a_t = w_at * log_S_t_b
        # See note above: _log1mexp's eps is a float-resolution guard that
        # should not be conflated with the plug-in's max-floor `eps`.
        log_one_minus_S_a_t = _log1mexp(log_S_a_t)
        lam_t = jnp.asarray(hazard.lam(t_img), dtype=jnp.float32)
        log_lam_t = jnp.log(
            jnp.maximum(lam_t, jnp.asarray(eps, dtype=jnp.float32))
        )
        log_lam_t_b = _expand_like(log_lam_t, gathered)
        log_lam_base = log_lam_t_b + log_w_at + log_S_a_t - log_one_minus_S_a_t
        lam_base = jnp.exp(log_lam_base)
    return jnp.maximum(lam_base, jnp.asarray(eps, dtype=jnp.float32)) * jnp.exp(gathered)


def plugin_hazard_and_allocation(
    *,
    logits: Array,
    y: Array,
    t_img: Array,
    anchors: Any,
    beta: Any,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    logit_temperature: float = 1.0,
    log_ratio_clip: float | None = 10.0,
    eps: float = 1e-20,
    sitewise_lam_off: Array | None = None,
    tau_grid_size: int = 32,
) -> Tuple[Array, Array]:
    """Total plug-in hazard backvec-lambda_t^star(y) and per-anchor allocation
    pi_t^star(a | y), per Section 3.1 of the paper.

    Returns (lam_total, choice_probs).
    """
    return _full_hazard_and_allocation(
        logits=logits,
        y=y,
        t_img=t_img,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        logit_temperature=logit_temperature,
        log_ratio_clip=log_ratio_clip,
        eps=eps,
        sitewise_lam_off=sitewise_lam_off,
        tau_grid_size=tau_grid_size,
    )


def plugin_hazard_and_anchor(
    *,
    key: PRNGKey,
    logits: Array,
    y: Array,
    t_img: Array,
    anchors: Any,
    beta: Any,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    alloc_mode: str,
    categorical_sampling_policy: str = "legacy_low",
    logit_temperature: float = 1.0,
    log_ratio_clip: float = 10.0,
    eps: float = 1e-20,
    sitewise_lam_off: Array | None = None,
    tau_grid_size: int = 32,
) -> Tuple[Array, Array]:
    """Wrapper returning (lam_total, sampled_anchor_index) per site."""
    lam_total, choice_probs = plugin_hazard_and_allocation(
        logits=logits,
        y=y,
        t_img=t_img,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        logit_temperature=logit_temperature,
        log_ratio_clip=log_ratio_clip,
        eps=eps,
        sitewise_lam_off=sitewise_lam_off,
        tau_grid_size=tau_grid_size,
    )
    a_idx = _sample_anchor_from_probs(
        key=key,
        choice_probs=choice_probs,
        alloc_mode=alloc_mode,
        categorical_sampling_policy=categorical_sampling_policy,
    )
    return lam_total, a_idx
