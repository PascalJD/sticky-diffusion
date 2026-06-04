"""End-to-end SJD ELBO training loss at eta=1 (CORRECTED appendix
`alg:sjd-e2e-eta1`).

Implements the THREE-term ELBO:
    L = L_CE + rb_weight * L_RB + L_score + prior_strength * Omega(w)
where
    L_CE    = E_{X_0, t, eps}  [ lam(t) * w(X_0) * S_{X_0}(t) *
                                  (-log P_theta(X_0 | Y, t)) ]
    L_RB    = E_{X_0, t, eps'} [ (1 - S_{X_0}(t)) *
                                  Sum_a lam_hat_t(a) * (P_theta(a | X_t, t)
                                                        - 1{a = X_0}) ]
    L_score = E_{X_0, t, eps}  [ (1 - S_{X_0}(t)) * (g_t^2 / 2)
                                  * ||s_theta - grad log p^ac_t(.|X_0)||^2 ]
    Omega   = Sum_a (w(a) - 1)^2

L_score is the missing term that makes d_w L vanish at the consistent
hazard. The original appendix claim "d_w L_score = 0" (proposition
prop:score-no-train(i)) was WRONG: the (1 - S_a(t)) occupancy weight makes
L_score w-differentiable. Without L_score, d_w (L_CE + L_RB) = -d_w L_score
!= 0 at the Bayes classifier, producing a systematic push on log_w with no
restoring force (see plan v3 Correction 1).

The score-target residual is wrapped in stop_gradient so L_score
contributes only to the w-gradient, not theta: the score head is still fit
by cross-entropy only. Side note: this also blocks L_score's contribution
to the anchor-table gradient. Fine while anchors are frozen
(anchor.learnable=false); if anchors are ever made learnable under this
objective, revisit.

All three terms share a 1/N normalizer so the joint loss is an unbiased
Monte Carlo estimate of the ELBO integrand.

Y = alpha(t) * E(X_0) + sigma(t) * eps is drawn from the VP perturbation
kernel; X_t for L_RB shares the sample (rb_share_sample=True) or is drawn
independently.

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
    anchor_table: Optional[Array] = None,
    given_mask: Optional[Array] = None,
    time_sampling: str = "uniform",
    t_floor: float = 1e-3,
    rb_share_sample: bool = True,
    rb_weight: float = 1.0,
    prior_strength: float = 0.0,
    score_w_stop_gradient: bool = False,
    rb_theta_stop_gradient: bool = False,
    eps: float = 1e-8,
) -> Tuple[Array, Metrics]:
    if anchor_log_w is None:
        raise ValueError("elbo_eta1_loss requires anchor_log_w (the learned hazard).")
    if anchor_table is None:
        raise ValueError(
            "elbo_eta1_loss requires anchor_table (the NORMALIZED anchor table "
            "E(.); needed for the L_score residual ||abar_theta - E(X_0)||^2). "
            "Pass model.apply({'params': params}, method=model.anchor_table)."
        )

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
    # 1/N normalization, identical to L_RB / L_score. (Plan v3 H5 + Correction
    # 1.) The previous stop_gradient(sum(w_ce)) normalizer was correct for
    # theta-direction (w_ce doesn't depend on theta) but wrong for log_w: the
    # rescaling magnitude was a function of log_w and broke the I-divergence
    # balance with L_RB + L_score. With unit-1/N normalization for all three
    # terms, L_CE + L_RB + L_score is the unbiased Monte Carlo estimate of
    # the appendix's ELBO integrand, and (theta, log_w) gradients track it
    # within MC noise (verified by the autograd-vs-FD parity test).
    ce_den = jnp.maximum(jnp.sum(mask_f), 1.0)
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
    # rb_theta_stop_gradient (default False): stop-gradient P_theta in L_RB so
    # the term contributes ZERO gradient to theta (the classifier then trains on
    # the weighted-CE term only). P_theta depends only on theta (not log_w), so
    # this leaves L_RB's log_w gradient — via lam_hat and one_minus_S — exactly
    # unchanged, and the loss VALUE is unchanged (stop_gradient is identity in
    # the forward pass). When False the path is bit-identical to before.
    p_theta_rb_for_rb = (
        jax.lax.stop_gradient(p_theta_rb) if rb_theta_stop_gradient else p_theta_rb
    )
    rb_inner = jnp.sum(
        lam_hat_per_site * (p_theta_rb_for_rb - one_hot_x0), axis=-1
    )  # (B, *site)
    rb_per_site = mask_f * one_minus_S * rb_inner
    rb_num = jnp.sum(rb_per_site)
    # Normalizer = count of active sites, w-independent (no stop_gradient needed).
    rb_count = jnp.maximum(jnp.sum(mask_f), 1.0)
    loss_rb = rb_num / rb_count

    # ---- L_score (corrected appendix Correction 1) ----
    # The (1 - S_a) occupancy weight is w-differentiable; the bracketed
    # residual depends only on theta via P_theta, so stop_gradient blocks
    # its theta-gradient. This way:
    #   - d_theta L_score = 0 (the score head is still fit by CE only).
    #   - d_w     L_score != 0 via the (1 - S_a) factor — this is the
    #     restoring force that pins log_w at the consistent hazard.
    # See plan v3 Correction 1 for the derivation; ||s_theta - grad log
    # p^ac(.|X_0)||^2 at eta=1 reduces to (alpha^2 / sigma^4) * ||abar -
    # E(X_0)||^2, where abar = sum_a P_theta(a|Y) E(a) is the classifier's
    # expected anchor embedding. Numerically verified against autograd to
    # ~4e-7.
    #
    # IMPORTANT (Correction 2 footgun): anchor_table MUST be the NORMALIZED
    # table (model.anchor_table()), the same space x0_anchor lives in. The
    # raw params["anchors"]["table"] is un-normalized; using it here would
    # silently give a wrong L_score (no crash). See the caller in
    # sjd_base.py.
    anchor_table_E = jnp.asarray(anchor_table, dtype=jnp.float32)  # (K, d)
    # abar.shape == (B, *site, d). Use p_theta_rb (which is softmax(logits_rb));
    # under rb_share_sample=True this is the same forward pass as L_CE so cost
    # is one matmul beyond what L_CE/L_RB already paid for.
    abar = jnp.einsum("...k,kd->...d", p_theta_rb, anchor_table_E)
    # g_t^2 = beta_VP(t) for the VP-SDE dX = -1/2 beta X dt + sqrt(beta) dW
    # (Correction 1). NOT hazard.lam(t) — beta and hazard schedules share
    # the symbol but are different schedules.
    g2_t = jnp.asarray(beta(t_img), dtype=jnp.float32)  # (B,)
    alpha_t_score, sigma_t_score = alpha_sigma(beta, t_img)  # (B,), (B,)
    alpha_b = jnp.asarray(alpha_t_score, dtype=jnp.float32).reshape(
        (B,) + site_axes
    )
    sigma_b = jnp.asarray(sigma_t_score, dtype=jnp.float32).reshape(
        (B,) + site_axes
    )
    g2_b = g2_t.reshape((B,) + site_axes)
    ratio = (alpha_b ** 2) / (sigma_b ** 4 + eps)  # (B, *site)
    resid2 = jnp.sum(
        (abar - jnp.asarray(x0_anchor, dtype=jnp.float32)) ** 2, axis=-1
    )  # (B, *site)
    score_integrand = (g2_b / 2.0) * ratio * resid2  # (B, *site)
    # Default (score_w_stop_gradient=False): stop_gradient ONLY the
    # theta-dependent residual. The (1-S_a) occupancy factor remains
    # w-differentiable, so L_score contributes a nonzero gradient to log_w
    # (the v3 corrected behavior; theta still fit by CE only).
    #
    # When score_w_stop_gradient=True: stop_gradient the FULL product, so
    # L_score contributes ZERO gradient to both theta AND log_w. The loss
    # value is identical; only the gradient path differs. Used by the CE
    # condition of the Sudoku World-1-vs-World-2 test to isolate the
    # discrete-channel-preferred schedule from the score term's opposition.
    if score_w_stop_gradient:
        ell_score_per_site = jax.lax.stop_gradient(
            one_minus_S * score_integrand
        )
    else:
        ell_score_per_site = one_minus_S * jax.lax.stop_gradient(
            score_integrand
        )
    score_num = jnp.sum(mask_f * ell_score_per_site)
    loss_score = score_num / rb_count  # SAME 1/N normalizer as L_CE and L_RB

    # ---- Prior Omega(w) = lambda_Omega * sum_a (w(a) - 1)^2 ----
    # Finite under the log_w_clip applied in TokenAnchors.log_w_float()
    # (plan v3 H1+H2): without the clamp, exp(large log_w) overflows and
    # prior_strength * inf produces NaN even when prior_strength=0.
    w_table = jnp.exp(log_w_table)
    loss_prior = jnp.asarray(prior_strength, dtype=jnp.float32) * jnp.sum(
        (w_table - 1.0) ** 2
    )

    loss = (
        loss_ce
        + jnp.asarray(rb_weight, dtype=jnp.float32) * loss_rb
        + loss_score
        + loss_prior
    )

    metrics: Metrics = {
        "loss": loss,
        "loss/ce": loss_ce,
        "loss/rb": loss_rb,
        "loss/score": loss_score,
        "loss/prior": loss_prior,
        "loss/ce_nll_bits": loss_ce / jnp.log(2.0),
        # Un-normalized weighted sums + counts. Useful for gradient
        # correctness tests: with all three terms sharing a 1/N normalizer,
        # autograd vs FD parity on the total loss is the implementation
        # gate (plan v3 test 3a).
        "loss/ce_num": ce_num,
        "loss/ce_den": ce_den,
        "loss/rb_num": rb_num,
        "loss/score_num": score_num,
        "t/mean": jnp.mean(t_img),
        "t/std": jnp.std(t_img),
        "log_w/mean": jnp.mean(log_w_table),
        "log_w/std": jnp.std(log_w_table),
        "log_w/range": jnp.max(log_w_table) - jnp.min(log_w_table),
        # log_w_clip engagement monitor: w/max sitting at exp(clip_hi) means
        # the optimizer wants to push log_w further than the clamp allows;
        # if persistent post-Layer-B, re-probe log_w_lr_multiplier or
        # widen the clamp.
        "w/max": jnp.max(w_table),
        "w/min": jnp.min(w_table),
    }
    return loss, metrics
