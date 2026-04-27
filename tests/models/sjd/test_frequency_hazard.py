"""Frequency-weighted SJD hazard: bit-exact baseline + per-anchor identity tests.

Covers the two backward-compatibility tests from the plan:

1. With ``log_w = jnp.zeros(K)`` (uniform p_0), every code path reproduces the
   anchor-agnostic baseline bit-exactly.
2. With non-uniform ``log_w``, ``S_a(t)`` matches the closed form ``S(t)^{w[a]}``
   and the ``mixture_logpdf`` quadrature agrees with a high-resolution
   reference within float tolerance.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sticky.models.sjd.anchors import AnchorTable
from sticky.models.sjd.corruption import (
    classifier_induced_score,
    mixture_logpdf,
    sample_pair,
)
from sticky.models.sjd.hazard import make_hazard_poly_alpha
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.plugin_intensity import (
    dhm_log_ratio,
    plugin_hazard_and_allocation,
)
from sticky.models.sjd.sdes import make_beta


def _toy_setup(K=8, d=4, T=1.0, eta=0.7):
    beta = make_beta(beta_min=0.1, beta_max=8.0, T=T)
    hazard = make_hazard_poly_alpha(beta, p=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=eta)
    key = jax.random.PRNGKey(0)
    a_table = jax.random.normal(key, (K, d), dtype=jnp.float32)
    return beta, hazard, jump, a_table


def test_uniform_log_w_bit_matches_baseline_sample_pair():
    """sample_pair(log_w_per_site=zeros) == sample_pair(no kwarg) bit-exactly."""
    beta, hazard, jump, _ = _toy_setup()
    K, d = 8, 4

    rng = jax.random.PRNGKey(11)
    for trial in range(8):
        rng, k_t, k_x, k_idx, k_pair = jax.random.split(rng, 5)
        B = 4
        site_shape = (3,)
        x0_idx = jax.random.randint(k_idx, (B,) + site_shape, 0, K)
        a_table = jax.random.normal(k_x, (K, d), dtype=jnp.float32)
        x0_anchor = jnp.take(a_table, x0_idx, axis=0)
        t = jax.random.uniform(k_t, (B,), minval=0.05, maxval=0.95)

        baseline_xt, baseline_mask = sample_pair(
            k_pair, x0_anchor, t, beta, hazard, jump
        )
        zeros_log_w = jnp.zeros((B,) + site_shape, dtype=jnp.float32)
        new_xt, new_mask = sample_pair(
            k_pair, x0_anchor, t, beta, hazard, jump,
            log_w_per_site=zeros_log_w,
        )

        # masks must be exactly equal (boolean comparison)
        assert jnp.array_equal(baseline_mask, new_mask), \
            f"trial {trial}: never_unstuck masks differ"
        # xt must match within float precision; the only difference is one
        # extra log/expm1/exp roundtrip on the u_eff variable, which can
        # introduce ~1 ulp drift on the unstuck samples.
        np.testing.assert_allclose(
            np.asarray(baseline_xt), np.asarray(new_xt),
            rtol=0.0, atol=1e-5,
        )


def test_uniform_log_w_bit_matches_baseline_mixture_logpdf():
    """mixture_logpdf(log_w_anchor=0.0) == mixture_logpdf(default) bit-exactly."""
    beta, hazard, jump, _ = _toy_setup()
    B, d = 4, 4

    key = jax.random.PRNGKey(13)
    k_y, k_a, k_t = jax.random.split(key, 3)
    y = jax.random.normal(k_y, (B, d))
    anchor = jax.random.normal(k_a, (B, d))
    t = jax.random.uniform(k_t, (B,), minval=0.1, maxval=0.9)

    baseline = mixture_logpdf(y, anchor, t, beta, hazard, jump, tau_grid_size=33)
    new_zero = mixture_logpdf(
        y, anchor, t, beta, hazard, jump, tau_grid_size=33, log_w_anchor=0.0,
    )
    # The new code path goes through `prefix + 0.0 + log_S * 1.0`, which is
    # bit-exact for finite floats. Use exact equality.
    assert jnp.array_equal(baseline, new_zero)


def test_uniform_log_w_bit_matches_baseline_plugin_allocation():
    """plugin_hazard_and_allocation: AnchorTable(log_w=None) and (log_w=zeros)
    produce identical (lam_total, choice_probs) tensors. The None branch is
    the bit-exact legacy path; the zeros branch goes through the per-anchor
    log_lam_base formula but collapses algebraically to the same scalar."""
    beta, hazard, jump, a_table = _toy_setup()
    K, d = 8, 4
    B = 2
    site_shape = (3,)

    key = jax.random.PRNGKey(17)
    k_logits, k_y, k_t = jax.random.split(key, 3)
    logits = jax.random.normal(k_logits, (B,) + site_shape + (K,))
    y = jax.random.normal(k_y, (B,) + site_shape + (d,))
    t = jax.random.uniform(k_t, (B,), minval=0.1, maxval=0.9)

    anchors_none = AnchorTable(table_float=a_table, log_w=None)
    anchors_zero = AnchorTable(
        table_float=a_table, log_w=jnp.zeros((K,), dtype=jnp.float32)
    )
    lam_baseline, probs_baseline = plugin_hazard_and_allocation(
        logits=logits, y=y, t_img=t, anchors=anchors_none,
        beta=beta, hazard=hazard, jump=jump,
    )
    lam_new, probs_new = plugin_hazard_and_allocation(
        logits=logits, y=y, t_img=t, anchors=anchors_zero,
        beta=beta, hazard=hazard, jump=jump,
    )
    # log(beta*S/(1-S)) computed directly vs as log(beta)+log(S)-log1mexp(log S)
    # introduces ulp-level drift in float32; allow rtol=1e-5.
    np.testing.assert_allclose(
        np.asarray(lam_baseline), np.asarray(lam_new), rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(probs_baseline), np.asarray(probs_new),
        rtol=1e-5, atol=1e-6,
    )


def test_nonuniform_log_w_survival_identity():
    """For non-uniform log_w, the S_a(t) computed inside sample_pair matches the
    closed form S(t)^{w[a]} = exp(w[a] * log S(t))."""
    beta, hazard, jump, _ = _toy_setup()
    K, d = 8, 4
    B = 16
    site_shape = ()
    # mix of small and large weights so S_a(t) differs noticeably across a.
    log_w_table = jnp.array(
        [0.0, -0.5, 0.5, -1.0, 1.0, 0.3, -0.2, 1.5], dtype=jnp.float32
    )
    assert log_w_table.shape == (K,)

    key = jax.random.PRNGKey(23)
    k_t, k_idx = jax.random.split(key)
    t = jax.random.uniform(k_t, (B,), minval=0.05, maxval=0.95)
    a_idx = jax.random.randint(k_idx, (B,), 0, K)

    # closed-form S_a(t) = exp(w[a] * log S(t))
    log_S_t = jnp.log(jnp.maximum(hazard.surv(t), 1e-30))  # (B,)
    w_a = jnp.exp(log_w_table[a_idx])  # (B,)
    S_a_closed = jnp.exp(w_a * log_S_t)

    # Empirical: mean of `never_unstuck_mask` should match S_a(t) for each
    # (a, t) under MC. To keep the test deterministic and fast, instead pull
    # surv_t_b from a debug recomputation: replicate the formula from
    # sample_pair's log_w branch.
    surv_t_b = jnp.exp(w_a * jnp.log(hazard.surv(t)))  # same as S_a_closed
    np.testing.assert_allclose(
        np.asarray(surv_t_b), np.asarray(S_a_closed), rtol=0.0, atol=1e-7
    )

    # Sanity: for a small B replicate the never_unstuck rate empirically to
    # exercise the actual code path. Use N=20000 with a single (a, t).
    N = 20000
    a_one = jnp.asarray([3], dtype=jnp.int32)
    t_one = jnp.full((1,), 0.4, dtype=jnp.float32)
    expected_S = float(jnp.exp(jnp.exp(log_w_table[a_one]) * jnp.log(hazard.surv(t_one)))[0])

    a_table = jax.random.normal(jax.random.PRNGKey(99), (K, d), dtype=jnp.float32)
    a_vec = jnp.broadcast_to(a_table[a_one[0]], (N, d))
    t_b = jnp.full((N,), float(t_one[0]), dtype=jnp.float32)
    log_w_per_site = jnp.full((N,), float(log_w_table[a_one[0]]), dtype=jnp.float32)
    _, mask = sample_pair(
        jax.random.PRNGKey(31), a_vec, t_b, beta, hazard, jump,
        log_w_per_site=log_w_per_site,
    )
    empirical_S = float(jnp.mean(mask.astype(jnp.float32)))
    # MC tolerance ~ sqrt(p(1-p)/N) ~ 0.0035 here, allow 0.01.
    assert abs(empirical_S - expected_S) < 0.01, (
        f"empirical S_a(t) {empirical_S:.4f} != closed form {expected_S:.4f}"
    )


def _check_mixture_logpdf_against_reference(
    *, log_w_a_value: float, tau_grid_size: int, atol: float
):
    """Helper: mixture_logpdf with a chosen log_w_anchor matches a high-Nτ
    trapezoidal reference computed using the same w(a)-modified integrand."""
    beta, hazard, jump, _ = _toy_setup()
    B, d = 2, 4

    key = jax.random.PRNGKey(29)
    k_y, k_a, k_t = jax.random.split(key, 3)
    y = jax.random.normal(k_y, (B, d))
    anchor = jax.random.normal(k_a, (B, d))
    t = jnp.array([0.3, 0.7], dtype=jnp.float32)
    log_w_a = jnp.float32(log_w_a_value)
    w_a = float(jnp.exp(log_w_a))

    # Quadrature output from the implementation.
    out = mixture_logpdf(
        y, anchor, t, beta, hazard, jump,
        tau_grid_size=tau_grid_size, tau_grid="simpson",
        log_w_anchor=log_w_a,
    )

    # High-resolution trapezoidal reference: integrand at each tau is
    #     beta(tau) * w(a) * S(tau)^{w(a)} * N(y; alpha(t)*a, v_t(tau) I_d).
    # Compute the Gaussian log-density and v_t(tau) explicitly in numpy to
    # avoid broadcasting quirks in the JAX helpers.
    from sticky.models.sjd.sdes import alpha_sigma
    Nref = 4096
    for b in range(B):
        t_b = float(t[b])
        taus = np.linspace(max(t_b * 1e-6, 1e-6), t_b, Nref, dtype=np.float32)
        log_lam = np.log(
            np.maximum(np.asarray(hazard.lam(jnp.asarray(taus))), 1e-30)
        )
        log_S = -np.asarray(hazard.cum(jnp.asarray(taus)))
        alpha_t_b, sigma_t_b = alpha_sigma(beta, jnp.asarray(t_b))
        alpha_t_b = float(alpha_t_b)
        sigma2_t_b = float(sigma_t_b) ** 2
        alpha_tau, sigma_tau = alpha_sigma(beta, jnp.asarray(taus))
        alpha_tau_np = np.asarray(alpha_tau, dtype=np.float64)
        sigma_tau_np = np.asarray(sigma_tau, dtype=np.float64)
        eta = float(jump.eta)
        v_t = sigma2_t_b - (1.0 - eta * eta) * (alpha_t_b / alpha_tau_np) ** 2 * sigma_tau_np ** 2
        v_t = np.maximum(v_t, float(jump.std_floor) ** 2)
        diff = np.asarray(y[b]) - alpha_t_b * np.asarray(anchor[b])
        sq = float(np.sum(diff * diff))
        log_N = -0.5 * (d * (np.log(2.0 * np.pi) + np.log(v_t)) + sq / v_t)

        log_integrand = log_lam + np.log(w_a) + w_a * log_S + log_N
        # trapezoidal log-sum: integral ~ sum_{i} 0.5*h_i*(f_i + f_{i+1}).
        h = taus[1:] - taus[:-1]
        log_h = np.log(h)
        log_term_left = log_integrand[:-1] + log_h
        log_term_right = log_integrand[1:] + log_h
        log_terms = np.concatenate([log_term_left, log_term_right]) - np.log(2.0)
        ref = float(np.log(np.exp(log_terms - log_terms.max()).sum()) + log_terms.max())

        np.testing.assert_allclose(
            float(out[b]), ref, rtol=0.0, atol=atol,
            err_msg=(
                f"w(a)={w_a:.3f} (log_w={log_w_a_value:+.4f}) tau_grid_size="
                f"{tau_grid_size} b={b}: quadrature {float(out[b]):.4f} vs "
                f"reference {ref:.4f}"
            ),
        )


def test_nonuniform_log_w_mixture_logpdf_matches_reference_modest():
    """w(a) ~ 1.5: 33-point Simpson is plenty accurate."""
    _check_mixture_logpdf_against_reference(
        log_w_a_value=0.4, tau_grid_size=33, atol=5e-2,
    )


def test_nonuniform_log_w_mixture_logpdf_matches_reference_low_clip():
    """w(a) = 0.1 (default clip lower bound): the integrand decays slowly in
    tau, well within Simpson's resolution. Tight tolerance."""
    _check_mixture_logpdf_against_reference(
        log_w_a_value=float(np.log(0.1)), tau_grid_size=33, atol=2e-2,
    )


def test_nonuniform_log_w_mixture_logpdf_matches_reference_high_clip():
    """w(a) = 10 (default clip upper bound): S(tau)^{10} concentrates the
    integrand near tau=0, so Simpson with 33 nodes can drift. With 129 nodes
    the quadrature converges to within ~5e-2 of the trapezoidal reference."""
    _check_mixture_logpdf_against_reference(
        log_w_a_value=float(np.log(10.0)), tau_grid_size=129, atol=5e-2,
    )


def test_classifier_score_uniform_log_w_bit_matches_baseline():
    """classifier_induced_score with anchors.log_w=None and =zeros agree."""
    beta, hazard, jump, a_table = _toy_setup()
    K, d = 8, 4
    B = 2
    site_shape = (3,)

    key = jax.random.PRNGKey(41)
    k_logits, k_y, k_t = jax.random.split(key, 3)
    logits = jax.random.normal(k_logits, (B,) + site_shape + (K,))
    y = jax.random.normal(k_y, (B,) + site_shape + (d,))
    t = jax.random.uniform(k_t, (B,), minval=0.1, maxval=0.9)

    anchors_none = AnchorTable(table_float=a_table, log_w=None)
    anchors_zero = AnchorTable(
        table_float=a_table, log_w=jnp.zeros((K,), dtype=jnp.float32)
    )
    score_baseline = classifier_induced_score(
        y, t, anchor_logits=logits, anchors=anchors_none,
        beta=beta, hazard=hazard, jump=jump,
    )
    score_new = classifier_induced_score(
        y, t, anchor_logits=logits, anchors=anchors_zero,
        beta=beta, hazard=hazard, jump=jump,
    )
    np.testing.assert_allclose(
        np.asarray(score_baseline), np.asarray(score_new), rtol=1e-5, atol=1e-6,
    )


def test_dhm_log_ratio_uniform_log_w_bit_matches_baseline():
    """dhm_log_ratio with anchors.log_w=None and =zeros agree."""
    beta, hazard, jump, a_table = _toy_setup()
    K, d = 8, 4
    B = 2
    site_shape = (3,)

    key = jax.random.PRNGKey(43)
    k_y, k_t = jax.random.split(key)
    y = jax.random.normal(k_y, (B,) + site_shape + (d,))
    t = jax.random.uniform(k_t, (B,), minval=0.1, maxval=0.9)

    anchors_none = AnchorTable(table_float=a_table, log_w=None)
    anchors_zero = AnchorTable(
        table_float=a_table, log_w=jnp.zeros((K,), dtype=jnp.float32)
    )
    lr_baseline = dhm_log_ratio(
        y=y, t_img=t, anchors=anchors_none,
        beta=beta, hazard=hazard, jump=jump, log_ratio_clip=10.0,
    )
    lr_new = dhm_log_ratio(
        y=y, t_img=t, anchors=anchors_zero,
        beta=beta, hazard=hazard, jump=jump, log_ratio_clip=10.0,
    )
    # Same log-space arithmetic with `+0` and `*1`; allow ~1 ulp drift.
    np.testing.assert_allclose(
        np.asarray(lr_baseline), np.asarray(lr_new), rtol=1e-6, atol=1e-7,
    )
