from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from sticky.models.sjd.convolution import mixture_component_logpdf
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sdes import alpha_sigma, make_beta
from sticky.models.sjd.corruption import (
    classifier_induced_score,
    mixture_logpdf,
)


def _setup(*, eta=0.5, kappa=2.0, L=4, d=3, B=3, S=5, seed=0):
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=kappa)
    jump = VPMatchedGaussianJump(beta=beta, eta=eta, std_floor=1e-3)
    key = jax.random.PRNGKey(seed)
    k_a, k_y, k_logits = jax.random.split(key, 3)
    a_table = jax.random.normal(k_a, (L, d), dtype=jnp.float32)
    anchors = SimpleNamespace(table_float=a_table)
    y = jax.random.normal(k_y, (B, S, d), dtype=jnp.float32)
    anchor_logits = jax.random.normal(k_logits, (B, S, L), dtype=jnp.float32)
    t = jnp.linspace(0.3, 0.7, B, dtype=jnp.float32)
    return beta, hazard, jump, anchors, y, anchor_logits, t


def _dense_mixture_logpdf(
    y, anchor, t, beta, hazard, jump, tau_grid_size
):
    """Reference: stack mixture_component_logpdf over the same midpoint
    tau-grid as `_build_quadrature(uniform)` and reduce via logsumexp."""
    B = t.shape[0]
    eta = float(jump.eta)
    std_floor = float(jump.std_floor)
    i_arr = jnp.arange(tau_grid_size, dtype=jnp.float32) + 0.5
    nodes = (i_arr[:, None] / float(tau_grid_size)) * t[None, :]  # (tau_grid_size, B)
    log_h = jnp.log(t / float(tau_grid_size))[None, :]
    lam_nodes = jnp.asarray(hazard.lam(nodes), dtype=jnp.float32)
    log_lam = jnp.log(jnp.maximum(lam_nodes, 1e-30))
    log_S = -jnp.asarray(hazard.cum(nodes), dtype=jnp.float32)
    log_p = jax.vmap(
        lambda tau_b: mixture_component_logpdf(
            y=y, anchor=anchor, t=t, tau=tau_b,
            beta=beta, eta=eta, std_floor=std_floor,
        ),
        in_axes=0,
    )(nodes)  # (tau_grid_size, B, *site_shape)
    site_pad = (1,) * (log_p.ndim - 2)
    log_w_b = (log_h + log_lam + log_S).reshape(log_lam.shape + site_pad)
    return logsumexp(log_w_b + log_p, axis=0)


def _dense_sjd_score(
    y, t, anchor_logits, anchors, beta, hazard, jump, tau_grid_size
):
    """Reference: dense materialization of the SJD-branch score."""
    a_table = jnp.asarray(anchors.table_float, dtype=jnp.float32)
    B, S, d = y.shape[0], y.shape[1], y.shape[2]
    L = a_table.shape[0]
    eta = float(jump.eta)
    std_floor = float(jump.std_floor)
    probs = jax.nn.softmax(anchor_logits, axis=-1)
    alpha_t, sigma_t = alpha_sigma(beta, t)
    alpha_t = alpha_t.astype(jnp.float32)
    sigma_t = sigma_t.astype(jnp.float32)

    i_arr = jnp.arange(tau_grid_size, dtype=jnp.float32) + 0.5
    nodes = (i_arr[:, None] / float(tau_grid_size)) * t[None, :]
    log_h = jnp.log(t / float(tau_grid_size))[None, :]
    lam_nodes = jnp.asarray(hazard.lam(nodes), dtype=jnp.float32)
    log_lam = jnp.log(jnp.maximum(lam_nodes, 1e-30))
    log_S = -jnp.asarray(hazard.cum(nodes), dtype=jnp.float32)

    alpha_tau, sigma_tau = alpha_sigma(beta, nodes)
    alpha_tau = alpha_tau.astype(jnp.float32)
    sigma_tau = sigma_tau.astype(jnp.float32)
    deficit = (
        (1.0 - eta * eta)
        * jnp.square(alpha_t[None, :] / alpha_tau)
        * jnp.square(sigma_tau)
    )
    v = jnp.square(sigma_t)[None, :] - deficit  # (tau_grid_size, B)
    v = jnp.maximum(v, std_floor * std_floor)

    dot = jnp.einsum("bsd,ld->bsl", y, a_table)
    y_norm2 = jnp.sum(y * y, axis=-1, keepdims=True)
    a_norm2 = jnp.sum(a_table * a_table, axis=-1)[None, None, :]
    alpha_t_3 = alpha_t[:, None, None]
    dist2 = y_norm2 - 2.0 * alpha_t_3 * dot + (alpha_t_3 * alpha_t_3) * a_norm2

    log_2pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float32))
    v_b = v[:, :, None, None]
    log_N = -0.5 * (d * (log_2pi + jnp.log(v_b)) + dist2[None, :, :, :] / v_b)
    log_w_b = (log_h + log_lam + log_S)[:, :, None, None]
    log_integrand = log_w_b + log_N
    w_tau = jax.nn.softmax(log_integrand, axis=0)
    inv_v = 1.0 / v_b
    e = jnp.sum(w_tau * inv_v, axis=0)

    coeff = probs * e
    coeff_total = jnp.sum(coeff, axis=-1, keepdims=True)
    weighted_anchor_sum = jnp.einsum("bsl,ld->bsd", coeff, a_table)
    return -(coeff_total * y - alpha_t_3 * weighted_anchor_sum)


def test_streaming_matches_dense_mixture_logpdf():
    """The lax.scan online-LSE form must agree with a dense
    `logsumexp(stack)` reference within float32 noise."""
    beta, hazard, jump, anchors, y, _logits, t = _setup(eta=0.6, kappa=2.0)
    a = jnp.broadcast_to(anchors.table_float[0], y.shape)
    streaming = mixture_logpdf(y, a, t, beta, hazard, jump, tau_grid_size=32)
    dense = _dense_mixture_logpdf(y, a, t, beta, hazard, jump, tau_grid_size=32)
    np.testing.assert_allclose(
        np.asarray(streaming), np.asarray(dense), atol=1e-4, rtol=0.0
    )


def test_streaming_classifier_score_matches_dense():
    """The lax.scan running-(m, l, n) form must agree with the dense
    softmax-and-weighted-sum reference within float32 noise."""
    beta, hazard, jump, anchors, y, anchor_logits, t = _setup(
        eta=0.5, kappa=2.0
    )
    streaming = classifier_induced_score(
        y, t, anchor_logits=anchor_logits, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump, tau_grid_size=32,
    )
    dense = _dense_sjd_score(
        y, t, anchor_logits, anchors, beta, hazard, jump, tau_grid_size=32,
    )
    np.testing.assert_allclose(
        np.asarray(streaming), np.asarray(dense), atol=1e-4, rtol=0.0
    )


def test_streaming_memory_scaling():
    """Smoke test: large tau_grid_size runs without OOM and the integrand converges
    as tau_grid_size grows. With B=2, S=8, L=4 the streaming peak memory is
    O(B*S*L), independent of tau_grid_size."""
    beta, hazard, jump, anchors, y, anchor_logits, t = _setup(
        eta=0.5, kappa=2.0, B=2, S=8, L=4, d=3
    )
    a = jnp.broadcast_to(anchors.table_float[0], y.shape)

    results = {}
    for tau_grid_size in (16, 64, 256):
        v = mixture_logpdf(y, a, t, beta, hazard, jump, tau_grid_size=tau_grid_size)
        assert bool(jnp.all(jnp.isfinite(v))), f"non-finite at tau_grid_size={tau_grid_size}"
        results[tau_grid_size] = np.asarray(v)

    # Convergence: 64 -> 256 should change the result by less than 1e-3.
    np.testing.assert_allclose(results[256], results[64], atol=1e-3, rtol=0.0)
