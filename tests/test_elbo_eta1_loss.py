"""Tests for the eta=1 SJD ELBO training loss
(L_CE + L_RB + L_score + Omega).

Plan v3 brought the three-term ELBO online: L_score is now part of the loss
(its w-gradient is the restoring force at the consistent hazard), and all
three terms share a 1/N normalizer (the previous stop_gradient(sum(w_ce))
CE normalizer was biased in log_w). These tests cover the original
ce_num/rb_num FD-vs-autograd gates plus the new normalization + L_score
behaviour. The deeper Layer-B gates (autograd-vs-FD on the total loss,
analytic-posterior stationarity, theta-grad zero, discrete limit) live in
test_score_term_layer_b.py.
"""
import jax
import jax.numpy as jnp

from sticky.models.sjd.sdes import make_beta, alpha_sigma
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sjd_elbo_loss import elbo_eta1_loss


# Tiny dims so finite-difference gradient checks are tractable.
B, L_seq, K, d = 4, 1, 3, 2


def _make_schedules():
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    jump = VPMatchedGaussianJump(beta=beta)  # default eta=1 unless overridden
    return beta, hazard, jump


def _fake_apply_fn(params, x_in, t_img):
    """Logits that depend deterministically on (x_in[:, :, 0], t) only.

    Has no dependence on params or log_w, so the loss's only log_w-dependence
    is through the L_CE weight term (and L_RB's lambda_hat factor). This makes
    finite-difference gradient checks exact up to numerical noise.
    """
    x_first = x_in[..., 0]  # (B, L_seq)
    # Logits shape (B, L_seq, K) with entries x_first + t_img + (anchor-id offset)
    base = x_first[..., None] + t_img[:, None, None]
    offsets = jnp.arange(K, dtype=jnp.float32)
    return base + offsets[None, None, :], {}


def _make_x0(rng):
    """Stable per-batch x0 indices and embedding-like anchors."""
    rng_idx, rng_anc = jax.random.split(rng)
    x0_idx = jax.random.randint(rng_idx, shape=(B, L_seq), minval=0, maxval=K)
    # A fixed (K, d) anchor table; x0_anchor[b, s] = anchor_table[x0_idx[b, s]]
    anchor_table = jax.random.normal(rng_anc, shape=(K, d))
    x0_anchor = jnp.take(anchor_table, x0_idx, axis=0)
    return x0_idx, x0_anchor, anchor_table


def test_elbo_eta1_loss_value_matches_closed_form_at_w_eq_1():
    """At log_w = 0 and a fixed seed, the loss is finite and the three
    normalized loss-component metrics exist."""
    beta, hazard, jump = _make_schedules()
    rng = jax.random.PRNGKey(0)
    x0_idx, x0_anchor, anchor_table = _make_x0(rng)
    log_w = jnp.zeros((K,), dtype=jnp.float32)

    loss, metrics = elbo_eta1_loss(
        key=jax.random.PRNGKey(1),
        params={},
        apply_fn=_fake_apply_fn,
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta,
        hazard=hazard,
        jump=jump,
        T=1.0,
        anchor_log_w=log_w,
        anchor_table=anchor_table,
        prior_strength=0.0,
        rb_share_sample=True,  # one forward pass; logits reused
    )
    assert loss.ndim == 0
    assert bool(jnp.isfinite(loss)), f"loss not finite: {loss}"
    for key in ("loss/ce", "loss/rb", "loss/score", "loss/prior"):
        assert key in metrics
        assert bool(jnp.isfinite(metrics[key])), f"{key} not finite"


def test_grad_log_w_L_CE_num_matches_finite_difference():
    """Sanity: autograd grad of ce_num (the un-normalized weighted sum) wrt
    log_w matches FD. This is the L_CE component of the v3 implementation
    gate. The total-loss FD parity test lives in test_score_term_layer_b.py
    (test 3a).
    """
    beta, hazard, jump = _make_schedules()
    rng = jax.random.PRNGKey(2)
    x0_idx, x0_anchor, anchor_table = _make_x0(rng)
    log_w0 = jnp.array([-0.3, 0.1, 0.2], dtype=jnp.float32)  # asymmetric
    key_fixed = jax.random.PRNGKey(7)

    def ce_num_only(log_w):
        loss, metrics = elbo_eta1_loss(
            key=key_fixed,
            params={},
            apply_fn=_fake_apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=beta,
            hazard=hazard,
            jump=jump,
            T=1.0,
            anchor_log_w=log_w,
            anchor_table=anchor_table,
            prior_strength=0.0,
            rb_weight=0.0,  # isolate L_CE
        )
        return metrics["loss/ce_num"]

    grad_auto = jax.grad(ce_num_only)(log_w0)

    h = 1e-3
    grad_fd_list = []
    for a in range(K):
        e_a = jnp.zeros_like(log_w0).at[a].set(1.0)
        plus = ce_num_only(log_w0 + h * e_a)
        minus = ce_num_only(log_w0 - h * e_a)
        grad_fd_list.append((plus - minus) / (2.0 * h))
    grad_fd = jnp.stack(grad_fd_list)

    err = jnp.max(jnp.abs(grad_auto - grad_fd))
    assert err < 1e-3, (
        f"autograd vs FD ce_num gradient mismatch: max |delta| = {float(err)}\n"
        f"autograd = {grad_auto}\nFD      = {grad_fd}"
    )


def test_loss_ce_normalizer_matches_rb_normalizer():
    """Plan v3 B2: L_CE and L_RB share a 1/N normalizer (sum(mask_f), not
    stop_grad(sum(w_ce))). This makes L_CE + L_RB + L_score an unbiased MC
    estimator of the appendix's ELBO integrand.

    Check that ce_den equals the active-site count exactly (no
    stop_gradient detour), so FD on the total loss can match autograd.
    """
    beta, hazard, jump = _make_schedules()
    rng = jax.random.PRNGKey(4)
    x0_idx, x0_anchor, anchor_table = _make_x0(rng)
    log_w0 = jnp.array([-0.3, 0.1, 0.2], dtype=jnp.float32)

    _, metrics = elbo_eta1_loss(
        key=jax.random.PRNGKey(11),
        params={},
        apply_fn=_fake_apply_fn,
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta,
        hazard=hazard,
        jump=jump,
        T=1.0,
        anchor_log_w=log_w0,
        anchor_table=anchor_table,
        prior_strength=0.0,
        rb_weight=0.0,
    )
    # No given_mask provided, so every site is active; ce_den == B * L_seq.
    expected_den = float(B * L_seq)
    got_den = float(metrics["loss/ce_den"])
    assert abs(got_den - expected_den) < 1e-6, (
        f"ce_den should equal active-site count {expected_den}, got {got_den}. "
        "The stop_gradient(sum(w_ce)) normalizer is the previous biased "
        "behavior; it has been replaced by the unified 1/N path."
    )


def test_L_RB_zero_at_w_eq_1():
    """At w == 1, lam_hat(a) is anchor-agnostic, so the inner sum
    Sum_a lam_hat * (P - 1{a=x0}) = lam_hat * (1 - 1) = 0. Loss_rb == 0
    exactly at log_w == 0, regardless of the model."""
    beta, hazard, jump = _make_schedules()
    rng = jax.random.PRNGKey(5)
    x0_idx, x0_anchor, anchor_table = _make_x0(rng)
    log_w = jnp.zeros((K,), dtype=jnp.float32)

    _, metrics = elbo_eta1_loss(
        key=jax.random.PRNGKey(13),
        params={},
        apply_fn=_fake_apply_fn,
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta,
        hazard=hazard,
        jump=jump,
        T=1.0,
        anchor_log_w=log_w,
        anchor_table=anchor_table,
        prior_strength=0.0,
        rb_weight=1.0,
    )
    assert float(jnp.abs(metrics["loss/rb"])) < 1e-5, (
        f"loss_rb should be exactly 0 at w==1 (anchor-symmetric lam_hat), got {metrics['loss/rb']}"
    )


def test_grad_log_w_L_RB_num_matches_finite_difference():
    """PRIMARY correctness gate for L_RB: autograd grad of rb_num matches FD.

    Verifies that the w-gradient flows analytically through *both* lam_hat
    (the reparametrized weight Sum_a) and (1 - S_a) (the reparametrized
    per-site filter that replaces Algorithm 1's Bernoulli `1{u >= S}`).
    """
    beta, hazard, jump = _make_schedules()
    rng = jax.random.PRNGKey(6)
    x0_idx, x0_anchor, anchor_table = _make_x0(rng)
    log_w0 = jnp.array([-0.3, 0.1, 0.2], dtype=jnp.float32)
    key_fixed = jax.random.PRNGKey(17)

    def rb_num_only(log_w):
        _, metrics = elbo_eta1_loss(
            key=key_fixed,
            params={},
            apply_fn=_fake_apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=beta,
            hazard=hazard,
            jump=jump,
            T=1.0,
            anchor_log_w=log_w,
            anchor_table=anchor_table,
            prior_strength=0.0,
            rb_weight=1.0,
            rb_share_sample=False,  # independent draw — same gradient guarantee
        )
        return metrics["loss/rb_num"]

    grad_auto = jax.grad(rb_num_only)(log_w0)

    h = 1e-3
    grad_fd_list = []
    for a in range(K):
        e_a = jnp.zeros_like(log_w0).at[a].set(1.0)
        plus = rb_num_only(log_w0 + h * e_a)
        minus = rb_num_only(log_w0 - h * e_a)
        grad_fd_list.append((plus - minus) / (2.0 * h))
    grad_fd = jnp.stack(grad_fd_list)

    err = jnp.max(jnp.abs(grad_auto - grad_fd))
    assert err < 1e-3, (
        f"autograd vs FD rb_num gradient mismatch: max |delta| = {float(err)}\n"
        f"autograd = {grad_auto}\nFD      = {grad_fd}"
    )


def test_prior_omega_contributes_to_loss():
    """Omega(w) = strength * sum_a (w(a) - 1)^2. At w == 1 the prior is 0;
    at w != 1 it adds a positive penalty that is computed correctly."""
    beta, hazard, jump = _make_schedules()
    rng = jax.random.PRNGKey(8)
    x0_idx, x0_anchor, anchor_table = _make_x0(rng)

    # log_w such that w = exp([-0.5, 0, 0.5]) = [0.6065, 1.0, 1.6487]
    log_w = jnp.array([-0.5, 0.0, 0.5], dtype=jnp.float32)
    w = jnp.exp(log_w)
    expected_prior = float(jnp.sum((w - 1.0) ** 2))  # ~ 0.575
    strength = 0.7

    _, metrics = elbo_eta1_loss(
        key=jax.random.PRNGKey(19),
        params={},
        apply_fn=_fake_apply_fn,
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta,
        hazard=hazard,
        jump=jump,
        T=1.0,
        anchor_log_w=log_w,
        anchor_table=anchor_table,
        prior_strength=strength,
        rb_weight=0.0,
    )
    got = float(metrics["loss/prior"])
    assert abs(got - strength * expected_prior) < 1e-5, (
        f"prior mismatch: got {got}, want {strength * expected_prior}"
    )


def test_grad_log_w_L_CE_nonzero_and_finite():
    """Sanity: at non-trivial inputs, grad is finite and not all zero (would
    indicate a silent stop_gradient broke the w-path)."""
    beta, hazard, jump = _make_schedules()
    rng = jax.random.PRNGKey(3)
    x0_idx, x0_anchor, anchor_table = _make_x0(rng)
    log_w0 = jnp.array([-0.3, 0.0, 0.2], dtype=jnp.float32)

    def L_CE_only(log_w):
        loss, metrics = elbo_eta1_loss(
            key=jax.random.PRNGKey(9),
            params={},
            apply_fn=_fake_apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=beta,
            hazard=hazard,
            jump=jump,
            T=1.0,
            anchor_log_w=log_w,
            anchor_table=anchor_table,
            prior_strength=0.0,
            rb_weight=0.0,
        )
        return metrics["loss/ce"]

    g = jax.grad(L_CE_only)(log_w0)
    assert bool(jnp.all(jnp.isfinite(g))), f"grad not finite: {g}"
    assert float(jnp.linalg.norm(g)) > 1e-4, f"grad ~ 0 (suggests stop_grad bug): {g}"
