"""hazard_deriv loss weighting: makes log_w enter the loss differentiably.

Four invariants:

1. Finiteness across hazard schedules.
2. Gradient on a learned ``log_w`` param is non-zero everywhere (the test the
   previous learned-w(a) PR could not satisfy).
3. Reference value: pre-normalization w_t matches the closed form
   β(t)·exp(−H)/(1−exp(−H)) at log_w=0 with linear hazard.
4. None-fallback: weighting works when ``anchor_log_w`` is ``None``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sticky.models.sjd.anchors import AnchorTableConfig, TokenAnchors
from sticky.models.sjd.hazard import (
    make_hazard_linear_time,
    make_hazard_poly_alpha,
)
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.losses import _hazard_deriv_weights, ce_allocation_loss
from sticky.models.sjd.sdes import make_beta


def _toy(*, K=8, d=4, B=16, seq_len=4, hazard_kind="linear", anchor_log_w=None):
    table = jax.random.normal(
        jax.random.PRNGKey(0), (K, d), dtype=jnp.float32
    )
    x0_idx = jax.random.randint(
        jax.random.PRNGKey(1), (B, seq_len), 0, K
    ).astype(jnp.int32)
    x0_anchor = table[x0_idx]
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    if hazard_kind == "linear":
        hazard = make_hazard_linear_time(beta, kappa=1.5)
    elif hazard_kind == "poly":
        hazard = make_hazard_poly_alpha(beta, p=1.0)
    else:
        raise ValueError(hazard_kind)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.5, std_floor=1e-3)

    # Trivial parameterized "classifier": logits = x_t @ W. Lets jax.grad
    # produce a non-trivial gradient on classifier params.
    W = jax.random.normal(
        jax.random.PRNGKey(2), (d, K), dtype=jnp.float32
    )
    params = {"W": W}

    def apply_fn(p, xt, t_img):
        del t_img
        logits = jnp.einsum("...d,dv->...v", xt, p["W"])
        return logits, {}

    return dict(
        apply_fn=apply_fn,
        params=params,
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta,
        hazard=hazard,
        jump=jump,
        T=1.0,
        anchor_log_w=anchor_log_w,
    )


@pytest.mark.parametrize("hazard_kind", ["linear", "poly"])
def test_hazard_deriv_finiteness(hazard_kind):
    """Loss and grad are finite for both hazard schedules with non-uniform log_w."""
    K = 8
    log_w = jax.random.normal(jax.random.PRNGKey(7), (K,), dtype=jnp.float32)
    kw = _toy(K=K, hazard_kind=hazard_kind, anchor_log_w=log_w)

    def loss_only(p):
        loss, _ = ce_allocation_loss(
            key=jax.random.PRNGKey(11),
            loss_weighting="hazard_deriv",
            **{**kw, "params": p},
        )
        return loss

    loss_val = loss_only(kw["params"])
    assert jnp.isfinite(loss_val), f"loss is non-finite: {loss_val}"
    grad = jax.grad(loss_only)(kw["params"])
    assert jnp.all(jnp.isfinite(grad["W"])), "non-finite entries in classifier grad"


def test_hazard_deriv_grad_on_log_w_is_nonzero():
    """The whole point of this PR: learned log_w receives a non-zero gradient."""
    K, d, B, seq_len = 8, 4, 64, 6

    # Cover every anchor at least once so no log_w[i] is structurally
    # disconnected from the loss.
    x0_idx = (jnp.arange(B * seq_len) % K).reshape(B, seq_len).astype(jnp.int32)

    log_w_init = jax.random.normal(
        jax.random.PRNGKey(3), (K,), dtype=jnp.float32
    )
    config = AnchorTableConfig(
        family="normal", vocab_size=K, anchor_dim=d, init_std=1.0, seed=0
    )
    anchors_module = TokenAnchors(
        config=config, learnable=True, learnable_log_w=True, log_w_init=log_w_init,
    )
    anchor_params = anchors_module.init(
        jax.random.PRNGKey(4), jnp.zeros((1,), dtype=jnp.int32)
    )["params"]
    table = anchor_params["table"]
    x0_anchor = jnp.take(table, x0_idx, axis=0)

    W = jax.random.normal(jax.random.PRNGKey(5), (d, K), dtype=jnp.float32)
    params = {"anchors": anchor_params, "classifier": {"W": W}}

    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=1.5)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.5, std_floor=1e-3)

    def apply_fn(p, xt, t_img):
        del t_img
        return jnp.einsum("...d,dv->...v", xt, p["classifier"]["W"]), {}

    def loss_fn(p):
        log_w_eff = anchors_module.apply(
            {"params": p["anchors"]}, method=anchors_module.log_w_float
        )
        loss, _ = ce_allocation_loss(
            key=jax.random.PRNGKey(6),
            params=p,
            apply_fn=apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=beta,
            hazard=hazard,
            jump=jump,
            anchor_log_w=log_w_eff,
            loss_weighting="hazard_deriv",
            T=1.0,
        )
        return loss

    grads = jax.grad(loss_fn)(params)
    log_w_grad = grads["anchors"]["log_w"]
    grad_l2 = float(jnp.linalg.norm(log_w_grad))
    assert grad_l2 > 1e-3, (
        f"L2 norm of log_w grad is {grad_l2}, expected > 1e-3 — log_w is "
        "still not receiving signal."
    )
    assert not bool(jnp.any(log_w_grad == 0.0)), (
        f"some log_w grad coords are zero: {np.asarray(log_w_grad)}"
    )


def test_hazard_deriv_reference_value_log_w_zero_linear():
    """At log_w=0 with linear hazard, pre-norm w_t matches β(t)·exp(−kt/T)/(1−exp(−kt/T))."""
    K = 4
    T = 1.0
    kappa = 2.5
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=T)
    hazard = make_hazard_linear_time(beta, kappa=kappa)
    log_w = jnp.zeros((K,), dtype=jnp.float32)

    # B=1, single site (seq_len=1), so target_like has shape (1, 1) and
    # _expand_like is a no-op once we broadcast t_img of shape (1,).
    t_img = jnp.array([0.3], dtype=jnp.float32)
    x0_idx = jnp.array([[0]], dtype=jnp.int32)
    target_like = jnp.zeros((1, 1), dtype=jnp.float32)

    impl = _hazard_deriv_weights(
        t_img=t_img, beta=beta, hazard=hazard,
        anchor_log_w=log_w, x0_idx=x0_idx, target_like=target_like,
    )

    # Closed form, computed in numpy via the elementary expression.
    t_np = np.asarray(t_img, dtype=np.float64)
    beta_np = np.asarray(beta(t_img), dtype=np.float64)
    H_np = (kappa / T) * t_np
    closed = beta_np * np.exp(-H_np) / (1.0 - np.exp(-H_np))

    np.testing.assert_allclose(
        np.asarray(impl).reshape(-1),
        closed.reshape(-1),
        rtol=1e-6,
        atol=1e-6,
    )


def test_hazard_deriv_none_fallback_runs():
    """anchor_log_w=None branch produces a finite scalar loss without error."""
    kw = _toy(anchor_log_w=None)
    loss, metrics = ce_allocation_loss(
        key=jax.random.PRNGKey(13),
        loss_weighting="hazard_deriv",
        **kw,
    )
    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(metrics["t/mean"]))
