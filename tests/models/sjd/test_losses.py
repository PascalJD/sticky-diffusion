from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sticky.models.sjd.losses import ce_allocation_loss
from sticky.models.sjd.sdes import make_beta


def _fixture(vocab_size: int = 4, seq_len: int = 6, B: int = 8, d: int = 3):
    from sticky.models.sjd.hazard import make_hazard_linear_time
    from sticky.models.sjd.jump import VPMatchedGaussianJump

    table = jax.random.normal(
        jax.random.PRNGKey(0), (vocab_size, d), dtype=jnp.float32
    )
    x0_idx = jax.random.randint(
        jax.random.PRNGKey(1), (B, seq_len), 0, vocab_size
    ).astype(jnp.int32)
    x0_anchor = table[x0_idx]

    def apply_fn(params, xt, t_img):
        del params, xt, t_img
        return jnp.zeros((B, seq_len, vocab_size), dtype=jnp.float32), {}

    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    return dict(
        apply_fn=apply_fn,
        params={},
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta,
        hazard=make_hazard_linear_time(beta, kappa=1.5),
        jump=VPMatchedGaussianJump(beta=beta, eta=0.5, std_floor=1e-3),
        T=1.0,
    )


def test_uniform_defaults_regression():
    kwargs = _fixture()
    key = jax.random.PRNGKey(42)
    loss_default, metrics_default = ce_allocation_loss(key=key, **kwargs)
    loss_explicit, metrics_explicit = ce_allocation_loss(
        key=key,
        time_sampling="uniform",
        loss_weighting="uniform",
        **kwargs,
    )
    np.testing.assert_array_equal(
        np.asarray(loss_default), np.asarray(loss_explicit)
    )
    for k in metrics_default:
        np.testing.assert_array_equal(
            np.asarray(metrics_default[k]), np.asarray(metrics_explicit[k])
        )


def test_antithetic_pairs_times():
    B = 8
    T = 1.0
    kwargs = _fixture(B=B)
    key = jax.random.PRNGKey(7)

    # ce_allocation_loss splits key 2-way (key_t, key_vp).
    key_t, _ = jax.random.split(key, 2)
    half_B = B // 2
    t_half = jax.random.uniform(key_t, shape=(half_B,), minval=0.0, maxval=T)
    expected_t = jnp.concatenate([t_half, T - t_half], axis=0)

    _, metrics = ce_allocation_loss(
        key=key, time_sampling="antithetic", **kwargs
    )

    np.testing.assert_allclose(
        float(metrics["t/mean"]), float(jnp.mean(expected_t)), rtol=1e-5
    )
    np.testing.assert_allclose(
        float(metrics["t/std"]), float(jnp.std(expected_t)), rtol=1e-5
    )


def test_antithetic_handles_odd_batch():
    kwargs = _fixture(B=31)
    loss, metrics = ce_allocation_loss(
        key=jax.random.PRNGKey(13), time_sampling="antithetic", **kwargs
    )
    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(metrics["t/mean"]))


def test_alpha_deriv_weights_positive_finite():
    kwargs = _fixture(B=16)
    loss, _ = ce_allocation_loss(
        key=jax.random.PRNGKey(3),
        loss_weighting="alpha_deriv",
        **kwargs,
    )
    assert bool(jnp.isfinite(loss).all())


def test_combined_features_scalar_finite():
    kwargs = _fixture(B=8)
    loss, _ = ce_allocation_loss(
        key=jax.random.PRNGKey(11),
        time_sampling="antithetic",
        loss_weighting="alpha_deriv",
        **kwargs,
    )
    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    assert bool(jnp.isfinite(loss))


def test_t_metrics_present():
    _, metrics = ce_allocation_loss(key=jax.random.PRNGKey(5), **_fixture())
    assert "t/mean" in metrics
    assert "t/std" in metrics
    assert "CE/loss_weight_mean" not in metrics
    assert "CE/loss_weight_std" not in metrics
