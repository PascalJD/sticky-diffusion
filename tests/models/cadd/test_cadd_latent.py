from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from sticky.models.cadd.cadd_model import CADD


@pytest.mark.parametrize("latent_type", ["gaussian", "flow_matching"])
def test_masked_latent_marginal_uses_selected_process(latent_type: str):
    model = CADD(
        data_shape=(2,),
        vocab_size=8,
        feature_dim=4,
        num_heads=1,
        n_layers=1,
        ch_mult=(1,),
        dropout_rate=0.0,
        use_attn_dropout=False,
        sequence_backbone="transformer",
        image_backbone="auto",
        discrete_schedule_type="linear",
        continuous_schedule_type="linear",
        continuous_latent_type=latent_type,
        schedule_eps=0.0,
    )

    x = jnp.asarray([[0, 1]], dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "sample": jax.random.PRNGKey(1)},
        x,
        cond=None,
        train=False,
    )

    z0 = model.apply(
        variables,
        x,
        method=lambda mdl, ids: mdl.token_embed(ids),
    )
    t = jnp.asarray([0.75], dtype=jnp.float32)
    sample_rng = jax.random.PRNGKey(7)
    latent = model.apply(
        variables,
        rng=sample_rng,
        z0=z0,
        t=t,
        method=model._sample_masked_continuous_latent,
    )

    gbar = 1.0 - t
    noise = jax.random.normal(sample_rng, z0.shape)
    if latent_type == "gaussian":
        expected = jnp.sqrt(gbar)[:, None, None] * z0 + jnp.sqrt(1.0 - gbar)[:, None, None] * noise
    else:
        expected = gbar[:, None, None] * z0 + (1.0 - gbar)[:, None, None] * noise

    np.testing.assert_allclose(
        np.asarray(latent),
        np.asarray(expected),
        atol=1e-6,
        rtol=1e-6,
    )


def test_gaussian_reverse_reaches_z0_without_noise_at_final_step():
    model = CADD(
        data_shape=(2,),
        vocab_size=8,
        feature_dim=4,
        num_heads=1,
        n_layers=1,
        ch_mult=(1,),
        dropout_rate=0.0,
        use_attn_dropout=False,
        sequence_backbone="transformer",
        image_backbone="auto",
        discrete_schedule_type="linear",
        continuous_schedule_type="linear",
        continuous_latent_type="gaussian",
        schedule_eps=0.0,
    )

    x = jnp.asarray([[0, 1]], dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(2), "sample": jax.random.PRNGKey(3)},
        x,
        cond=None,
        train=False,
    )

    z_t = jnp.arange(8, dtype=jnp.float32).reshape(1, 2, 4)
    z0_hat = -jnp.ones_like(z_t)
    z_s = model.apply(
        variables,
        rng=jax.random.PRNGKey(5),
        z_t=z_t,
        z0_hat=z0_hat,
        s=jnp.asarray(0.0, dtype=jnp.float32),
        t=jnp.asarray(0.5, dtype=jnp.float32),
        method=model._reverse_masked_continuous_latent,
    )

    np.testing.assert_allclose(
        np.asarray(z_s),
        np.asarray(z0_hat),
        atol=1e-6,
        rtol=1e-6,
    )
