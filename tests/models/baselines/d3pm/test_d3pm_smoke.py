from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from sticky.models.baselines.d3pm import sampling as d3pm_sampling
from sticky.models.baselines.d3pm.d3pm_model import D3PM


def _make_model(
    transition_type: str,
    *,
    sampling_grid: str = "uniform",
) -> D3PM:
    schedule = {
        "absorb": "absorbing_linear",
        "uniform": "cosine",
        "gaussian": "linear",
    }[transition_type]
    return D3PM(
        data_shape=(8, 8, 3),
        timesteps=6,
        transition_type=transition_type,
        transition_beta_schedule=schedule,
        beta_start=1.0e-4,
        beta_end=2.0e-2,
        auxiliary_loss_weight=1.0e-3,
        absorbing_state=128,
        feature_dim=32,
        num_heads=4,
        antithetic_time_sampling=True,
        n_layers=32,
        n_dit_layers=0,
        dit_num_heads=4,
        dit_hidden_size=128,
        ch_mult=(1, 1),
        vocab_size=256,
        dropout_rate=0.0,
        use_attn_dropout=False,
        mlp_type="swiglu",
        depth_scaled_init=True,
        cond_type="adaln_zero",
        outside_embed=False,
        sequence_backbone="auto",
        image_backbone="adm_unet5d",
        adm_num_res_blocks=1,
        adm_attention_resolutions=(),
        adm_num_heads=1,
        adm_num_head_channels=-1,
        adm_num_heads_upsample=-1,
        adm_conv_resample=True,
        adm_use_scale_shift_norm=True,
        adm_resblock_updown=False,
        adm_use_conv_skip=False,
        adm_use_new_attention_order=False,
        time_features="t",
        classes=-1,
        sampler="ancestral",
        sampling_grid=sampling_grid,
        categorical_sampling_policy="exact",
        model_sharding=False,
    )


def _init_variables(model: D3PM, rng: jax.Array):
    rng_params, rng_sample = jax.random.split(rng)
    x = jnp.reshape(jnp.arange(2 * 8 * 8 * 3, dtype=jnp.int32), (2, 8, 8, 3)) % 256
    variables = model.init({"params": rng_params, "sample": rng_sample}, x, train=False)
    return x, variables


@pytest.mark.parametrize("transition_type", ["absorb", "uniform", "gaussian"])
def test_d3pm_posterior_probabilities_normalize(transition_type: str):
    model = _make_model(transition_type)
    rng = jax.random.PRNGKey(transition_type.__hash__() & 0xFFFF)
    x0, variables = _init_variables(model, rng)
    t_index = jnp.asarray([2, 5], dtype=jnp.int32)

    rng_xt = jax.random.PRNGKey(123)
    xt = model.apply(
        variables,
        rng_xt,
        x0,
        t_index,
        method=model.q_sample,
    )

    posterior = model.apply(
        variables,
        x0,
        xt,
        t_index,
        method=model.q_posterior_probs,
    )
    assert posterior.shape == x0.shape + (256,)
    assert jnp.isfinite(posterior).all()
    assert jnp.all(posterior >= 0.0)
    assert jnp.allclose(jnp.sum(posterior, axis=-1), 1.0, atol=1.0e-5)

    rng_logits = jax.random.PRNGKey(456)
    clean_logits = jax.random.normal(rng_logits, x0.shape + (256,))
    model_posterior = model.apply(
        variables,
        clean_logits,
        xt,
        t_index,
        method=model.model_posterior_probs,
    )
    assert model_posterior.shape == x0.shape + (256,)
    assert jnp.isfinite(model_posterior).all()
    assert jnp.all(model_posterior >= 0.0)
    assert jnp.allclose(jnp.sum(model_posterior, axis=-1), 1.0, atol=1.0e-5)


@pytest.mark.parametrize("transition_type", ["absorb", "uniform", "gaussian"])
def test_d3pm_one_step_reverse_sampling_smoke(transition_type: str):
    model = _make_model(transition_type)
    rng = jax.random.PRNGKey((transition_type.__hash__() >> 4) & 0xFFFF)
    x, variables = _init_variables(model, rng)

    rng_prior, rng_step = jax.random.split(jax.random.PRNGKey(7))
    if transition_type == "absorb":
        state = model.apply(variables, 2, method=model.prior_sample)
    else:
        state = model.apply(
            variables,
            2,
            method=model.prior_sample,
            rngs={"sample": rng_prior},
        )
    reverse_step = model.apply(
        variables,
        rng_step,
        0,
        model.timesteps,
        state,
        method=model.sample_step,
    )

    assert reverse_step.shape == x.shape
    assert reverse_step.dtype == jnp.int32
    assert int(reverse_step.min()) >= 0
    assert int(reverse_step.max()) <= 255


def test_d3pm_gaussian_smoke_end_to_end():
    model = _make_model("gaussian")

    rng = jax.random.PRNGKey(0)
    rng_init, rng_loss, rng_prior, rng_step, rng_generate = jax.random.split(rng, 5)
    x, variables = _init_variables(model, rng_init)

    stats = model.apply(
        variables,
        x,
        train=False,
        rngs={"sample": rng_loss},
    )

    assert jnp.isfinite(stats["loss"])
    assert jnp.isfinite(stats["loss_vb"])
    assert jnp.isfinite(stats["loss_aux"])
    assert jnp.isfinite(stats["loss_prior"])
    assert stats["loss"].shape == ()
    assert stats["loss_vb"].shape == ()
    assert stats["loss_aux"].shape == ()
    assert stats["loss_prior"].shape == ()

    prior = model.apply(
        variables,
        2,
        method=model.prior_sample,
        rngs={"sample": rng_prior},
    )
    assert prior.shape == x.shape
    assert prior.dtype == jnp.int32

    reverse_step = model.apply(
        variables,
        rng_step,
        0,
        model.timesteps,
        prior,
        method=model.sample_step,
    )
    assert reverse_step.shape == prior.shape
    assert reverse_step.dtype == jnp.int32

    decoded = model.apply(
        variables,
        reverse_step,
        method=model.decode,
    )
    assert decoded.shape == prior.shape
    assert decoded.dtype == jnp.int32
    assert int(decoded.min()) >= 0
    assert int(decoded.max()) <= 255

    generated = d3pm_sampling.simple_generate(
        rng_generate,
        {"params": variables["params"], "ema_params": None},
        model=model,
        batch_size=2,
        timesteps=model.timesteps,
        use_ema=False,
    )
    generated = jax.block_until_ready(generated)
    assert generated.shape == x.shape
    assert generated.dtype == jnp.int32
    assert int(generated.min()) >= 0
    assert int(generated.max()) <= 255


def test_d3pm_rejects_non_uniform_sampling_grid():
    model = _make_model("gaussian", sampling_grid="cosine")
    x = jnp.reshape(jnp.arange(2 * 8 * 8 * 3, dtype=jnp.int32), (2, 8, 8, 3)) % 256

    with pytest.raises(ValueError, match="sampling_grid='uniform'"):
        model.init(
            {"params": jax.random.PRNGKey(0), "sample": jax.random.PRNGKey(1)},
            x,
            train=False,
        )


def test_d3pm_terminal_prior_behavior_is_documented():
    x_terminal = jnp.stack(
        [
            jnp.zeros((8, 8, 3), dtype=jnp.int32),
            jnp.full((8, 8, 3), 127, dtype=jnp.int32),
        ],
        axis=0,
    )

    absorb_model = _make_model("absorb")
    _, absorb_variables = _init_variables(absorb_model, jax.random.PRNGKey(21))
    absorb_kl = absorb_model.apply(
        absorb_variables,
        x_terminal,
        method=absorb_model.terminal_prior_kl,
    )
    assert jnp.allclose(absorb_kl, 0.0)
    absorb_loss_prior = absorb_model.apply(
        absorb_variables,
        x_terminal,
        method=absorb_model.latent_loss,
    )
    assert float(absorb_loss_prior) == 0.0

    uniform_model = _make_model("uniform")
    _, uniform_variables = _init_variables(uniform_model, jax.random.PRNGKey(22))
    uniform_kl = uniform_model.apply(
        uniform_variables,
        x_terminal,
        method=uniform_model.terminal_prior_kl,
    )
    assert jnp.allclose(uniform_kl, 0.0)
    uniform_loss_prior = uniform_model.apply(
        uniform_variables,
        x_terminal,
        method=uniform_model.latent_loss,
    )
    assert float(uniform_loss_prior) == 0.0

    gaussian_model = _make_model("gaussian")
    _, gaussian_variables = _init_variables(gaussian_model, jax.random.PRNGKey(23))
    gaussian_kl = gaussian_model.apply(
        gaussian_variables,
        x_terminal,
        method=gaussian_model.terminal_prior_kl,
    )
    assert jnp.isfinite(gaussian_kl).all()
    assert bool(jnp.all(gaussian_kl > 0.0))
    assert not bool(jnp.allclose(gaussian_kl[0], gaussian_kl[1]))
    gaussian_loss_prior = gaussian_model.apply(
        gaussian_variables,
        x_terminal,
        method=gaussian_model.latent_loss,
    )
    assert float(gaussian_loss_prior) > 0.0
