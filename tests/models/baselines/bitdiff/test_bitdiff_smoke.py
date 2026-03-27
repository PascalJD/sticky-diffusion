from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import sticky.models.common.continuous_discrete_core as continuous_core
from sticky.models.baselines.bitdiff import sampling as bitdiff_sampling
from sticky.models.baselines.bitdiff.bitdiff_model import BitDiffusion


def _make_model(
    *,
    self_conditioning: bool = True,
    sampler: str = "ddim",
    stochasticity: float = 0.0,
) -> BitDiffusion:
    return BitDiffusion(
        data_shape=(8, 8, 3),
        cont_time=True,
        timesteps=8,
        num_bits=8,
        encoding="uint8",
        predict_target="x0",
        loss_type="mse",
        self_conditioning=self_conditioning,
        self_conditioning_rate=0.5,
        analog_bit_scale=1.0,
        clip_x0=True,
        signal_schedule_type="linear",
        schedule_eps=0.0,
        feature_dim=32,
        num_heads=4,
        n_layers=32,
        n_dit_layers=0,
        dit_num_heads=4,
        dit_hidden_size=128,
        ch_mult=(1, 1),
        dropout_rate=0.0,
        use_attn_dropout=False,
        mlp_type="swiglu",
        depth_scaled_init=False,
        cond_type="adaln",
        model_sharding=False,
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
        sampler=sampler,
        sampling_grid="uniform",
        time_difference=0.0,
        stochasticity=stochasticity,
    )


def test_bitdiff_uint8_roundtrip():
    x = jnp.asarray([[0, 7, 31, 127, 128, 255]], dtype=jnp.int32)
    analog = continuous_core.uint8_to_analog_bits(x)
    restored = continuous_core.analog_bits_to_uint8(analog)
    assert jnp.array_equal(restored, x)


def test_bitdiff_smoke_end_to_end():
    model = _make_model(self_conditioning=True)

    rng = jax.random.PRNGKey(0)
    rng_params, rng_sample, rng_loss, rng_prior, rng_step, rng_generate = jax.random.split(
        rng,
        6,
    )

    x = jnp.reshape(jnp.arange(2 * 8 * 8 * 3, dtype=jnp.int32), (2, 8, 8, 3)) % 256
    variables = model.init({"params": rng_params, "sample": rng_sample}, x, train=False)

    stats = model.apply(
        variables,
        x,
        train=False,
        rngs={"sample": rng_loss},
    )
    assert jnp.isfinite(stats["loss"])
    assert jnp.isfinite(stats["loss_mse"])
    assert stats["loss"].shape == ()
    assert stats["loss_mse"].shape == ()
    assert float(stats["loss"]) > 0.0

    prior = model.apply(
        variables,
        2,
        method=model.prior_sample,
        rngs={"sample": rng_prior},
    )
    assert prior.shape == (2, 8, 8, 3, 8)
    assert prior.dtype == jnp.float32
    assert jnp.isfinite(prior).all()

    reverse_step = model.apply(
        variables,
        rng_step,
        0,
        8,
        prior,
        method=model.sample_step,
    )
    assert reverse_step.shape == prior.shape
    assert reverse_step.dtype == jnp.float32
    assert jnp.isfinite(reverse_step).all()

    decoded = model.apply(
        variables,
        reverse_step,
        method=model.decode,
    )
    assert decoded.shape == (2, 8, 8, 3)
    assert decoded.dtype == jnp.int32
    assert int(decoded.min()) >= 0
    assert int(decoded.max()) <= 255

    generated = bitdiff_sampling.simple_generate(
        rng_generate,
        {"params": variables["params"], "ema_params": None},
        model=model,
        batch_size=2,
        timesteps=8,
        use_ema=False,
    )
    generated = jax.block_until_ready(generated)

    assert generated.shape == (2, 8, 8, 3)
    assert generated.dtype == jnp.int32
    assert int(generated.min()) >= 0
    assert int(generated.max()) <= 255


def test_bitdiff_sampler_contract_runtime_behavior():
    x = jnp.reshape(jnp.arange(2 * 8 * 8 * 3, dtype=jnp.int32), (2, 8, 8, 3)) % 256
    latent = jnp.reshape(
        jnp.linspace(-1.0, 1.0, 2 * 8 * 8 * 3 * 8, dtype=jnp.float32),
        (2, 8, 8, 3, 8),
    )
    rng_params = jax.random.PRNGKey(21)
    rng_sample = jax.random.PRNGKey(22)
    step_rng = jax.random.PRNGKey(23)

    model_ddim_det = _make_model(self_conditioning=False, sampler="ddim", stochasticity=0.0)
    variables_ddim_det = model_ddim_det.init({"params": rng_params, "sample": rng_sample}, x, train=False)
    out_ddim_det = model_ddim_det.apply(
        variables_ddim_det,
        step_rng,
        0,
        8,
        latent,
        method=model_ddim_det.sample_step,
    )

    model_ddim_noisy = _make_model(self_conditioning=False, sampler="ddim", stochasticity=0.35)
    variables_ddim_noisy = model_ddim_noisy.init({"params": rng_params, "sample": rng_sample}, x, train=False)
    out_ddim_noisy = model_ddim_noisy.apply(
        variables_ddim_noisy,
        step_rng,
        0,
        8,
        latent,
        method=model_ddim_noisy.sample_step,
    )

    model_ddim_unit = _make_model(self_conditioning=False, sampler="ddim", stochasticity=1.0)
    variables_ddim_unit = model_ddim_unit.init({"params": rng_params, "sample": rng_sample}, x, train=False)
    out_ddim_unit = model_ddim_unit.apply(
        variables_ddim_unit,
        step_rng,
        0,
        8,
        latent,
        method=model_ddim_unit.sample_step,
    )

    model_ddpm = _make_model(self_conditioning=False, sampler="ddpm", stochasticity=1.0)
    variables_ddpm = model_ddpm.init({"params": rng_params, "sample": rng_sample}, x, train=False)
    out_ddpm = model_ddpm.apply(
        variables_ddpm,
        step_rng,
        0,
        8,
        latent,
        method=model_ddpm.sample_step,
    )

    assert model_ddim_det._sampling_eta() == 0.0
    assert model_ddim_noisy._sampling_eta() == 0.35
    assert model_ddpm._sampling_eta() == 1.0
    assert not jnp.allclose(out_ddim_det, out_ddim_noisy)
    assert jnp.allclose(out_ddpm, out_ddim_unit)


def test_bitdiff_ddpm_alias_rejects_non_unit_stochasticity():
    model = _make_model(self_conditioning=False, sampler="ddpm", stochasticity=0.4)
    x = jnp.reshape(jnp.arange(2 * 8 * 8 * 3, dtype=jnp.int32), (2, 8, 8, 3)) % 256

    with pytest.raises(ValueError, match="fixed eta=1.0"):
        model.init(
            {"params": jax.random.PRNGKey(30), "sample": jax.random.PRNGKey(31)},
            x,
            train=False,
        )
