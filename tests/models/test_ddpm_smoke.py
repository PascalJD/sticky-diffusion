from __future__ import annotations

import jax
import jax.numpy as jnp

from sticky.models.ddpm.ddpm_model import DDPM
from sticky.models.ddpm import sampling as ddpm_sampling


def test_ddpm_smoke_end_to_end():
    model = DDPM(
        data_shape=(8, 8, 3),
        timesteps=8,
        beta_schedule="linear",
        beta_start=1.0e-4,
        beta_end=2.0e-2,
        prediction_type="eps",
        variance_type="fixed_small",
        clip_x0=True,
        feature_dim=32,
        ch_mult=(1, 1),
        dropout_rate=0.0,
        image_backbone="adm_unet2d",
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
        classes=-1,
    )

    rng = jax.random.PRNGKey(0)
    rng_params, rng_sample, rng_loss, rng_prior, rng_step, rng_generate = jax.random.split(
        rng, 6
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

    prior = model.apply(
        variables,
        2,
        method=model.prior_sample,
        rngs={"sample": rng_prior},
    )
    assert prior.shape == (2, 8, 8, 3)
    assert prior.dtype == jnp.float32

    reverse_step = model.apply(
        variables,
        rng_step,
        0,
        8,
        prior,
        method=model.sample_step,
    )
    assert reverse_step.shape == prior.shape
    assert jnp.isfinite(reverse_step).all()

    decoded = model.apply(
        variables,
        reverse_step,
        method=model.decode,
    )
    assert decoded.shape == prior.shape
    assert decoded.dtype == jnp.float32
    assert float(decoded.min()) >= 0.0
    assert float(decoded.max()) <= 255.0

    generated = ddpm_sampling.simple_generate(
        rng_generate,
        {"params": variables["params"], "ema_params": None},
        model=model,
        batch_size=2,
        timesteps=8,
        use_ema=False,
    )
    generated = jax.block_until_ready(generated)
    assert generated.shape == (2, 8, 8, 3)
    assert generated.dtype == jnp.float32
    assert float(generated.min()) >= 0.0
    assert float(generated.max()) <= 255.0
