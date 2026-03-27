from __future__ import annotations

import jax
import jax.numpy as jnp

import sticky.models.common.masked_discrete_core as masked_core
from sticky.models.baselines.mdlm import sampling as mdlm_sampling
from sticky.models.baselines.mdlm.mdlm_model import MDLM, _selected_log_prob_sums


def _make_model() -> MDLM:
    return MDLM(
        data_shape=(8, 8, 3),
        cont_time=True,
        timesteps=8,
        feature_dim=32,
        num_heads=4,
        antithetic_time_sampling=True,
        n_layers=32,
        n_dit_layers=0,
        dit_num_heads=4,
        dit_hidden_size=128,
        ch_mult=(1, 1),
        vocab_size=256,
        noise_schedule_type="linear",
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
        sampling_grid="cosine",
        categorical_sampling_policy="legacy_low",
        cache_predictions=False,
        model_sharding=False,
    )


def test_mdlm_smoke_end_to_end():
    model = _make_model()

    rng = jax.random.PRNGKey(0)
    rng_params, rng_sample, rng_loss, rng_step, rng_generate = jax.random.split(rng, 5)

    x = jnp.reshape(jnp.arange(2 * 8 * 8 * 3, dtype=jnp.int32), (2, 8, 8, 3)) % 256
    variables = model.init({"params": rng_params, "sample": rng_sample}, x, train=False)

    stats = model.apply(
        variables,
        x,
        train=False,
        rngs={"sample": rng_loss},
    )

    assert jnp.isfinite(stats["loss"])
    assert jnp.isfinite(stats["loss_diff"])
    assert jnp.isfinite(stats["loss_recon"])
    assert stats["loss"].shape == ()
    assert stats["loss_diff"].shape == ()
    assert stats["loss_recon"].shape == ()
    assert float(stats["loss"]) > 0.0
    assert float(stats["loss_diff"]) > 0.0
    assert float(stats["loss_recon"]) == 0.0

    prior = model.apply(
        variables,
        2,
        method=model.prior_sample,
    )
    assert prior.shape == (2, 8, 8, 3)
    assert prior.dtype == jnp.int32
    assert bool(jnp.all(prior == model.mask_token_id))

    reverse_step = model.apply(
        variables,
        rng_step,
        0,
        8,
        prior,
        method=model.sample_step,
    )
    assert reverse_step.shape == prior.shape
    assert reverse_step.dtype == jnp.int32
    assert jnp.isfinite(reverse_step.astype(jnp.float32)).all()

    decoded = model.apply(
        variables,
        reverse_step,
        method=model.decode,
    )
    assert decoded.shape == prior.shape
    assert decoded.dtype == jnp.int32
    assert int(decoded.min()) >= 0
    assert int(decoded.max()) <= 255
    assert not bool(jnp.any(decoded == model.mask_token_id))

    generated = mdlm_sampling.simple_generate(
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


def test_mdlm_masked_discrete_helpers_preserve_core_semantics():
    model = _make_model()

    rng = jax.random.PRNGKey(11)
    rng_params, rng_sample, rng_forward = jax.random.split(rng, 3)
    x = jnp.reshape(jnp.arange(2 * 8 * 8 * 3, dtype=jnp.int32), (2, 8, 8, 3)) % 256
    variables = model.init({"params": rng_params, "sample": rng_sample}, x, train=False)

    prior = model.apply(
        variables,
        2,
        method=model.prior_sample,
    )
    assert bool(jnp.all(prior == model.mask_token_id))

    s, t = model.apply(
        variables,
        0,
        8,
        method=model.get_sampling_grid,
    )
    expected_s, expected_t = masked_core.make_sampling_time_pair(
        0,
        8,
        sampling_grid=model.sampling_grid,
    )
    assert jnp.allclose(s, expected_s)
    assert jnp.allclose(t, expected_t)

    zt = model.apply(
        variables,
        x,
        jnp.asarray([0.37, 0.81], dtype=jnp.float32),
        method=model.forward_sample,
        rngs={"sample": rng_forward},
    )
    assert zt.shape == x.shape
    assert zt.dtype == jnp.int32
    assert bool(
        jnp.all((zt == x) | (zt == jnp.asarray(model.mask_token_id, dtype=jnp.int32)))
    )


def test_mdlm_diffusion_loss_matches_md4_sign_convention():
    model = _make_model()

    rng = jax.random.PRNGKey(17)
    rng_params, rng_sample, rng_noise = jax.random.split(rng, 3)
    x = (jnp.reshape(jnp.arange(8 * 8 * 3, dtype=jnp.int32), (1, 8, 8, 3)) * 7) % 256
    variables = model.init({"params": rng_params, "sample": rng_sample}, x, train=False)

    t = jnp.asarray([0.37], dtype=jnp.float32)
    zt = model.apply(
        variables,
        x,
        t,
        method=model.forward_sample,
        rngs={"sample": rng_noise},
    )
    logits, _ = model.apply(
        variables,
        zt,
        t,
        method=model.predict_x,
        train=False,
    )
    log_probs, _ = model.apply(
        variables,
        zt,
        t,
        method=model.predict_clean_log_probs,
        train=False,
    )
    weight = masked_core.masked_dgamma_times_alpha(
        t,
        schedule_fn_type=model.noise_schedule_type,
    )
    mask = masked_core.masked_positions(zt, mask_token_id=model.mask_token_id)

    expected = weight * _selected_log_prob_sums(log_probs, x)
    md4_style = weight * masked_core.masked_logprob_sums(logits, x, mask=mask)
    actual = model.apply(
        variables,
        t,
        x,
        method=model.diffusion_loss,
        train=False,
        rngs={"sample": rng_noise},
    )

    assert jnp.allclose(actual, expected)
    assert jnp.allclose(actual, md4_style)
    assert float(actual[0]) > 0.0
