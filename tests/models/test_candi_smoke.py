from __future__ import annotations

import jax
import jax.numpy as jnp

from sticky.models.candi import sampling as candi_sampling
from sticky.models.candi.candi_model import CANDI


def _make_model(*, representation: str = "embed", sampler: str = "hybrid_cache") -> CANDI:
    return CANDI(
        data_shape=(8, 8, 3),
        vocab_size=256,
        cont_time=True,
        timesteps=8,
        representation=representation,
        experimental=True,
        alpha_schedule_type="linear",
        schedule_eps=0.0,
        use_percentile_scheduling=True,
        min_percentile=0.01,
        max_percentile=0.45,
        sigma_min=0.2,
        sigma_max=4.0,
        ode_step_scale=1.0,
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
        categorical_sampling_policy="exact",
        guidance_scale=0.0,
    )


def _init_model(model: CANDI, x: jax.Array):
    rng = jax.random.PRNGKey(0)
    rng_params, rng_sample = jax.random.split(rng)
    variables = model.init({"params": rng_params, "sample": rng_sample}, x, train=False)
    return variables


def test_candi_smoke_end_to_end():
    model = _make_model()
    x = jnp.reshape(jnp.arange(2 * 8 * 8 * 3, dtype=jnp.int32), (2, 8, 8, 3)) % 256
    variables = _init_model(model, x)

    rng_loss, rng_prior, rng_step, rng_generate = jax.random.split(jax.random.PRNGKey(1), 4)

    stats = model.apply(
        variables,
        x,
        train=False,
        rngs={"sample": rng_loss},
    )
    assert jnp.isfinite(stats["loss"])
    assert jnp.isfinite(stats["loss_ce"])
    assert float(stats["loss"]) > 0.0
    assert float(stats["loss_ce"]) > 0.0

    prior = model.apply(
        variables,
        2,
        method=model.prior_sample,
        rngs={"sample": rng_prior},
    )
    assert prior["tokens"].shape == (2, 8, 8, 3)
    assert prior["clean_mask"].shape == (2, 8, 8, 3)
    assert prior["continuous"].shape == (2, 8, 8, 3, 32)
    assert prior["tokens"].dtype == jnp.int32
    assert prior["clean_mask"].dtype == bool
    assert jnp.isfinite(prior["continuous"]).all()

    reverse_step = model.apply(
        variables,
        rng_step,
        0,
        8,
        prior,
        method=model.sample_step,
    )
    assert reverse_step["continuous"].shape == prior["continuous"].shape
    assert jnp.isfinite(reverse_step["continuous"]).all()
    assert reverse_step["tokens"].dtype == jnp.int32

    decoded = model.apply(variables, reverse_step, method=model.decode)
    assert decoded.shape == (2, 8, 8, 3)
    assert decoded.dtype == jnp.int32
    assert int(decoded.min()) >= 0
    assert int(decoded.max()) <= 255

    generated = candi_sampling.simple_generate(
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


def test_candi_corruption_keeps_revealed_positions_exact():
    model = _make_model()
    x = jnp.reshape(jnp.arange(2 * 8 * 8 * 3, dtype=jnp.int32), (2, 8, 8, 3)) % 256
    variables = _init_model(model, x)

    corruption = model.apply(
        variables,
        x,
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        rng=jax.random.PRNGKey(7),
        method=model.corrupt_input,
    )

    clean_mask = corruption["clean_mask"]
    corrupted_mask = corruption["corrupted_mask"]
    diff = jnp.abs(corruption["hybrid_repr"] - corruption["clean_repr"])

    assert bool(jnp.any(clean_mask))
    assert bool(jnp.any(corrupted_mask))
    assert float(jnp.max(diff * clean_mask[..., None])) == 0.0
    assert float(jnp.max(diff * corrupted_mask[..., None])) > 0.0


def test_candi_reverse_step_preserves_clean_positions():
    model = _make_model()
    x = jnp.reshape(jnp.arange(2 * 8 * 8 * 3, dtype=jnp.int32), (2, 8, 8, 3)) % 256
    variables = _init_model(model, x)

    prior = model.apply(
        variables,
        2,
        method=model.prior_sample,
        rngs={"sample": jax.random.PRNGKey(11)},
    )
    clean_mask = prior["clean_mask"].at[:, :2, :2, :].set(True)
    exact_repr = model.apply(variables, prior["tokens"], method=model.clean_representation)
    state = {
        "tokens": prior["tokens"],
        "clean_mask": clean_mask,
        "continuous": jnp.where(clean_mask[..., None], exact_repr, prior["continuous"]),
    }

    next_state = model.apply(
        variables,
        jax.random.PRNGKey(12),
        0,
        8,
        state,
        method=model.sample_step,
    )

    assert jnp.array_equal(next_state["tokens"][:, :2, :2, :], state["tokens"][:, :2, :2, :])
    assert bool(jnp.all(next_state["clean_mask"][:, :2, :2, :]))
    assert jnp.isfinite(next_state["continuous"]).all()
