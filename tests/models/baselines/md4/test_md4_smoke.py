from __future__ import annotations

import sys
from types import ModuleType

import jax
import jax.numpy as jnp

import sticky.models.common.masked_discrete_core as masked_core


def _install_md4_optional_dependency_stubs() -> None:
    def _ensure_module(name: str, *, attr_name: str | None = None, attr_value=None):
        if name in sys.modules:
            return sys.modules[name]
        module = ModuleType(name)
        if attr_name is not None:
            setattr(module, attr_name, attr_value)
        sys.modules[name] = module
        return module

    _ensure_module("distrax", attr_name="Distribution", attr_value=object)
    matplotlib_mod = _ensure_module("matplotlib")
    pyplot_mod = _ensure_module("matplotlib.pyplot")
    if not hasattr(matplotlib_mod, "pyplot"):
        matplotlib_mod.pyplot = pyplot_mod
    orbax_mod = _ensure_module("orbax")
    checkpoint_mod = _ensure_module("orbax.checkpoint")
    if not hasattr(checkpoint_mod, "CheckpointManager"):
        checkpoint_mod.CheckpointManager = object
    if not hasattr(orbax_mod, "checkpoint"):
        orbax_mod.checkpoint = checkpoint_mod
    _ensure_module("seaborn")


_install_md4_optional_dependency_stubs()

from sticky.models.baselines.md4 import sampling as md4_sampling
from sticky.models.baselines.md4.md4_model import MD4


def _make_model() -> MD4:
    return MD4(
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
        model_sharding=False,
    )


def test_md4_smoke_end_to_end():
    model = _make_model()

    rng = jax.random.PRNGKey(0)
    rng_params, rng_sample, rng_loss, rng_forward, rng_step, rng_generate = jax.random.split(
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
    assert jnp.isfinite(stats["loss_diff"])
    assert jnp.isfinite(stats["loss_prior"])
    assert jnp.isfinite(stats["loss_recon"])
    assert stats["loss"].shape == ()
    assert stats["loss_diff"].shape == ()
    assert stats["loss_prior"].shape == ()
    assert stats["loss_recon"].shape == ()
    assert float(stats["loss"]) > 0.0
    assert float(stats["loss_diff"]) > 0.0
    assert float(stats["loss_recon"]) > 0.0

    prior = model.apply(
        variables,
        2,
        method=model.prior_sample,
    )
    assert prior.shape == (2, 8, 8, 3)
    assert prior.dtype == jnp.int32
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

    forward = model.apply(
        variables,
        x,
        jnp.asarray([0.37, 0.81], dtype=jnp.float32),
        method=model.forward_sample,
        rngs={"sample": rng_forward},
    )
    assert forward.shape == x.shape
    assert forward.dtype == jnp.int32
    assert bool(
        jnp.all((forward == x) | (forward == jnp.asarray(model.mask_token_id, dtype=jnp.int32)))
    )

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

    generated = md4_sampling.simple_generate(
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
