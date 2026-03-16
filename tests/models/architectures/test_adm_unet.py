from __future__ import annotations

import math

import numpy as np
import pytest

import flax.linen as nn
import jax
import jax.numpy as jnp

import sticky.models.architectures.discrete as discrete_mod
from sticky.models.architectures.networks.adm_unet import (
    GroupNorm32,
    QKVAttention,
    QKVAttentionLegacy,
    openai_timestep_embedding,
)


def _manual_qkv_attention_legacy(qkv: jnp.ndarray, n_heads: int) -> jnp.ndarray:
    bs, width, length = qkv.shape
    ch = width // (3 * n_heads)
    qkv = qkv.reshape(bs * n_heads, ch * 3, length)
    q, k, v = jnp.split(qkv, 3, axis=1)
    scale = 1.0 / math.sqrt(math.sqrt(ch))
    weight = jnp.einsum("bct,bcs->bts", q * scale, k * scale)
    weight = jax.nn.softmax(weight.astype(jnp.float32), axis=-1).astype(weight.dtype)
    a = jnp.einsum("bts,bcs->bct", weight, v)
    return a.reshape(bs, -1, length)


def _manual_qkv_attention_new(qkv: jnp.ndarray, n_heads: int) -> jnp.ndarray:
    bs, width, length = qkv.shape
    ch = width // (3 * n_heads)
    q, k, v = jnp.split(qkv, 3, axis=1)
    q = q.reshape(bs * n_heads, ch, length)
    k = k.reshape(bs * n_heads, ch, length)
    v = v.reshape(bs * n_heads, ch, length)
    scale = 1.0 / math.sqrt(math.sqrt(ch))
    weight = jnp.einsum("bct,bcs->bts", q * scale, k * scale)
    weight = jax.nn.softmax(weight.astype(jnp.float32), axis=-1).astype(weight.dtype)
    a = jnp.einsum("bts,bcs->bct", weight, v)
    return a.reshape(bs, -1, length)


def test_openai_timestep_embedding_matches_reference_formula():
    timesteps = jnp.asarray([0.0, 1.25], dtype=jnp.float32)
    emb = openai_timestep_embedding(timesteps, 5)

    half = 5 // 2
    freqs = jnp.exp(
        -math.log(10_000)
        * jnp.arange(start=0, stop=half, dtype=jnp.float32)
        / half
    )
    args = timesteps[:, None] * freqs[None]
    expected = jnp.concatenate(
        [jnp.cos(args), jnp.sin(args), jnp.zeros((timesteps.shape[0], 1), dtype=jnp.float32)],
        axis=-1,
    )

    np.testing.assert_allclose(np.asarray(emb), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_group_norm32_uses_fixed_float32_norm_and_casts_back():
    class NormCaller(nn.Module):
        @nn.compact
        def __call__(self, x):
            return GroupNorm32(num_channels=32, name="gn")(x)

    x16 = jnp.arange(2 * 4 * 4 * 32, dtype=jnp.float16).reshape(2, 4, 4, 32) / 97.0
    module = NormCaller()
    variables = module.init(jax.random.PRNGKey(0), x16)

    y16 = module.apply(variables, x16)
    y32 = module.apply(variables, x16.astype(jnp.float32)).astype(x16.dtype)

    assert y16.dtype == x16.dtype
    np.testing.assert_allclose(np.asarray(y16), np.asarray(y32), atol=5e-3, rtol=5e-3)


def test_group_norm32_requires_channels_divisible_by_32():
    class NormCaller(nn.Module):
        @nn.compact
        def __call__(self, x):
            return GroupNorm32(num_channels=24)(x)

    with pytest.raises(ValueError, match="divisible by 32"):
        NormCaller().init(jax.random.PRNGKey(0), jnp.ones((1, 2, 2, 24), dtype=jnp.float32))


def test_qkv_attention_legacy_and_new_match_openai_orders():
    qkv = jnp.arange(1, 1 + (1 * 12 * 4), dtype=jnp.float32).reshape(1, 12, 4) / 13.0

    legacy = QKVAttentionLegacy(num_heads=2)
    legacy_vars = legacy.init(jax.random.PRNGKey(0), qkv)
    legacy_out = legacy.apply(legacy_vars, qkv)

    new = QKVAttention(num_heads=2)
    new_vars = new.init(jax.random.PRNGKey(1), qkv)
    new_out = new.apply(new_vars, qkv)

    legacy_expected = _manual_qkv_attention_legacy(qkv, n_heads=2)
    new_expected = _manual_qkv_attention_new(qkv, n_heads=2)

    np.testing.assert_allclose(
        np.asarray(legacy_out),
        np.asarray(legacy_expected),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(new_out),
        np.asarray(new_expected),
        atol=1e-6,
        rtol=1e-6,
    )
    assert not np.allclose(np.asarray(legacy_out), np.asarray(new_out))


def test_adm_image_path_accepts_raw_timesteps_without_cond_embedding(monkeypatch):
    def _boom(self, t, cond=None):
        raise AssertionError("CondEmbedding should not be used on the ADM image path.")

    monkeypatch.setattr(discrete_mod.CondEmbedding, "__call__", _boom)

    model = discrete_mod.DiscreteClassifier(
        feature_dim=32,
        num_heads=1,
        vocab_size=8,
        image_backbone="adm_unet5d",
        ch_mult=(1,),
        adm_num_res_blocks=1,
        adm_attention_resolutions=(),
        dropout_rate=0.0,
        use_attn_dropout=False,
        adm_use_new_attention_order=False,
    )

    x = jnp.zeros((1, 4, 4, 1), dtype=jnp.int32)
    cond = jnp.ones((1, 5), dtype=jnp.float32)
    variables = model.init(
        {"params": jax.random.PRNGKey(0)},
        x,
        t=jnp.asarray(0.5, dtype=jnp.float32),
        cond=cond,
        train=False,
    )
    logits, _ = model.apply(
        variables,
        x,
        t=jnp.asarray(0.5, dtype=jnp.float32),
        cond=cond,
        train=False,
    )

    assert logits.shape == (1, 4, 4, 1, 8)
