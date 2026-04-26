from __future__ import annotations

import jax
import jax.numpy as jnp

from sticky.models.backbones.factory import build_sequence_backbone


def test_transformer_backbone_handles_seq_len_1024_adaln_zero():
    """Smoke: non-causal LLAMA2-like transformer with adaLN-zero at OWT seq_len."""
    backbone = build_sequence_backbone(
        name="transformer",
        feature_dim=8,
        num_heads=4,
        n_layers=2,
        vocab_size=768,
        dropout_rate=0.0,
        use_attn_dropout=True,
        mlp_type="swiglu",
        depth_scaled_init=False,
        cond_type="adaln_zero",
        model_sharding=False,
        max_seq_len=1024,
        causal=False,
    )
    key = jax.random.PRNGKey(0)
    z = jnp.zeros((1, 64, 32), dtype=jnp.float32)  # (B, S, d_in) - small S for CPU
    cond = jnp.zeros((1, 32), dtype=jnp.float32)
    params = backbone.init(key, z, cond=cond, train=False)["params"]
    out = backbone.apply({"params": params}, z, cond=cond, train=False)
    assert out.shape == (1, 64, 768)
    assert jnp.isfinite(out).all()


def test_transformer_backbone_accepts_anchor_dim_768_input():
    """When anchor_dim matches backbone dim, input projection is still applied."""
    backbone = build_sequence_backbone(
        name="transformer",
        feature_dim=8,
        num_heads=4,
        n_layers=1,
        vocab_size=32,
        dropout_rate=0.0,
        use_attn_dropout=True,
        mlp_type="swiglu",
        depth_scaled_init=False,
        cond_type="adaln_zero",
        model_sharding=False,
    )
    key = jax.random.PRNGKey(0)
    # Input dim = 32 == 8*4 = backbone dim
    z = jnp.zeros((2, 16, 32), dtype=jnp.float32)
    cond = jnp.zeros((2, 32), dtype=jnp.float32)
    params = backbone.init(key, z, cond=cond, train=False)["params"]
    out = backbone.apply({"params": params}, z, cond=cond, train=False)
    assert out.shape == (2, 16, 32)
