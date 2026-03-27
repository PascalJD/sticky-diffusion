from __future__ import annotations

import jax
import jax.numpy as jnp

from sticky.models.backbones.sequence import GPT2LikeBackbone


def test_gpt2_like_backbone_outputs_vocab_logits_for_sudoku_length_sequences():
    backbone = GPT2LikeBackbone(
        dim=96,
        n_layers=2,
        n_heads=4,
        output_channels=10,
        dropout_rate=0.0,
        use_attn_dropout=False,
        embed_input=True,
        n_embed_classes=11,
        hidden_dim=384,
        max_seq_len=243,
        causal=False,
    )

    x = jnp.zeros((2, 243), dtype=jnp.int32)
    variables = backbone.init({"params": jax.random.PRNGKey(0)}, x, train=False)
    logits = backbone.apply(variables, x, train=False)

    assert logits.shape == (2, 243, 10)
    assert logits.dtype == jnp.float32
