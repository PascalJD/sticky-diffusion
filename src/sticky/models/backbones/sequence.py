from __future__ import annotations

import flax.linen as nn

from .networks import gpt2_like, sharded_transformer, transformer


class TransformerBackbone(nn.Module):
    """Transformer backbone wrapper for continuous/discrete classifier heads."""

    dim: int
    n_layers: int
    n_heads: int
    output_channels: int

    dropout_rate: float = 0.0
    use_attn_dropout: bool = True
    mlp_type: str = "swiglu"
    depth_scaled_init: bool = False
    cond_type: str = "adaln"
    multiple_of: int = 32

    embed_input: bool = False
    n_embed_classes: int = 1
    sharded: bool = False

    @nn.compact
    def __call__(self, z, *, cond=None, train: bool = False):
        if self.sharded:
            args = sharded_transformer.ModelArgs(
                dim=self.dim,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                n_kv_heads=self.n_heads,
                output_channels=self.output_channels,
                multiple_of=self.multiple_of,
                dropout_rate=self.dropout_rate,
                depth_scaled_init=self.depth_scaled_init,
                mlp_type=self.mlp_type,
                cond_type=self.cond_type,
                embed_input=self.embed_input,
                n_embed_classes=self.n_embed_classes,
                use_attn_dropout=self.use_attn_dropout,
            )
            net = sharded_transformer.Transformer(args)
        else:
            args = transformer.ModelArgs(
                dim=self.dim,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                n_kv_heads=self.n_heads,
                output_channels=self.output_channels,
                multiple_of=self.multiple_of,
                dropout_rate=self.dropout_rate,
                depth_scaled_init=self.depth_scaled_init,
                mlp_type=self.mlp_type,
                cond_type=self.cond_type,
                embed_input=self.embed_input,
                n_embed_classes=self.n_embed_classes,
            )
            net = transformer.Transformer(args)
        return net(z, cond=cond, train=train)


class GPT2LikeBackbone(nn.Module):
    """GPT-2-style sequence backbone with learned absolute positions."""

    dim: int
    n_layers: int
    n_heads: int
    output_channels: int

    dropout_rate: float = 0.0
    use_attn_dropout: bool = True
    embed_input: bool = False
    n_embed_classes: int = 1
    hidden_dim: int | None = None
    max_seq_len: int | None = None
    causal: bool = False

    @nn.compact
    def __call__(self, z, *, cond=None, train: bool = False):
        args = gpt2_like.ModelArgs(
            dim=self.dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            output_channels=self.output_channels,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            use_attn_dropout=self.use_attn_dropout,
            embed_input=self.embed_input,
            n_embed_classes=self.n_embed_classes,
            max_seq_len=self.max_seq_len,
            causal=self.causal,
        )
        net = gpt2_like.GPT2LikeTransformer(args)
        return net(z, cond=cond, train=train)
