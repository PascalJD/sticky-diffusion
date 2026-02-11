from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from sticky.models.md4.backward import CondEmbedding, UNet5DWrapper
from sticky.models.md4.networks import sharded_transformer, transformer


class ContinuousClassifier(nn.Module):
    # backbone config
    n_layers: int = 12
    n_dit_layers: int = 0
    dit_num_heads: int = 12
    dit_hidden_size: int = 768
    ch_mult: Sequence[int] = (1,)
    feature_dim: int = 64
    num_heads: int = 12

    # output space (number of anchors / tokens)
    vocab_size: int = 256

    # regularization / MLP choices (for transformer mode)
    dropout_rate: float = 0.0
    use_attn_dropout: bool = True
    mlp_type: str = "swiglu"
    depth_scaled_init: bool = False
    cond_type: str = "adaln"
    model_sharding: bool = False

    # continuous-input specific
    project_input: bool = True
    """If True, project last dim (d_anchor) -> feature_dim before UNet path.
    If False, feed raw d_anchor to the UNet path (cheaper first conv, but input
    channel count differs from MD4).
    """

    @nn.compact
    def __call__(self, z, t=None, cond=None, train: bool = False):
        """Forward.

        z:
          - (B, S, d)  sequence continuous embeddings
          - (B, H, W, d) or (B, H, W, C, d) image continuous embeddings
        t:
          - scalar or (B,)
        cond:
          - optional conditioning vector already in embedding space (B, ?)
        """
        if t is not None:
            # Ensure vector time and build (time, cond) embedding as in MD4
            assert jnp.isscalar(t) or t.ndim == 0 or t.ndim == 1
            t = t * jnp.ones(z.shape[0], dtype=jnp.asarray(t).dtype)
            cond = CondEmbedding(self.feature_dim)(t * 1000.0, cond=cond)

        # Sequence mode: (B, S, d_anchor)
        if z.ndim == 3:
            # Use the same Transformer stack as MD4, but with embed_input=False
            # since z is already continuous.
            if self.model_sharding:
                args = sharded_transformer.ModelArgs(
                    dim=self.feature_dim * self.num_heads,
                    n_layers=self.n_layers,
                    n_heads=self.num_heads,
                    n_kv_heads=self.num_heads,
                    output_channels=self.vocab_size,
                    multiple_of=32,
                    dropout_rate=self.dropout_rate,
                    depth_scaled_init=self.depth_scaled_init,
                    mlp_type=self.mlp_type,
                    cond_type=self.cond_type,
                    embed_input=False,
                    n_embed_classes=1,  # unused when embed_input=False
                    use_attn_dropout=self.use_attn_dropout,
                )
                net = sharded_transformer.Transformer(args)
            else:
                args = transformer.ModelArgs(
                    dim=self.feature_dim * self.num_heads,
                    n_layers=self.n_layers,
                    n_heads=self.num_heads,
                    n_kv_heads=self.num_heads,
                    output_channels=self.vocab_size,
                    multiple_of=32,
                    dropout_rate=self.dropout_rate,
                    depth_scaled_init=self.depth_scaled_init,
                    mlp_type=self.mlp_type,
                    cond_type=self.cond_type,
                    embed_input=False,
                    n_embed_classes=1,
                )
                net = transformer.Transformer(args)

            logits = net(z, cond=cond, train=train)
            return logits, {}

        # Image mode (no explicit channel): (B, H, W, d_anchor)
        if z.ndim == 4:
            z = z[:, :, :, None, :]  # -> (B, H, W, 1, d_anchor)
            squeeze_channel = True
        else:
            squeeze_channel = False

        # Image mode (with channel): (B, H, W, C, d_anchor)
        if z.ndim == 5:
            if self.project_input:
                # Match MD4 behavior: token_id -> feature_dim embedding.
                z = nn.Dense(self.feature_dim, name="input_proj")(z)

            net = UNet5DWrapper(
                feature_dim=self.feature_dim,
                n_layers=self.n_layers,
                n_dit_layers=self.n_dit_layers,
                dit_num_heads=self.dit_num_heads,
                dit_hidden_size=self.dit_hidden_size,
                ch_mult=self.ch_mult,
                output_channels=self.vocab_size,
                dropout_rate=self.dropout_rate,
            )
            logits = net(z, cond=cond, train=train)  # (B,H,W,C,L)

            if squeeze_channel:
                logits = logits.squeeze(axis=-2)  # (B,H,W,L)
            return logits, {}

        raise NotImplementedError(
            f"ContinuousClassifier expects z.ndim in {{3,4,5}}, got {z.ndim} with shape {z.shape}."
        )
