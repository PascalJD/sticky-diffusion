from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

from .factory import build_image_backbone, build_sequence_backbone
from .networks.conditioning import CondEmbedding


class DiscreteClassifier(nn.Module):
    """Discrete-token classifier that reuses shared architecture backbones."""

    n_layers: int = 12
    n_dit_layers: int = 0
    dit_num_heads: int = 12
    dit_hidden_size: int = 768
    ch_mult: Sequence[int] = (1,)
    feature_dim: int = 64
    num_heads: int = 12
    vocab_size: int = 1000

    dropout_rate: float = 0.0
    use_attn_dropout: bool = True
    mlp_type: str = "swiglu"
    depth_scaled_init: bool = False
    cond_type: str = "adaln"
    outside_embed: bool = False
    model_sharding: bool = False
    sequence_backbone: str = "auto"
    image_backbone: str = "auto"
    adm_num_res_blocks: int = 2
    adm_attention_resolutions: Sequence[int] = (2, 4, 8)
    adm_num_heads: int = 4
    adm_num_head_channels: int = -1
    adm_num_heads_upsample: int = -1
    adm_conv_resample: bool = True
    adm_use_scale_shift_norm: bool = True
    adm_resblock_updown: bool = False
    adm_use_conv_skip: bool = False

    @nn.compact
    def __call__(self, z, t=None, cond=None, train: bool = False):
        if t is not None:
            assert jnp.isscalar(t) or t.ndim == 0 or t.ndim == 1
            t = t * jnp.ones(z.shape[0], dtype=jnp.asarray(t).dtype)
            cond = CondEmbedding(self.feature_dim)(t * 1000.0, cond=cond)

        if z.ndim == 2:
            if self.outside_embed:
                z = nn.Embed(self.vocab_size + 1, self.feature_dim)(z)

            net = build_sequence_backbone(
                name=self.sequence_backbone,
                feature_dim=self.feature_dim,
                num_heads=self.num_heads,
                n_layers=self.n_layers,
                vocab_size=self.vocab_size,
                dropout_rate=self.dropout_rate,
                use_attn_dropout=self.use_attn_dropout,
                mlp_type=self.mlp_type,
                depth_scaled_init=self.depth_scaled_init,
                cond_type=self.cond_type,
                model_sharding=self.model_sharding,
                embed_input=not self.outside_embed,
                n_embed_classes=self.vocab_size + 1,
            )
            logits = net(z, cond=cond, train=train)
            return logits, {}

        if z.ndim == 4:
            z = nn.Embed(self.vocab_size + 1, self.feature_dim)(z)
            net = build_image_backbone(
                name=self.image_backbone,
                feature_dim=self.feature_dim,
                n_layers=self.n_layers,
                n_dit_layers=self.n_dit_layers,
                dit_num_heads=self.dit_num_heads,
                dit_hidden_size=self.dit_hidden_size,
                ch_mult=self.ch_mult,
                vocab_size=self.vocab_size,
                dropout_rate=self.dropout_rate,
                adm_num_res_blocks=self.adm_num_res_blocks,
                adm_attention_resolutions=self.adm_attention_resolutions,
                adm_num_heads=self.adm_num_heads,
                adm_num_head_channels=self.adm_num_head_channels,
                adm_num_heads_upsample=self.adm_num_heads_upsample,
                adm_conv_resample=self.adm_conv_resample,
                adm_use_scale_shift_norm=self.adm_use_scale_shift_norm,
                adm_resblock_updown=self.adm_resblock_updown,
                adm_use_conv_skip=self.adm_use_conv_skip,
            )
            logits = net(z, cond=cond, train=train)
            return logits, {}

        raise NotImplementedError(
            f"DiscreteClassifier expects z.ndim in {{2,4}}, got {z.ndim} with shape {z.shape}."
        )
