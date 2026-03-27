from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

from sticky.models.backbones import (
    CondEmbedding,
    build_image_backbone,
    build_sequence_backbone,
)


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
    adm_use_new_attention_order: bool = False

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
        time_cond = cond
        adm_timesteps = None
        if t is not None:
            # Ensure vector time and build (time, cond) embedding as in MD4
            assert jnp.isscalar(t) or t.ndim == 0 or t.ndim == 1
            t = t * jnp.ones(z.shape[0], dtype=jnp.asarray(t).dtype)
            adm_timesteps = t * 1000.0
            if str(self.image_backbone).lower() != "adm_unet5d" or z.ndim == 3:
                time_cond = CondEmbedding(self.feature_dim)(adm_timesteps, cond=cond)

        # Sequence mode: (B, S, d_anchor)
        if z.ndim == 3:
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
            )
            logits = net(z, cond=time_cond, train=train)
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
                adm_use_new_attention_order=self.adm_use_new_attention_order,
            )
            logits = net(
                z,
                cond=cond if str(self.image_backbone).lower() == "adm_unet5d" else time_cond,
                timesteps=adm_timesteps if str(self.image_backbone).lower() == "adm_unet5d" else None,
                train=train,
            )  # (B,H,W,C,L)

            if squeeze_channel:
                logits = logits.squeeze(axis=-2)  # (B,H,W,L)
            return logits, {}

        raise NotImplementedError(
            f"ContinuousClassifier expects z.ndim in {{3,4,5}}, got {z.ndim} with shape {z.shape}."
        )
