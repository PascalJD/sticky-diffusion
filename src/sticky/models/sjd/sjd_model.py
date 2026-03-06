# src/sticky/models/sjd/sjd_model.py
from __future__ import annotations
from collections.abc import Sequence
import flax.linen as nn
import jax.numpy as jnp

from .anchors import TokenAnchors
from .classifier import ContinuousClassifier

Array = jnp.ndarray


class SJD(nn.Module):
    anchor_dim: int = 64
    anchor_init: str = "normal"
    learnable_anchors: bool = True
    anchors_init_std: float = 1.0

    feature_dim: int = 128
    num_heads: int = 12
    n_layers: int = 32
    n_dit_layers: int = 0
    dit_num_heads: int = 12
    dit_hidden_size: int = 768
    ch_mult: Sequence[int] = (1,)

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

    vocab_size: int = 256

    def setup(self):
        self.anchors = TokenAnchors(
            vocab_size=self.vocab_size,
            anchor_dim=self.anchor_dim,
            anchor_init=self.anchor_init,
            learnable=self.learnable_anchors,
            init_std=self.anchors_init_std,
        )
        self.classifier = ContinuousClassifier(
            n_layers=self.n_layers,
            n_dit_layers=self.n_dit_layers,
            dit_num_heads=self.dit_num_heads,
            dit_hidden_size=self.dit_hidden_size,
            ch_mult=self.ch_mult,
            feature_dim=self.feature_dim,
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
            dropout_rate=self.dropout_rate,
            use_attn_dropout=self.use_attn_dropout,
            mlp_type=self.mlp_type,
            depth_scaled_init=self.depth_scaled_init,
            cond_type=self.cond_type,
            model_sharding=self.model_sharding,
            sequence_backbone=self.sequence_backbone,
            image_backbone=self.image_backbone,
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

    def embed(self, token_ids: Array) -> Array:
        return self.anchors(token_ids)

    def anchor_table(self) -> Array:
        return self.anchors.table_float()

    def __call__(self, y_t: Array, t: Array, *, cond=None, train: bool = False):
        return self.classifier(y_t, t=t, cond=cond, train=train)
