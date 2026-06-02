# src/sticky/models/sjd/sjd_model.py
from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import flax.linen as nn
import jax.numpy as jnp

from .anchors import AnchorTableConfig, TokenAnchors
from .classifier import ContinuousClassifier

Array = jnp.ndarray


class SJD(nn.Module):
    anchor_config: AnchorTableConfig
    learnable_anchors: bool = True
    learnable_log_w: bool = False
    log_w_init: Any = None
    # See TokenAnchors.log_w_clip; threaded through here so the anchor
    # factory in models/factories/sjd.py can wire hazard_weighting.clip.
    log_w_clip: tuple[float, float] | None = None
    # When True, the model maintains a learnable anchor_dim-sized bias
    # vector that is added to y_t at uncommitted (noisy) site positions
    # before the classifier sees it. The mask is supplied per-call via the
    # `noisy_position_mask` kwarg of `__call__`. Defaults to False so
    # existing tasks/checkpoints are unaffected.
    use_noisy_input_bias: bool = False

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
    sequence_mlp_hidden_dim: int | None = None
    sequence_max_length: int | None = None
    sequence_causal: bool = False
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

    vocab_size: int = 256

    def setup(self):
        if int(self.anchor_config.vocab_size) != int(self.vocab_size):
            raise ValueError(
                "SJD anchor_config.vocab_size must match model vocab_size, got "
                f"{self.anchor_config.vocab_size} vs {self.vocab_size}."
            )
        self.anchors = TokenAnchors(
            config=self.anchor_config,
            learnable=self.learnable_anchors,
            learnable_log_w=self.learnable_log_w,
            log_w_init=self.log_w_init,
            log_w_clip=self.log_w_clip,
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
            sequence_mlp_hidden_dim=self.sequence_mlp_hidden_dim,
            sequence_max_length=self.sequence_max_length,
            sequence_causal=self.sequence_causal,
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
            adm_use_new_attention_order=self.adm_use_new_attention_order,
        )
        if self.use_noisy_input_bias:
            self.noisy_input_bias = self.param(
                "noisy_input_bias",
                nn.initializers.zeros,
                (int(self.anchor_config.anchor_dim),),
                jnp.float32,
            )

    def embed(self, token_ids: Array) -> Array:
        return self.anchors(token_ids)

    def anchor_table(self) -> Array:
        return self.anchors.table_float()

    def anchor_log_w(self) -> Array | None:
        return self.anchors.log_w_float()

    def __call__(
        self,
        y_t: Array,
        t: Array,
        *,
        cond=None,
        anchor_token_ids: Array | None = None,
        noisy_position_mask: Array | None = None,
        train: bool = False,
    ):
        if anchor_token_ids is not None:
            _ = self.embed(anchor_token_ids)
        cell_input = y_t
        if self.use_noisy_input_bias:
            if noisy_position_mask is None:
                noisy_mask = jnp.zeros(cell_input.shape[:-1], dtype=jnp.bool_)
            else:
                noisy_mask = jnp.asarray(noisy_position_mask, dtype=jnp.bool_)
                if noisy_mask.shape != cell_input.shape[:-1]:
                    raise ValueError(
                        "noisy_position_mask must match y_t without the "
                        f"feature dimension, got {noisy_mask.shape} vs "
                        f"{cell_input.shape[:-1]}."
                    )
            cell_input = cell_input + noisy_mask[..., None].astype(
                cell_input.dtype
            ) * self.noisy_input_bias.astype(cell_input.dtype)
        return self.classifier(cell_input, t=t, cond=cond, train=train)
