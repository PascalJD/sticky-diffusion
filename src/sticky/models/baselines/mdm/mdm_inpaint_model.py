from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from sticky.models.backbones import DiscreteClassifier

from . import board_sampling


Array = jnp.ndarray


class MDMInpaint(nn.Module):
    """Minimal board-level masked discrete model for Sudoku inpainting.

    This path is intentionally separate from the legacy packed seq2seq Sudoku
    MDM implementation. It operates directly on 81 board cells with token `0`
    as the only mask / blank symbol.
    """

    data_shape: tuple[int, ...]
    cont_time: bool = False
    timesteps: int = 50
    feature_dim: int = 128
    num_heads: int = 12
    antithetic_time_sampling: bool = False
    n_layers: int = 3
    n_dit_layers: int = 0
    dit_num_heads: int = 12
    dit_hidden_size: int = 768
    ch_mult: Sequence[int] = (1,)
    vocab_size: int = 10
    noise_schedule_type: str = "loglinear"
    dropout_rate: float = 0.0
    use_attn_dropout: bool = True
    mlp_type: str = "gelu"
    depth_scaled_init: bool = False
    cond_type: str = "adaln"
    outside_embed: bool = False
    sequence_backbone: str = "gpt2_like"
    sequence_mlp_hidden_dim: int | None = None
    sequence_max_length: int | None = None
    sequence_causal: bool = False
    image_backbone: str = "auto"
    adm_num_res_blocks: int = 2
    adm_attention_resolutions: Sequence[int] = (2, 4)
    adm_num_heads: int = 4
    adm_num_head_channels: int = -1
    adm_num_heads_upsample: int = -1
    adm_conv_resample: bool = True
    adm_use_scale_shift_norm: bool = True
    adm_resblock_updown: bool = False
    adm_use_conv_skip: bool = False
    adm_use_new_attention_order: bool = False
    time_features: str = "none"
    classes: int = -1
    sampler: str = "top_prob_margin"
    sampling_grid: str = "loglinear"
    categorical_sampling_policy: str = "exact"
    model_sharding: bool = False

    def setup(self):
        self.classifier = DiscreteClassifier(
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
            outside_embed=self.outside_embed,
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

    @property
    def mask_token_id(self) -> int:
        return 0

    def prior_sample(self, batch_size: int) -> Array:
        return jnp.zeros(
            (int(batch_size), *tuple(int(v) for v in self.data_shape)),
            dtype=jnp.int32,
        )

    def predict_x(
        self,
        zt: Array,
        t: Array | None = None,
        *,
        cond: Array | None = None,
        train: bool = False,
    ) -> tuple[Array, dict[str, Any]]:
        return self.classifier(zt, t=t, cond=cond, train=train)

    def predict_logits(
        self,
        zt: Array,
        t: Array | None = None,
        *,
        cond: Array | None = None,
        train: bool = False,
    ) -> Array:
        logits, _ = self.predict_x(zt, t=t, cond=cond, train=train)
        return logits

    def token_cross_entropy(
        self,
        logits: Array,
        targets: Array,
    ) -> Array:
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.take_along_axis(
            log_probs,
            jnp.expand_dims(targets.astype(jnp.int32), axis=-1),
            axis=-1,
        )[..., 0]

    def __call__(
        self,
        zt: Array,
        t: Array | None = None,
        *,
        cond: Array | None = None,
        train: bool = False,
    ) -> dict[str, Array]:
        logits = self.predict_logits(zt, t=t, cond=cond, train=train)
        return {"logits": logits}

    def sample_step(
        self,
        rng: Array,
        i: int,
        timesteps: int,
        state: Array,
        *,
        conditioning: Array | None = None,
        known_token_mask: Array | None = None,
        known_tokens: Array | None = None,
        sampler_override: str | None = None,
        return_info: bool = False,
    ) -> Array | tuple[Array, dict[str, Array]]:
        method = self.sampler if sampler_override is None else sampler_override
        return board_sampling.reveal_order_sample_step(
            self,
            rng,
            i,
            timesteps,
            state,
            conditioning=conditioning,
            known_token_mask=known_token_mask,
            known_tokens=known_tokens,
            method=method,
            return_info=return_info,
        )

    def decode(
        self,
        x: Array,
        *,
        conditioning: Array | None = None,
    ) -> Array:
        del conditioning
        arr = jnp.asarray(x)
        if arr.ndim >= 1 and arr.shape[-1] == int(self.vocab_size):
            return jnp.argmax(arr, axis=-1).astype(jnp.int32)
        return arr.astype(jnp.int32)
