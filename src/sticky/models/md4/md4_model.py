# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simplified masked diffusion (MD4) model (Flax/JAX). This is an architecture-only port of the MD4 model suitable for use inside the
Sticky repository. It avoids the TensorFlow Probability dependency by using JAX-native sampling utilities.
"""

from __future__ import annotations

import math
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from sticky.models.architectures import DiscreteClassifier
from sticky.models.discrete_mixture import (
    categorical_sample_from_probs,
    sample_mixture_categorical,
)
from sticky.models import masked_discrete_core as masked_core

from . import binary_search
from . import utils

Array = jnp.ndarray


def _categorical_sample_from_probs(rng: Array, probs: Array) -> Array:
  """Sample categorical indices from probability tensor [..., K]."""
  return categorical_sample_from_probs(rng, probs)


class MaskingSchedule(nn.Module):

  data_shape: tuple[int, ...]
  schedule_fn_type: str = "cosine"
  eps: float = 1e-4

  def __call__(self, t: Array) -> Array:
    return masked_core.masked_logit_schedule(
        t,
        schedule_fn_type=self.schedule_fn_type,
        eps=self.eps,
    )

  def _dalpha(self, t: Array) -> Array:
    return masked_core.clipped_schedule_dalpha(
        t,
        schedule_fn_type=self.schedule_fn_type,
        eps=0.0,
    )

  def dalpha(self, t: Array) -> Array:
    return masked_core.clipped_schedule_dalpha(
        t,
        schedule_fn_type=self.schedule_fn_type,
        eps=self.eps,
    )

  def _alpha(self, t: Array) -> Array:
    return masked_core.clipped_schedule_alpha(
        t,
        schedule_fn_type=self.schedule_fn_type,
        eps=0.0,
    )

  def alpha(self, t: Array) -> Array:
    return masked_core.clipped_schedule_alpha(
        t,
        schedule_fn_type=self.schedule_fn_type,
        eps=self.eps,
    )

  def dgamma_times_alpha(self, t: Array) -> Array:
    return masked_core.masked_dgamma_times_alpha(
        t,
        schedule_fn_type=self.schedule_fn_type,
        eps=self.eps,
    )


class MD4(nn.Module):

  data_shape: tuple[int, ...]
  cont_time: bool = False
  timesteps: int = 1000
  feature_dim: int = 128
  num_heads: int = 12
  antithetic_time_sampling: bool = True
  n_layers: int = 32
  n_dit_layers: int = 0
  dit_num_heads: int = 12
  dit_hidden_size: int = 768
  ch_mult: Sequence[int] = (1,)
  vocab_size: int = 256
  noise_schedule_type: str = "linear"
  dropout_rate: float = 0.0
  use_attn_dropout: bool = True
  mlp_type: str = "swiglu"
  depth_scaled_init: bool = False
  cond_type: str = "adaln"
  outside_embed: bool = False
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
  time_features: str = "t"  # 't' or 'none'
  classes: int = 10 + 1  # set <=0 for unconditional
  sampler: str = "ancestral"  # ancestral, mean (legacy JAX-unsafe), topp
  sampling_grid: str = "cosine"  # uniform, cosine
  topp: float = 0.98
  model_sharding: bool = False

  def setup(self):
    self.noise_schedule = MaskingSchedule(self.data_shape, self.noise_schedule_type)

    if self.classes > 0:
      self.cond_embeddings = nn.Embed(self.classes, self.feature_dim)

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
    return masked_core.mask_token_id(self.vocab_size)

  # Forward (noising) process

  def forward_sample(self, x: Array, t: Array) -> Array:
    keep_prob = self.noise_schedule.alpha(masked_core.reverse_broadcast(t, x.ndim))
    return masked_core.sample_masked_discrete_forward(
        self.make_rng("sample"),
        x,
        keep_prob=keep_prob,
        mask_token_id=self.mask_token_id,
    )

  def prior_sample(self, batch_size: int) -> Array:
    return masked_core.make_masked_token_prior(
        batch_size=batch_size,
        data_shape=self.data_shape,
        vocab_size=self.vocab_size,
    )

  # Conditioning helpers

  def get_cond_embedding(self, conditioning: Array | None) -> Array | None:
    if (self.classes > 0) and (conditioning is not None):
      return self.cond_embeddings(conditioning)
    return None

  # Model prediction utilities

  def predict_x(self, zt: Array, t: Array, *, cond: Array | None = None, train: bool = False) -> tuple[Array, dict]:
    t_in = None if self.time_features == "none" else t
    return self.classifier(zt, t=t_in, cond=cond, train=train)

  def decode(self, z0: Array, *, conditioning: Array | None = None) -> Array:
    """Map tokens to final tokens (replace any remaining mask tokens)."""
    masked = z0 == self.mask_token_id
    z0_clipped = jnp.where(masked, jnp.zeros_like(z0), z0)

    cond = self.get_cond_embedding(conditioning)
    logits, _ = self.predict_x(z0, jnp.array(0.0), cond=cond, train=False)

    # For masked positions, choose argmax prediction; otherwise keep existing token.
    pred = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    return jnp.where(masked, pred, z0_clipped).astype(jnp.int32)

  # Loss terms (training)

  def recon_loss(self) -> Array:
    alpha_t1 = self.noise_schedule.alpha(jnp.array(0.0))
    return jnp.prod(jnp.asarray(self.data_shape)) * (1.0 - alpha_t1) * jnp.log(self.vocab_size)

  def latent_loss(self) -> Array:
    return jnp.array(0.0)

  def diffusion_loss(self, t: Array, x: Array, *, cond: Array | None = None, train: bool = False) -> Array:
    if not self.cont_time:
      t = (jnp.floor(t * self.timesteps) + 1.0) / self.timesteps

    zt = self.forward_sample(x, t)
    logits, _ = self.predict_x(zt, t, cond=cond, train=train)

    mask = masked_core.masked_positions(zt, mask_token_id=self.mask_token_id)
    masked_neg_cross_ent = masked_core.masked_logprob_sums(logits, x, mask=mask)

    if not self.cont_time:
      s = t - (1.0 / self.timesteps)
      gt = self.noise_schedule(t)
      gs = self.noise_schedule(s)
      loss_diff = self.timesteps * jnp.expm1(gt - gs) * self.noise_schedule.alpha(s) * masked_neg_cross_ent
    else:
      loss_diff = self.noise_schedule.dgamma_times_alpha(t) * masked_neg_cross_ent

    return loss_diff  # [B]

  # Flax forward

  @nn.compact
  def __call__(self, x: Array, *, cond: Array | None = None, train: bool = False) -> dict[str, Array]:
    bs = x.shape[0]
    cond_emb = self.get_cond_embedding(cond)

    loss_recon = self.recon_loss()
    loss_prior = self.latent_loss()

    rng1 = self.make_rng("sample")
    if self.antithetic_time_sampling:
      t0 = jax.random.uniform(rng1)
      t = jnp.mod(t0 + jnp.arange(0.0, 1.0, step=1.0 / bs), 1.0)
    else:
      t = jax.random.uniform(rng1, shape=[bs])

    loss_diff = self.diffusion_loss(t, x, cond=cond_emb, train=train).mean()
    loss = loss_diff + loss_prior + loss_recon

    model_stats = {
        "loss": loss,
        "loss_diff": loss_diff,
        "loss_prior": loss_prior,
        "loss_recon": loss_recon,
    }
    return utils.loss2bpt(model_stats, self.data_shape)

  # Sampling utilities

  def get_sampling_grid(self, i: int, timesteps: int) -> tuple[Array, Array]:
    return masked_core.make_sampling_time_pair(
        i,
        timesteps,
        sampling_grid=self.sampling_grid,
    )

  def ancestral_sample_step(self, rng: Array, i: int, timesteps: int, zt: Array, *, conditioning: Array | None = None) -> Array:
    rng_body = jax.random.fold_in(rng, i)
    s, t = self.get_sampling_grid(i, timesteps)
    cond = self.get_cond_embedding(conditioning)

    alpha_t = self.noise_schedule.alpha(t)
    alpha_s = self.noise_schedule.alpha(s)

    logits, _ = self.predict_x(zt, t, cond=cond, train=False)
    mean_preds = jax.nn.softmax(logits, axis=-1)

    unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
    to_unmask, keep_mask = sample_mixture_categorical(
        rng_body,
        destination_probs=mean_preds,
        stay_prob=1.0 - unmask_prob,
        change_prob=unmask_prob,
    )
    sampled = jnp.where(keep_mask, self.mask_token_id, to_unmask).astype(jnp.int32)
    return masked_core.carry_over_unmasked(
        zt,
        sampled,
        mask_token_id=self.mask_token_id,
    )

  def topp_sample_step(
      self,
      rng: Array,
      i: int,
      timesteps: int,
      zt: Array,
      *,
      conditioning: Array | None = None,
      topp: float = 0.98,
  ) -> Array:
    rng_body = jax.random.fold_in(rng, i)
    s, t = self.get_sampling_grid(i, timesteps)
    cond = self.get_cond_embedding(conditioning)

    alpha_t = self.noise_schedule.alpha(t)
    alpha_s = self.noise_schedule.alpha(s)

    logits, _ = self.predict_x(zt, t, cond=cond, train=False)
    logits = binary_search.topp_mask(logits, topp, replace_val=jnp.array(-1e7))
    mean_preds = jax.nn.softmax(logits, axis=-1)

    unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
    to_unmask, keep_mask = sample_mixture_categorical(
        rng_body,
        destination_probs=mean_preds,
        stay_prob=1.0 - unmask_prob,
        change_prob=unmask_prob,
    )
    sampled = jnp.where(keep_mask, self.vocab_size, to_unmask).astype(jnp.int32)
    is_mask = zt == self.vocab_size
    return jnp.where(is_mask, sampled, zt)

  def mean_sample_step(self, rng: Array, i: int, timesteps: int, zt: Array, *, conditioning: Array | None = None) -> Array:
    """Legacy split sampler retained for backwards compatibility.

    MD4 Appendix G notes that in JAX this Bernoulli + categorical
    decomposition can produce worse sample quality than the one-shot mixture
    used by `ancestral_sample_step`. Keep it only for explicit legacy runs.
    """
    rng_body = jax.random.fold_in(rng, i)
    s, t = self.get_sampling_grid(i, timesteps)
    cond = self.get_cond_embedding(conditioning)

    alpha_t = self.noise_schedule.alpha(t)
    alpha_s = self.noise_schedule.alpha(s)

    logits, _ = self.predict_x(zt, t, cond=cond, train=False)
    unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)

    rng_body, rng = jax.random.split(rng_body)
    z0 = jax.random.categorical(rng_body, logits, axis=-1).astype(jnp.int32)

    rng_body, _ = jax.random.split(rng)
    unmask = jax.random.bernoulli(rng_body, unmask_prob, zt.shape)
    proposal = jnp.where(unmask, z0, self.mask_token_id).astype(jnp.int32)
    return masked_core.carry_over_unmasked(
        zt,
        proposal,
        mask_token_id=self.mask_token_id,
    )

  def sample_step(
      self,
      rng: Array,
      i: int,
      timesteps: int,
      zt: Array,
      *,
      conditioning: Array | None = None,
      topp: float | None = None,
  ) -> Array:
    if self.sampler == "ancestral":
      return self.ancestral_sample_step(rng, i, timesteps, zt, conditioning=conditioning)
    if self.sampler == "topp":
      topp_eff = self.topp if topp is None else topp
      return self.topp_sample_step(rng, i, timesteps, zt, conditioning=conditioning, topp=topp_eff)
    if self.sampler == "mean":
      return self.mean_sample_step(rng, i, timesteps, zt, conditioning=conditioning)
    raise NotImplementedError(f"Unknown sampler={self.sampler}")
