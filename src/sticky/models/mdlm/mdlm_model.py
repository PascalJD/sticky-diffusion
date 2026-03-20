from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from sticky.models import masked_discrete_core as masked_core
from sticky.models.architectures import DiscreteClassifier
from sticky.models.discrete_mixture import (
    categorical_sample_from_logits,
    sample_mixture_categorical,
)


Array = jnp.ndarray
SamplingState = tuple[Array, Array, Array]


def _loss_to_bpd(loss_dict: dict[str, Array], data_shape: tuple[int, ...]) -> dict[str, Array]:
    seq_len = jnp.prod(jnp.asarray(data_shape))
    scale = 1.0 / (seq_len * jnp.log(2.0))
    return {k: (v * scale if "loss" in k else v) for k, v in loss_dict.items()}


def _selected_log_prob_sums(log_probs: Array, targets: Array) -> Array:
    selected = jnp.take_along_axis(
        log_probs,
        jnp.expand_dims(targets, axis=-1),
        axis=-1,
    )[..., 0]
    return jnp.sum(selected, axis=tuple(range(1, targets.ndim)))


def _topp_mask(logits: Array, topp: float, replace_val: Array) -> Array:
    probs = jax.nn.softmax(logits, axis=-1)
    sorted_idx = jnp.argsort(probs, axis=-1)[..., ::-1]
    sorted_probs = jnp.take_along_axis(probs, sorted_idx, axis=-1)
    sorted_cdf = jnp.cumsum(sorted_probs, axis=-1)
    keep_sorted = (sorted_cdf - sorted_probs) < float(topp)
    keep_sorted = keep_sorted.at[..., 0].set(True)
    inverse_perm = jnp.argsort(sorted_idx, axis=-1)
    keep = jnp.take_along_axis(keep_sorted, inverse_perm, axis=-1)
    return jnp.where(keep, logits, jnp.asarray(replace_val, dtype=logits.dtype))


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

    def dalpha(self, t: Array) -> Array:
        return masked_core.clipped_schedule_dalpha(
            t,
            schedule_fn_type=self.schedule_fn_type,
            eps=self.eps,
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


class MDLM(nn.Module):
    """Image-first masked diffusion with SUBS-style parameterization."""

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
    image_backbone: str = "adm_unet5d"
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
    time_features: str = "t"
    classes: int = -1
    sampler: str = "ancestral"
    sampling_grid: str = "cosine"
    topp: float = 0.98
    categorical_sampling_policy: str = "legacy_low"
    cache_predictions: bool = False
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

    def get_cond_embedding(self, conditioning: Array | None) -> Array | None:
        if (self.classes > 0) and (conditioning is not None):
            return self.cond_embeddings(conditioning)
        return None

    def predict_x(
        self,
        zt: Array,
        t: Array,
        *,
        cond: Array | None = None,
        train: bool = False,
    ) -> tuple[Array, dict[str, Any]]:
        t_in = None if self.time_features == "none" else t
        return self.classifier(zt, t=t_in, cond=cond, train=train)

    def _subs_parameterize(self, logits: Array, zt: Array) -> Array:
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        carry_tokens = jnp.clip(zt, 0, self.vocab_size - 1)
        carry_log_probs = jnp.where(
            jax.nn.one_hot(carry_tokens, self.vocab_size, dtype=log_probs.dtype) > 0,
            jnp.asarray(0.0, dtype=log_probs.dtype),
            jnp.asarray(-jnp.inf, dtype=log_probs.dtype),
        )
        is_unmasked = zt != self.mask_token_id
        return jnp.where(is_unmasked[..., None], carry_log_probs, log_probs)

    def predict_clean_log_probs(
        self,
        zt: Array,
        t: Array,
        *,
        cond: Array | None = None,
        train: bool = False,
    ) -> tuple[Array, dict[str, Any]]:
        logits, aux = self.predict_x(zt, t, cond=cond, train=train)
        return self._subs_parameterize(logits, zt), aux

    def recon_loss(self) -> Array:
        return jnp.array(0.0, dtype=jnp.float32)

    def latent_loss(self) -> Array:
        return jnp.array(0.0, dtype=jnp.float32)

    def diffusion_loss(
        self,
        t: Array,
        x: Array,
        *,
        cond: Array | None = None,
        train: bool = False,
    ) -> Array:
        if not self.cont_time:
            t = (jnp.floor(t * self.timesteps) + 1.0) / self.timesteps

        zt = self.forward_sample(x, t)
        log_probs, _ = self.predict_clean_log_probs(zt, t, cond=cond, train=train)
        log_p_theta_sum = _selected_log_prob_sums(log_probs, x)

        if not self.cont_time:
            s = t - (1.0 / self.timesteps)
            gt = self.noise_schedule(t)
            gs = self.noise_schedule(s)
            weight = self.timesteps * jnp.expm1(gt - gs) * self.noise_schedule.alpha(s)
        else:
            weight = self.noise_schedule.dgamma_times_alpha(t)

        # Match MD4's convention: the schedule weight is negative and the
        # selected clean-token log-prob sum is non-positive, so the diffusion
        # contribution stays positive.
        return weight * log_p_theta_sum

    @nn.compact
    def __call__(
        self,
        x: Array,
        *,
        cond: Array | None = None,
        train: bool = False,
    ) -> dict[str, Array]:
        batch_size = x.shape[0]
        cond_emb = self.get_cond_embedding(cond)

        loss_recon = self.recon_loss()
        loss_prior = self.latent_loss()

        rng = self.make_rng("sample")
        if self.antithetic_time_sampling:
            t0 = jax.random.uniform(rng)
            t = jnp.mod(t0 + jnp.arange(0.0, 1.0, step=1.0 / batch_size), 1.0)
        else:
            t = jax.random.uniform(rng, shape=[batch_size])

        loss_diff = self.diffusion_loss(t, x, cond=cond_emb, train=train).mean()
        loss = loss_diff + loss_prior + loss_recon

        stats = {
            "loss": loss,
            "loss_diff": loss_diff,
            "loss_prior": loss_prior,
            "loss_recon": loss_recon,
        }
        return _loss_to_bpd(stats, self.data_shape)

    def _use_cache(self) -> bool:
        return bool(self.cache_predictions) and str(self.time_features) == "none"

    def get_sampling_grid(self, i: int, timesteps: int) -> tuple[Array, Array]:
        return masked_core.make_sampling_time_pair(
            i,
            timesteps,
            sampling_grid=self.sampling_grid,
        )

    def _unpack_sampling_state(
        self,
        state: Array | SamplingState,
    ) -> tuple[Array, Array | None, Array | None, bool]:
        if isinstance(state, tuple) and len(state) == 3:
            tokens, cached_log_probs, cache_valid = state
            return tokens, cached_log_probs, cache_valid, True
        return state, None, None, False

    def _pack_sampling_state(
        self,
        tokens: Array,
        *,
        log_probs: Array | None,
        cache_valid: Array | None,
        structured: bool,
    ) -> Array | SamplingState:
        if not structured:
            return tokens

        if log_probs is None or cache_valid is None:
            log_probs = jnp.zeros(tokens.shape + (self.vocab_size,), dtype=jnp.float32)
            cache_valid = jnp.asarray(False)

        cache_valid = jnp.asarray(cache_valid, dtype=jnp.bool_)
        cached = jnp.where(cache_valid, log_probs, jnp.zeros_like(log_probs))
        return (tokens, cached, cache_valid)

    def _sampling_log_probs(
        self,
        state: Array | SamplingState,
        *,
        t: Array,
        conditioning: Array | None = None,
    ) -> tuple[Array, Array, bool]:
        tokens, cached_log_probs, cache_valid, structured = self._unpack_sampling_state(state)
        if structured and self._use_cache() and cache_valid is not None:
            cond = self.get_cond_embedding(conditioning)

            def _use_cached(_):
                return cached_log_probs

            def _recompute(_):
                log_probs, _ = self.predict_clean_log_probs(tokens, t, cond=cond, train=False)
                return log_probs

            log_probs = jax.lax.cond(cache_valid, _use_cached, _recompute, operand=None)
            return tokens, log_probs, structured

        cond = self.get_cond_embedding(conditioning)
        log_probs, _ = self.predict_clean_log_probs(tokens, t, cond=cond, train=False)
        return tokens, log_probs, structured

    def _sample_from_logits(
        self,
        rng: Array,
        *,
        logits: Array,
        alpha_t: Array,
        alpha_s: Array,
    ) -> Array:
        unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
        to_unmask, keep_mask = sample_mixture_categorical(
            rng,
            destination_logits=logits,
            stay_prob=1.0 - unmask_prob,
            change_prob=unmask_prob,
            policy=self.categorical_sampling_policy,
        )
        return jnp.where(keep_mask, self.mask_token_id, to_unmask).astype(jnp.int32)

    def ancestral_sample_step(
        self,
        rng: Array,
        i: int,
        timesteps: int,
        state: Array | SamplingState,
        *,
        conditioning: Array | None = None,
    ) -> Array | SamplingState:
        rng_body = jax.random.fold_in(rng, i)
        s, t = self.get_sampling_grid(i, timesteps)

        tokens, log_probs, structured = self._sampling_log_probs(
            state,
            t=t,
            conditioning=conditioning,
        )
        proposal = self._sample_from_logits(
            rng_body,
            logits=log_probs,
            alpha_t=self.noise_schedule.alpha(t),
            alpha_s=self.noise_schedule.alpha(s),
        )
        next_tokens = masked_core.carry_over_unmasked(
            tokens,
            proposal,
            mask_token_id=self.mask_token_id,
        )

        cache_valid = jnp.asarray(self._use_cache()) & (~jnp.any(next_tokens != tokens))
        return self._pack_sampling_state(
            next_tokens,
            log_probs=log_probs,
            cache_valid=cache_valid,
            structured=structured,
        )

    def topp_sample_step(
        self,
        rng: Array,
        i: int,
        timesteps: int,
        state: Array | SamplingState,
        *,
        conditioning: Array | None = None,
        topp: float = 0.98,
    ) -> Array | SamplingState:
        rng_body = jax.random.fold_in(rng, i)
        s, t = self.get_sampling_grid(i, timesteps)

        tokens, log_probs, structured = self._sampling_log_probs(
            state,
            t=t,
            conditioning=conditioning,
        )
        filtered_logits = _topp_mask(
            log_probs,
            topp,
            replace_val=jnp.asarray(-1.0e7, dtype=log_probs.dtype),
        )
        proposal = self._sample_from_logits(
            rng_body,
            logits=filtered_logits,
            alpha_t=self.noise_schedule.alpha(t),
            alpha_s=self.noise_schedule.alpha(s),
        )
        next_tokens = masked_core.carry_over_unmasked(
            tokens,
            proposal,
            mask_token_id=self.mask_token_id,
        )

        cache_valid = jnp.asarray(self._use_cache()) & (~jnp.any(next_tokens != tokens))
        return self._pack_sampling_state(
            next_tokens,
            log_probs=log_probs,
            cache_valid=cache_valid,
            structured=structured,
        )

    def mean_sample_step(
        self,
        rng: Array,
        i: int,
        timesteps: int,
        state: Array | SamplingState,
        *,
        conditioning: Array | None = None,
    ) -> Array | SamplingState:
        tokens, _, _, structured = self._unpack_sampling_state(state)
        rng_body = jax.random.fold_in(rng, i)
        s, t = self.get_sampling_grid(i, timesteps)
        cond = self.get_cond_embedding(conditioning)

        logits, _ = self.predict_x(tokens, t, cond=cond, train=False)
        unmask_prob = (self.noise_schedule.alpha(s) - self.noise_schedule.alpha(t)) / (
            1 - self.noise_schedule.alpha(t)
        )

        rng_body, rng_aux = jax.random.split(rng_body)
        proposal = categorical_sample_from_logits(
            rng_body,
            logits,
            policy=self.categorical_sampling_policy,
        )
        unmask = jax.random.bernoulli(rng_aux, unmask_prob, tokens.shape)
        proposal = jnp.where(unmask, proposal, self.mask_token_id).astype(jnp.int32)
        next_tokens = masked_core.carry_over_unmasked(
            tokens,
            proposal,
            mask_token_id=self.mask_token_id,
        )
        return self._pack_sampling_state(
            next_tokens,
            log_probs=None,
            cache_valid=None,
            structured=structured,
        )

    def sample_step(
        self,
        rng: Array,
        i: int,
        timesteps: int,
        state: Array | SamplingState,
        *,
        conditioning: Array | None = None,
        topp: float | None = None,
    ) -> Array | SamplingState:
        if self.sampler == "ancestral":
            return self.ancestral_sample_step(
                rng,
                i,
                timesteps,
                state,
                conditioning=conditioning,
            )
        if self.sampler == "topp":
            topp_eff = self.topp if topp is None else topp
            return self.topp_sample_step(
                rng,
                i,
                timesteps,
                state,
                conditioning=conditioning,
                topp=topp_eff,
            )
        if self.sampler == "mean":
            return self.mean_sample_step(
                rng,
                i,
                timesteps,
                state,
                conditioning=conditioning,
            )
        raise NotImplementedError(f"Unknown sampler={self.sampler}")

    def decode(
        self,
        state: Array | SamplingState,
        *,
        conditioning: Array | None = None,
    ) -> Array:
        tokens, cached_log_probs, cache_valid, _ = self._unpack_sampling_state(state)
        masked = tokens == self.mask_token_id
        clean_tokens = jnp.where(masked, 0, tokens)

        if self._use_cache() and cache_valid is not None:
            cond = self.get_cond_embedding(conditioning)

            def _use_cached(_):
                return cached_log_probs

            def _recompute(_):
                log_probs, _ = self.predict_clean_log_probs(
                    tokens,
                    jnp.array(0.0, dtype=jnp.float32),
                    cond=cond,
                    train=False,
                )
                return log_probs

            log_probs = jax.lax.cond(cache_valid, _use_cached, _recompute, operand=None)
        else:
            cond = self.get_cond_embedding(conditioning)
            log_probs, _ = self.predict_clean_log_probs(
                tokens,
                jnp.array(0.0, dtype=jnp.float32),
                cond=cond,
                train=False,
            )

        pred = jnp.argmax(log_probs, axis=-1).astype(jnp.int32)
        return jnp.where(masked, pred, clean_tokens).astype(jnp.int32)
