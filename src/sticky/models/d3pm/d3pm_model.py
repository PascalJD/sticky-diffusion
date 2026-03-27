from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from sticky.core.metrics import scale_loss_metrics_to_bits
from sticky.models.architectures import DiscreteClassifier
from sticky.models.discrete_mixture import (
    categorical_sample_from_probs,
    normalize_probs,
)


Array = jnp.ndarray


def _gather_log_probs(log_probs: Array, targets: Array) -> Array:
    return jnp.take_along_axis(
        log_probs,
        jnp.expand_dims(targets, axis=-1),
        axis=-1,
    )[..., 0]


def _make_linear_betas(
    timesteps: int,
    *,
    beta_start: float,
    beta_end: float,
) -> Array:
    return jnp.linspace(
        float(beta_start),
        float(beta_end),
        int(timesteps),
        dtype=jnp.float32,
    )


def _make_cosine_betas(
    timesteps: int,
    *,
    max_beta: float = 0.999,
    cosine_s: float = 0.008,
) -> Array:
    steps = jnp.arange(int(timesteps) + 1, dtype=jnp.float32)
    frac = steps / float(timesteps)
    alpha_bar = jnp.cos(((frac + cosine_s) / (1.0 + cosine_s)) * math.pi / 2.0) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - (alpha_bar[1:] / jnp.maximum(alpha_bar[:-1], 1.0e-12))
    return jnp.clip(betas, 1.0e-6, float(max_beta)).astype(jnp.float32)


def _make_absorbing_linear_betas(timesteps: int) -> Array:
    idx = jnp.arange(1, int(timesteps) + 1, dtype=jnp.float32)
    return (1.0 / (float(timesteps) - idx + 1.0)).astype(jnp.float32)


def _discretized_gaussian_step_matrix(vocab_size: int, beta_t: Array) -> Array:
    beta_t = jnp.maximum(jnp.asarray(beta_t, dtype=jnp.float32), 1.0e-8)
    ids = jnp.arange(int(vocab_size), dtype=jnp.float32)
    diff = ids[:, None] - ids[None, :]
    scale = 4.0 / (((float(vocab_size) - 1.0) ** 2) * beta_t)

    weights = jnp.exp(-scale * diff**2)
    weights = weights * (1.0 - jnp.eye(int(vocab_size), dtype=jnp.float32))

    offsets = jnp.arange(-(int(vocab_size) - 1), int(vocab_size), dtype=jnp.float32)
    denom = jnp.sum(jnp.exp(-scale * offsets**2))
    off_diag = weights / jnp.maximum(denom, 1.0e-20)
    diag = 1.0 - jnp.sum(off_diag, axis=-1)
    matrix = off_diag + jnp.diag(jnp.clip(diag, min=0.0))
    return normalize_probs(matrix)


class D3PM(nn.Module):
    """Discrete D3PM for image-like token tensors.

    The absorbing CIFAR-10 variant intentionally uses an in-vocabulary pixel
    value as the absorbing state by default. We do not introduce an extra token
    here because the shared `DiscreteClassifier` keeps the clean-data head tied
    to the image vocabulary.
    """

    data_shape: tuple[int, ...]
    timesteps: int = 256
    transition_type: str = "gaussian"  # absorb | uniform | gaussian
    transition_beta_schedule: str = "linear"  # linear | cosine | absorbing_linear
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    cosine_s: float = 0.008
    max_beta: float = 0.999
    auxiliary_loss_weight: float = 1.0e-3
    absorbing_state: int = 128  # In-vocabulary absorbing pixel value for absorb mode.

    feature_dim: int = 96
    num_heads: int = 12
    antithetic_time_sampling: bool = True
    n_layers: int = 32
    n_dit_layers: int = 0
    dit_num_heads: int = 12
    dit_hidden_size: int = 768
    ch_mult: Sequence[int] = (3, 4, 4)
    vocab_size: int = 256
    dropout_rate: float = 0.1
    use_attn_dropout: bool = True
    mlp_type: str = "swiglu"
    depth_scaled_init: bool = True
    cond_type: str = "adaln_zero"
    outside_embed: bool = False
    sequence_backbone: str = "auto"
    image_backbone: str = "adm_unet5d"
    adm_num_res_blocks: int = 4
    adm_attention_resolutions: Sequence[int] = (2, 4)
    adm_num_heads: int = 4
    adm_num_head_channels: int = 64
    adm_num_heads_upsample: int = -1
    adm_conv_resample: bool = True
    adm_use_scale_shift_norm: bool = True
    adm_resblock_updown: bool = False
    adm_use_conv_skip: bool = False
    adm_use_new_attention_order: bool = False
    time_features: str = "t"
    classes: int = -1
    sampler: str = "ancestral"
    sampling_grid: str = "uniform"  # Exact trained chain only; non-uniform grids are rejected.
    categorical_sampling_policy: str = "legacy_low"
    model_sharding: bool = False

    def setup(self):
        transition_type = str(self.transition_type).lower()
        if transition_type not in {"absorb", "uniform", "gaussian"}:
            raise ValueError(
                f"Unknown transition_type={self.transition_type!r}. "
                "Expected one of: absorb, uniform, gaussian."
            )
        if str(self.sampling_grid).lower() != "uniform":
            raise ValueError(
                "D3PM currently only supports sampling_grid='uniform' because "
                "reverse sampling follows the exact trained discrete chain."
            )

        self._transition_type = transition_type
        self._eye = jnp.eye(int(self.vocab_size), dtype=jnp.float32)
        self._uniform_matrix = jnp.full(
            (int(self.vocab_size), int(self.vocab_size)),
            1.0 / float(self.vocab_size),
            dtype=jnp.float32,
        )
        absorbing_row = jax.nn.one_hot(
            int(self.absorbing_state),
            int(self.vocab_size),
            dtype=jnp.float32,
        )
        # This is an approximation to absorbing-mask diffusion for image
        # vocabularies: the absorb variant maps into a fixed in-vocabulary CIFAR
        # pixel value rather than reserving an extra dedicated token.
        self._absorbing_matrix = jnp.broadcast_to(
            absorbing_row,
            (int(self.vocab_size), int(self.vocab_size)),
        )

        self.betas = self._build_betas()
        self.alpha_bars = jnp.concatenate(
            [
                jnp.ones((1,), dtype=jnp.float32),
                jnp.cumprod(1.0 - self.betas, axis=0),
            ],
            axis=0,
        )

        if self._transition_type == "gaussian":
            step_mats = [self._eye]
            cum_mats = [self._eye]
            current = self._eye
            for idx in range(int(self.timesteps)):
                step = _discretized_gaussian_step_matrix(
                    int(self.vocab_size),
                    self.betas[idx],
                )
                step_mats.append(step)
                current = current @ step
                cum_mats.append(current)
            self.gaussian_step_matrices = jnp.stack(step_mats, axis=0)
            self.gaussian_cum_matrices = jnp.stack(cum_mats, axis=0)
        else:
            self.gaussian_step_matrices = None
            self.gaussian_cum_matrices = None

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

    def _build_betas(self) -> Array:
        schedule = str(self.transition_beta_schedule).lower()
        if schedule == "linear":
            return _make_linear_betas(
                int(self.timesteps),
                beta_start=float(self.beta_start),
                beta_end=float(self.beta_end),
            )
        if schedule == "cosine":
            return _make_cosine_betas(
                int(self.timesteps),
                max_beta=float(self.max_beta),
                cosine_s=float(self.cosine_s),
            )
        if schedule in {"absorbing_linear", "mask_linear"}:
            return _make_absorbing_linear_betas(int(self.timesteps))
        raise ValueError(
            f"Unknown transition_beta_schedule={self.transition_beta_schedule!r}. "
            "Expected one of: linear, cosine, absorbing_linear."
        )

    def _broadcast_t_index(self, t_index: Array | int, batch_size: int) -> Array:
        t_index = jnp.asarray(t_index, dtype=jnp.int32)
        if t_index.ndim == 0:
            t_index = jnp.full((batch_size,), t_index, dtype=jnp.int32)
        return t_index

    def _time_features_from_index(self, t_index: Array) -> Array:
        return jnp.asarray(t_index, dtype=jnp.float32) / float(self.timesteps)

    def _q_one_step_matrix(self, t_index: Array) -> Array:
        if self._transition_type == "gaussian":
            return self.gaussian_step_matrices[jnp.asarray(t_index, dtype=jnp.int32)]

        beta_t = self.betas[jnp.asarray(t_index, dtype=jnp.int32) - 1]
        if self._transition_type == "uniform":
            return (1.0 - beta_t) * self._eye + beta_t * self._uniform_matrix
        return (1.0 - beta_t) * self._eye + beta_t * self._absorbing_matrix

    def _q_cum_matrix(self, t_index: Array) -> Array:
        if self._transition_type == "gaussian":
            return self.gaussian_cum_matrices[jnp.asarray(t_index, dtype=jnp.int32)]

        alpha_t = self.alpha_bars[jnp.asarray(t_index, dtype=jnp.int32)]
        if self._transition_type == "uniform":
            return alpha_t * self._eye + (1.0 - alpha_t) * self._uniform_matrix
        return alpha_t * self._eye + (1.0 - alpha_t) * self._absorbing_matrix

    def get_cond_embedding(self, conditioning: Array | None) -> Array | None:
        if (self.classes > 0) and (conditioning is not None):
            return self.cond_embeddings(conditioning)
        return None

    def predict_x(
        self,
        xt: Array,
        t: Array,
        *,
        cond: Array | None = None,
        train: bool = False,
    ) -> tuple[Array, dict[str, Any]]:
        t_in = None if self.time_features == "none" else t
        return self.classifier(xt, t=t_in, cond=cond, train=train)

    def q_marginal_probs(self, x0: Array, t_index: Array | int) -> Array:
        t_index = self._broadcast_t_index(t_index, x0.shape[0])
        cum_mats = jax.vmap(self._q_cum_matrix)(t_index)
        return jax.vmap(lambda mat, x: jnp.take(mat, x, axis=0))(cum_mats, x0)

    def q_sample(self, rng: Array, x0: Array, t_index: Array | int) -> Array:
        probs = self.q_marginal_probs(x0, t_index)
        rngs = jax.random.split(rng, x0.shape[0])
        return jax.vmap(
            lambda key, probs_b: categorical_sample_from_probs(
                key,
                probs_b,
                policy=self.categorical_sampling_policy,
            )
        )(rngs, probs).astype(jnp.int32)

    def q_posterior_probs(self, x0: Array, xt: Array, t_index: Array | int) -> Array:
        t_index = self._broadcast_t_index(t_index, x0.shape[0])
        prev_mats = jax.vmap(self._q_cum_matrix)(jnp.maximum(t_index - 1, 0))
        curr_mats = jax.vmap(self._q_cum_matrix)(t_index)
        step_mats = jax.vmap(self._q_one_step_matrix)(t_index)

        def _per_example(prev_mat, curr_mat, step_mat, x0_b, xt_b):
            prev_rows = jnp.take(prev_mat, x0_b, axis=0)
            step_cols = jnp.take(step_mat.T, xt_b, axis=0)
            curr_rows = jnp.take(curr_mat, x0_b, axis=0)
            denom = jnp.take_along_axis(curr_rows, xt_b[..., None], axis=-1)[..., 0]
            probs = prev_rows * step_cols
            probs = jnp.where(
                denom[..., None] > 0.0,
                probs / denom[..., None],
                0.0,
            )
            return normalize_probs(probs)

        return jax.vmap(_per_example)(prev_mats, curr_mats, step_mats, x0, xt)

    def q_posterior_sample(self, rng: Array, x0: Array, xt: Array, t_index: Array | int) -> Array:
        probs = self.q_posterior_probs(x0, xt, t_index)
        rngs = jax.random.split(rng, x0.shape[0])
        return jax.vmap(
            lambda key, probs_b: categorical_sample_from_probs(
                key,
                probs_b,
                policy=self.categorical_sampling_policy,
            )
        )(rngs, probs).astype(jnp.int32)

    def model_posterior_probs(self, clean_logits: Array, xt: Array, t_index: Array | int) -> Array:
        t_index = self._broadcast_t_index(t_index, clean_logits.shape[0])
        clean_probs = jax.nn.softmax(clean_logits, axis=-1)
        curr_mats = jax.vmap(self._q_cum_matrix)(t_index)
        prev_mats = jax.vmap(self._q_cum_matrix)(jnp.maximum(t_index - 1, 0))
        step_mats = jax.vmap(self._q_one_step_matrix)(t_index)

        def _per_example(curr_mat, prev_mat, step_mat, clean_probs_b, xt_b):
            curr_cols = jnp.take(curr_mat.T, xt_b, axis=0)
            clean_weights = jnp.where(
                curr_cols > 0.0,
                clean_probs_b / curr_cols,
                0.0,
            )
            prev_probs = jnp.einsum("...v,vk->...k", clean_weights, prev_mat)
            step_cols = jnp.take(step_mat.T, xt_b, axis=0)
            probs = prev_probs * step_cols
            return normalize_probs(probs)

        return jax.vmap(_per_example)(curr_mats, prev_mats, step_mats, clean_probs, xt)

    def terminal_prior_probs(self) -> Array:
        if self._transition_type == "absorb":
            return jax.nn.one_hot(
                int(self.absorbing_state),
                int(self.vocab_size),
                dtype=jnp.float32,
            )
        return jnp.full(
            (int(self.vocab_size),),
            1.0 / float(self.vocab_size),
            dtype=jnp.float32,
        )

    def terminal_prior_kl(self, x0: Array) -> Array:
        if self._transition_type == "absorb":
            return jnp.zeros((x0.shape[0],), dtype=jnp.float32)
        if self._transition_type == "uniform":
            # q(x_T | x_0) = alpha_bar_T I + (1 - alpha_bar_T) U, so the KL to a
            # uniform prior is permutation-symmetric and constant across x_0.
            return jnp.zeros((x0.shape[0],), dtype=jnp.float32)

        terminal_mat = self._q_cum_matrix(jnp.asarray(self.timesteps, dtype=jnp.int32))
        q_terminal = jnp.take(terminal_mat, x0, axis=0)
        prior_probs = self.terminal_prior_probs()
        log_q = jnp.log(jnp.clip(q_terminal, a_min=1.0e-30))
        log_prior = jnp.log(prior_probs)
        kl_per_token = jnp.sum(q_terminal * (log_q - log_prior), axis=-1)
        return jnp.sum(kl_per_token, axis=tuple(range(1, x0.ndim)))

    def prior_sample(self, batch_size: int) -> Array:
        shape = (int(batch_size),) + tuple(self.data_shape)
        if self._transition_type == "absorb":
            return jnp.full(shape, int(self.absorbing_state), dtype=jnp.int32)
        return jax.random.randint(
            self.make_rng("sample"),
            shape=shape,
            minval=0,
            maxval=int(self.vocab_size),
            dtype=jnp.int32,
        )

    def decode(self, state: Array, *, conditioning: Array | None = None) -> Array:
        del conditioning
        return jnp.clip(state, 0, self.vocab_size - 1).astype(jnp.int32)

    def latent_loss(self, x0: Array | None = None) -> Array:
        if x0 is None:
            return jnp.array(0.0, dtype=jnp.float32)
        return self.terminal_prior_kl(x0).mean()

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

        loss_prior = self.latent_loss(x)

        rng_t, rng_xt, rng_prev = jax.random.split(self.make_rng("sample"), 3)
        if self.antithetic_time_sampling:
            t0 = jax.random.uniform(rng_t)
            frac = jnp.mod(t0 + jnp.arange(0.0, 1.0, step=1.0 / batch_size), 1.0)
            t_index = jnp.floor(frac * float(self.timesteps)).astype(jnp.int32) + 1
        else:
            t_index = jax.random.randint(
                rng_t,
                shape=(batch_size,),
                minval=1,
                maxval=int(self.timesteps) + 1,
                dtype=jnp.int32,
            )

        xt = self.q_sample(rng_xt, x, t_index)
        clean_logits, _ = self.predict_x(
            xt,
            self._time_features_from_index(t_index),
            cond=cond_emb,
            train=train,
        )
        clean_log_probs = jax.nn.log_softmax(clean_logits, axis=-1)
        aux_nll = -jnp.sum(
            _gather_log_probs(clean_log_probs, x),
            axis=tuple(range(1, x.ndim)),
        )

        x_prev = self.q_posterior_sample(rng_prev, x, xt, t_index)
        model_prev_probs = self.model_posterior_probs(clean_logits, xt, t_index)
        model_prev_log_probs = jnp.log(jnp.clip(model_prev_probs, a_min=1.0e-30))
        vb_nll = -jnp.sum(
            _gather_log_probs(model_prev_log_probs, x_prev),
            axis=tuple(range(1, x.ndim)),
        )

        loss_vb = float(self.timesteps) * vb_nll.mean()
        loss_aux = float(self.auxiliary_loss_weight) * aux_nll.mean()
        loss = loss_vb + loss_aux + loss_prior

        stats = {
            "loss": loss,
            "loss_vb": loss_vb,
            "loss_aux": loss_aux,
            "loss_prior": loss_prior,
        }
        return scale_loss_metrics_to_bits(stats, self.data_shape)

    def sample_step(
        self,
        rng: Array,
        i: int,
        timesteps: int,
        state: Array,
        *,
        conditioning: Array | None = None,
    ) -> Array:
        if self.sampler != "ancestral":
            raise NotImplementedError(f"Unknown sampler={self.sampler!r} for D3PM.")

        batch_size = state.shape[0]
        t_scalar = jnp.asarray(timesteps, dtype=jnp.int32) - jnp.asarray(i, dtype=jnp.int32)
        t_index = jnp.full(
            (batch_size,),
            t_scalar,
            dtype=jnp.int32,
        )
        cond_emb = self.get_cond_embedding(conditioning)
        clean_logits, _ = self.predict_x(
            state,
            self._time_features_from_index(t_index),
            cond=cond_emb,
            train=False,
        )
        prev_probs = self.model_posterior_probs(clean_logits, state, t_index)
        rngs = jax.random.split(rng, batch_size)
        return jax.vmap(
            lambda key, probs_b: categorical_sample_from_probs(
                key,
                probs_b,
                policy=self.categorical_sampling_policy,
            )
        )(rngs, prev_probs).astype(jnp.int32)
