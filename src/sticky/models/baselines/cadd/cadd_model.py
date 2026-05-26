from __future__ import annotations

import math
from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from sticky.core.metrics import scale_bpd_components
from sticky.models.backbones.factory import (
    build_image_backbone,
    build_sequence_backbone,
)
from sticky.models.common.discrete_mixture import (
    categorical_sample_from_probs,
    sample_mixture_categorical,
)
from sticky.models.backbones.networks.conditioning import CondEmbedding
from sticky.models.common import masked_discrete_core as masked_core


Array = jnp.ndarray


class ClippedSchedule(nn.Module):
    """Simple [0,1] -> [eps, 1-eps] schedule.

    Used for both:
      - discrete masking keep-probability alpha(t)
      - continuous latent signal coefficient \bar{gamma}(t)

    CADD (paper Appendix B.1) uses a linear schedule for both parts.
    """

    schedule_fn_type: str = "linear"  # linear | cosine | poly{p}
    eps: float = 1e-4

    def _base(self, t: Array) -> Array:
        return masked_core.clipped_schedule_alpha(
            t,
            schedule_fn_type=self.schedule_fn_type,
            eps=0.0,
        )

    def __call__(self, t: Array) -> Array:
        return masked_core.clipped_schedule_alpha(
            t,
            schedule_fn_type=self.schedule_fn_type,
            eps=self.eps,
        )


class TokenEmbedding(nn.Module):
    """Lightweight embedding table with an explicit `table` parameter."""

    vocab_size: int
    dim: int
    init_std: float = 1.0

    @nn.compact
    def __call__(self, ids: Array) -> Array:
        table = self.param(
            "table",
            nn.initializers.normal(stddev=float(self.init_std)),
            (int(self.vocab_size), int(self.dim)),
            jnp.float32,
        )
        return jnp.take(table, ids, axis=0)

    def table_float(self) -> Array:
        return self.get_variable("params", "table").astype(jnp.float32)


class CADD(nn.Module):
    """Continuously Augmented Discrete Diffusion (CADD).

    This implementation follows the paper's core recipe:
      - Discrete absorbing masking process with alpha(t) keep-probability.
      - Continuous latent process in embedding space for masked tokens.
      - Cross-entropy training on masked positions (Eq. 4.17).

    Notes:
      * This code focuses on the *image/token* setting used in Sticky.
      * The continuous latent can follow either the paper's main Gaussian
        diffusion derivation or the Appendix B.1 flow-matching path.
      * The sampling code supports the paper's cosine-decay temperature schedule.
      * Sampling can average K continuous latent hints per reverse step.
    """

    # Data / discretization.
    data_shape: tuple[int, ...]
    vocab_size: int = 256  # base vocabulary (excluding the mask token)

    # Training-time t sampling.
    cont_time: bool = True
    timesteps: int = 512
    antithetic_time_sampling: bool = False

    # Schedules.
    discrete_schedule_type: str = "linear"  # paper: linear
    continuous_schedule_type: str = "linear"
    continuous_latent_type: str = "gaussian"  # gaussian | flow_matching
    schedule_eps: float = 1e-4

    # Embedding/backbone.
    feature_dim: int = 128
    num_heads: int = 12
    n_layers: int = 32
    n_dit_layers: int = 0
    dit_num_heads: int = 12
    dit_hidden_size: int = 768
    ch_mult: Sequence[int] = (1, 2, 2, 2)

    dropout_rate: float = 0.0
    use_attn_dropout: bool = True
    mlp_type: str = "swiglu"
    depth_scaled_init: bool = False
    cond_type: str = "adaln"
    model_sharding: bool = False
    sequence_backbone: str = "auto"
    image_backbone: str = "auto"

    # ADM-style image UNet knobs (used when image_backbone=adm_unet5d).
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

    # Conditioning / time embedding.
    time_features: str = "t"  # 't' or 'none'
    classes: int = -1  # set >0 for class-conditional

    # Sampling options.
    sampling_grid: str = "cosine"  # uniform | cosine
    temperature_schedule: str = "cosine_decay"  # cosine_decay | constant
    tau_max: float = 2.5  # paper: 2.5
    logit_temperature: float = 1.0  # used when temperature_schedule=constant
    z0_estimator: str = "hard"  # hard | soft
    K: int = 1  # number of continuous latent hints averaged per reverse step
    force_decode_at_end: bool = True
    categorical_sampling_policy: str = "legacy_low"  # legacy_low | jax_high | exact

    # Optional sampling-time refinement (in the spirit of Gat et al., 2024).
    # The CADD paper reports applying a refinement for image generation, but does
    # not fully specify hyperparameters; we expose a simple remasking refinement.
    remask_refine_enabled: bool = False
    remask_refine_steps: int = 1
    remask_refine_frac: float = 0.0
    remask_refine_metric: str = "entropy"  # entropy | neg_entropy
    remask_refine_sample_mode: str = "sample"  # sample | argmax

    def setup(self):
        if int(self.K) < 1:
            raise ValueError(f"CADD requires K>=1, got K={self.K}.")

        # Schedules.
        self.alpha = ClippedSchedule(
            schedule_fn_type=self.discrete_schedule_type,
            eps=float(self.schedule_eps),
        )
        self.gamma_bar = ClippedSchedule(
            schedule_fn_type=self.continuous_schedule_type,
            eps=float(self.schedule_eps),
        )

        # Token embeddings: include an extra entry for the absorbing mask token.
        self.token_embed = TokenEmbedding(
            vocab_size=int(self.vocab_size) + 1,
            dim=int(self.feature_dim),
            init_std=1.0,
        )

        if self.classes > 0:
            self.cond_embeddings = nn.Embed(int(self.classes), int(self.feature_dim))
        # Pre-register time/conditioning embedder so it can be used from
        # non-compact helper methods (e.g., sampling/eval paths).
        self.time_cond_embed = CondEmbedding(int(self.feature_dim))

        self._seq_backbone = build_sequence_backbone(
            name=self.sequence_backbone,
            feature_dim=int(self.feature_dim),
            num_heads=int(self.num_heads),
            n_layers=int(self.n_layers),
            vocab_size=int(self.vocab_size),
            dropout_rate=float(self.dropout_rate),
            use_attn_dropout=bool(self.use_attn_dropout),
            mlp_type=str(self.mlp_type),
            depth_scaled_init=bool(self.depth_scaled_init),
            cond_type=str(self.cond_type),
            model_sharding=bool(self.model_sharding),
            embed_input=False,  # we provide embeddings
            n_embed_classes=int(self.vocab_size) + 1,
        )
        self._img_backbone = build_image_backbone(
            name=self.image_backbone,
            feature_dim=int(self.feature_dim),
            n_layers=int(self.n_layers),
            n_dit_layers=int(self.n_dit_layers),
            dit_num_heads=int(self.dit_num_heads),
            dit_hidden_size=int(self.dit_hidden_size),
            ch_mult=tuple(int(x) for x in self.ch_mult),
            vocab_size=int(self.vocab_size),
            dropout_rate=float(self.dropout_rate),
            adm_num_res_blocks=int(self.adm_num_res_blocks),
            adm_attention_resolutions=tuple(int(v) for v in self.adm_attention_resolutions),
            adm_num_heads=int(self.adm_num_heads),
            adm_num_head_channels=int(self.adm_num_head_channels),
            adm_num_heads_upsample=int(self.adm_num_heads_upsample),
            adm_conv_resample=bool(self.adm_conv_resample),
            adm_use_scale_shift_norm=bool(self.adm_use_scale_shift_norm),
            adm_resblock_updown=bool(self.adm_resblock_updown),
            adm_use_conv_skip=bool(self.adm_use_conv_skip),
            adm_use_new_attention_order=bool(self.adm_use_new_attention_order),
        )

    @property
    def mask_token_id(self) -> int:
        return masked_core.mask_token_id(self.vocab_size)

    # ------------------------- Conditioning helpers -------------------------

    def get_cond_embedding(self, conditioning: Array | None) -> Array | None:
        if (self.classes > 0) and (conditioning is not None):
            return self.cond_embeddings(conditioning)
        return None

    def _make_time_cond(
        self,
        t: Array | None,
        *,
        cond: Array | None,
        batch_size: int,
    ) -> Array | None:
        """Combine (optional) time with (optional) conditioning embedding."""
        if t is None:
            return cond
        assert jnp.isscalar(t) or t.ndim == 0 or t.ndim == 1
        if jnp.isscalar(t) or t.ndim == 0:
            t = t * jnp.ones((int(batch_size),), dtype=jnp.asarray(t).dtype)
        # If cond is None, CondEmbedding will just embed time.
        return self.time_cond_embed(t * 1000.0, cond=cond)

    # --------------------------- Core network call --------------------------

    def predict_logits(self, z_tilde: Array, t: Array | None, *, cond: Array | None, train: bool) -> Array:
        """Predict logits over the base vocabulary given fused embeddings."""
        if self.time_features == "none":
            t = None
        use_adm_image_time_path = (z_tilde.ndim == 5) and (str(self.image_backbone).lower() == "adm_unet5d")
        time_cond = None
        if not use_adm_image_time_path:
            time_cond = self._make_time_cond(t, cond=cond, batch_size=int(z_tilde.shape[0]))

        if z_tilde.ndim == 3:
            return self._seq_backbone(z_tilde, cond=time_cond, train=train)
        if z_tilde.ndim == 5:
            return self._img_backbone(
                z_tilde,
                cond=cond if use_adm_image_time_path else time_cond,
                timesteps=None if t is None else jnp.asarray(t) * 1000.0,
                train=train,
            )
        raise NotImplementedError(
            f"CADD expects embedded inputs with ndim in {{3,5}}, got {z_tilde.ndim} ({z_tilde.shape})."
        )

    # ------------------------ Forward / training loss -----------------------

    def forward_sample_discrete(self, x0: Array, t: Array) -> Array:
        """Sample x_t from the absorbing masking process."""
        keep_prob = self.alpha(masked_core.reverse_broadcast(t, x0.ndim))
        return masked_core.sample_masked_discrete_forward(
            self.make_rng("sample"),
            x0,
            keep_prob=keep_prob,
            mask_token_id=self.mask_token_id,
        )

    def _continuous_gamma_bar(self, t: Array, *, target_ndim: int) -> Array:
        gbar = self.gamma_bar(masked_core.reverse_broadcast(jnp.asarray(t), target_ndim))
        gbar = jnp.clip(gbar, a_min=0.0, a_max=1.0)
        return gbar[..., None]

    def _sample_masked_continuous_latent(self, *, rng: Array, z0: Array, t: Array) -> Array:
        """Sample the masked-position continuous marginal q(z_t | x_t=m, x_0)."""
        gbar = self._continuous_gamma_bar(t, target_ndim=z0.ndim - 1)
        noise = jax.random.normal(rng, z0.shape)

        latent_type = str(self.continuous_latent_type)
        if latent_type == "gaussian":
            return jnp.sqrt(gbar) * z0 + jnp.sqrt(jnp.maximum(1.0 - gbar, 0.0)) * noise
        if latent_type == "flow_matching":
            return gbar * z0 + (1.0 - gbar) * noise
        raise NotImplementedError(
            f"Unknown continuous_latent_type={self.continuous_latent_type!r}"
        )

    def _reverse_masked_continuous_latent(
        self,
        *,
        rng: Array,
        z_t: Array,
        z0_hat: Array,
        s: Array,
        t: Array,
    ) -> Array:
        """Sample or transport the masked latent from time `t` to earlier time `s`."""
        gbar_t = self._continuous_gamma_bar(t, target_ndim=z_t.ndim - 1)
        gbar_s = self._continuous_gamma_bar(s, target_ndim=z_t.ndim - 1)

        latent_type = str(self.continuous_latent_type)
        if latent_type == "gaussian":
            denom = jnp.maximum(1.0 - gbar_t, 1e-6)
            gamma_step = jnp.clip(gbar_t / jnp.maximum(gbar_s, 1e-6), a_min=0.0, a_max=1.0)
            mu = (
                jnp.sqrt(gbar_s) * (1.0 - gamma_step) / denom * z0_hat
                + jnp.sqrt(gamma_step) * (1.0 - gbar_s) / denom * z_t
            )
            beta_tilde = (1.0 - gbar_s) * (1.0 - gamma_step) / denom
            noise = jax.random.normal(rng, z_t.shape)
            return mu + jnp.sqrt(jnp.maximum(beta_tilde, 0.0)) * noise

        if latent_type == "flow_matching":
            denom = jnp.maximum(1.0 - gbar_t, 1e-6)
            coef_t = (1.0 - gbar_s) / denom
            coef_0 = gbar_s - coef_t * gbar_t
            return coef_0 * z0_hat + coef_t * z_t

        raise NotImplementedError(
            f"Unknown continuous_latent_type={self.continuous_latent_type!r}"
        )

    def _make_continuous_latent(self, *, x0: Array, x_t: Array, t: Array) -> Array:
        """Construct z_t for masked positions from the selected latent process."""
        z0 = self.token_embed(x0)
        mask = (x_t == self.mask_token_id).astype(jnp.float32)[..., None]
        z_t = self._sample_masked_continuous_latent(
            rng=self.make_rng("sample"),
            z0=z0,
            t=t,
        )
        return z_t * mask  # 0 for unmasked positions

    def _cross_entropy_on_masked(
        self,
        *,
        logits: Array,
        x0: Array,
        x_t: Array,
        t: Array,
    ) -> tuple[Array, dict[str, Array]]:
        """MDLM continuous-time CE bound on masked positions (Eq. 4.14 / 4.17)."""
        mask = masked_core.masked_positions(x_t, mask_token_id=self.mask_token_id)
        masked_neg_ce = masked_core.masked_logprob_sums(logits, x0, mask=mask)
        # alpha'(t) / (1 - alpha(t)): non-positive, multiplied by non-positive
        # log-prob sum to yield a positive loss. Requires schedule_eps > 0.
        weight = masked_core.masked_dgamma_times_alpha(
            t,
            schedule_fn_type=self.discrete_schedule_type,
            eps=self.schedule_eps,
        )
        loss = jnp.mean(weight * masked_neg_ce)

        metrics = {
            "bpd_ce": loss,
            "mask_frac": jnp.mean(mask),
        }
        return loss, metrics

    @nn.compact
    def __call__(self, x: Array, *, cond: Array | None = None, train: bool = False) -> dict[str, Array]:
        """Training forward pass returning a metrics dict (must include 'loss')."""
        bs = int(x.shape[0])
        cond_emb = self.get_cond_embedding(cond)

        rng1 = self.make_rng("sample")
        if self.antithetic_time_sampling:
            t0 = jax.random.uniform(rng1)
            t = jnp.mod(t0 + jnp.arange(0.0, 1.0, step=1.0 / bs), 1.0)
        else:
            t = jax.random.uniform(rng1, shape=(bs,))

        if not self.cont_time:
            t = (jnp.floor(t * self.timesteps) + 1.0) / float(self.timesteps)

        # Forward noising.
        x_t = self.forward_sample_discrete(x, t)
        z_hint = self._make_continuous_latent(x0=x, x_t=x_t, t=t)
        z_disc = self.token_embed(x_t)
        z_tilde = z_disc + z_hint

        # Model prediction.
        logits = self.predict_logits(z_tilde, t, cond=cond_emb, train=train)

        loss, metrics = self._cross_entropy_on_masked(logits=logits, x0=x, x_t=x_t, t=t)
        metrics["bpd"] = loss
        metrics["t_mean"] = jnp.mean(t)

        # Match MD4-style logging: convert losses to bits-per-token.
        return scale_bpd_components(metrics, self.data_shape)

    # ------------------------------- Sampling -------------------------------

    def get_sampling_grid(self, i: int, timesteps: int) -> tuple[Array, Array]:
        """Return (s, t) with t decreasing from ~1 to ~0."""
        return masked_core.make_sampling_time_pair(
            i,
            timesteps,
            sampling_grid=self.sampling_grid,
        )

    def temperature(self, t: Array) -> Array:
        if self.temperature_schedule == "constant":
            return jnp.asarray(self.logit_temperature)
        if self.temperature_schedule == "cosine_decay":
            # tau(t=1)=tau_max, tau(t=0)=1.
            return 1.0 + (float(self.tau_max) - 1.0) * jnp.cos(
                math.pi / 2.0 * (1.0 - t)
            )
        raise NotImplementedError(
            f"Unknown temperature_schedule={self.temperature_schedule!r}"
        )

    def prior_sample(self, batch_size: int) -> tuple[Array, Array]:
        """Sample the CADD prior: x_T=mask, z_T~N(0,I)."""
        x = masked_core.make_masked_token_prior(
            batch_size=batch_size,
            data_shape=self.data_shape,
            vocab_size=self.vocab_size,
        )
        latent_shape = (batch_size,) + tuple(self.data_shape) + (int(self.feature_dim),)
        if int(self.K) > 1:
            latent_shape = (int(self.K),) + latent_shape
        z = jax.random.normal(
            self.make_rng("sample"),
            latent_shape,
        )
        return x, z

    def _token_probs_per_latent(
        self,
        *,
        x_t: Array,
        z_samples: Array,
        t: Array,
        cond_emb: Array | None,
    ) -> Array:
        """Return p_theta(x0 | x_t, z_t^(k)) for each latent hint sample."""
        if z_samples.ndim == x_t.ndim + 1:
            z_samples = z_samples[None, ...]
        elif z_samples.ndim != x_t.ndim + 2:
            raise ValueError(
                "CADD z_samples must have either a single latent shape "
                f"{x_t.ndim + 1}D or a leading-K latent batch, got {z_samples.shape}."
            )

        z_disc = self.token_embed(x_t)
        mask = (x_t == self.mask_token_id).astype(jnp.float32)[..., None]
        t_arr = jnp.asarray(t)
        temp = self.temperature(t_arr)

        def _single_probs(z_sample: Array) -> Array:
            z_tilde = z_disc + z_sample * mask
            logits = self.predict_logits(z_tilde, t_arr, cond=cond_emb, train=False)
            return jax.nn.softmax(logits / temp, axis=-1)

        return jax.vmap(_single_probs)(z_samples)

    def _token_probs_from_latents(
        self,
        *,
        x_t: Array,
        z_samples: Array,
        t: Array,
        cond_emb: Array | None,
    ) -> Array:
        """Average token probabilities across one or more latent hints."""
        return jnp.mean(
            self._token_probs_per_latent(
                x_t=x_t,
                z_samples=z_samples,
                t=t,
                cond_emb=cond_emb,
            ),
            axis=0,
        )

    def _sampling_probs(
        self,
        *,
        x_t: Array,
        z_t: Array,
        t: Array,
        cond_emb: Array | None,
    ) -> Array:
        """Approximate E[p_theta(x0 | x_t, z_t)] with K latent hints."""
        return self._token_probs_from_latents(
            x_t=x_t,
            z_samples=z_t,
            t=t,
            cond_emb=cond_emb,
        )

    def _estimate_z0(self, *, probs: Array) -> Array:
        """Estimate z0 embedding from token probabilities."""
        if self.z0_estimator == "hard":
            x0_hat = jnp.argmax(probs, axis=-1).astype(jnp.int32)
            return self.token_embed(x0_hat)
        if self.z0_estimator == "soft":
            # Expected embedding under the predicted token distribution.
            table = self.token_embed.table_float()[: self.vocab_size]  # exclude mask
            return jnp.einsum("...k,kd->...d", probs, table)
        raise NotImplementedError(f"Unknown z0_estimator={self.z0_estimator!r}")

    def _remask_refine_step(
        self,
        rng: Array,
        *,
        x: Array,
        z: Array,
        t: Array,
        cond_emb: Array | None,
    ) -> tuple[Array, Array]:
        """A simple remask/resample refinement step at fixed time `t`.

        This is a lightweight, JIT-friendly approximation inspired by
        remasking-based refinements for discrete diffusion/flow models.
        """
        remask_frac = float(self.remask_refine_frac)
        if remask_frac <= 0.0:
            return x, z

        # 1) Score current tokens by uncertainty (entropy).
        is_mask = x == self.mask_token_id
        rng_eps, rng_tok = jax.random.split(rng, 2)
        probs = self._sampling_probs(
            x_t=x,
            z_t=z,
            t=jnp.asarray(t),
            cond_emb=cond_emb,
        )

        eps = 1e-20
        ent = -jnp.sum(probs * jnp.log(jnp.clip(probs, eps, 1.0)), axis=-1)
        if self.remask_refine_metric == "neg_entropy":
            score = -ent
        elif self.remask_refine_metric == "entropy":
            score = ent
        else:
            raise NotImplementedError(
                f"Unknown remask_refine_metric={self.remask_refine_metric!r}"
            )

        # Never remask positions that are already masked.
        score = jnp.where(is_mask, -jnp.inf, score)

        b = int(x.shape[0])
        n_tokens = int(math.prod(self.data_shape))
        k = int(remask_frac * n_tokens)
        if k <= 0:
            return x, z
        k = min(k, n_tokens)

        score_flat = jnp.reshape(score, (b, n_tokens))
        _, top_idx = jax.lax.top_k(score_flat, k)

        remask_flat = jnp.zeros((b, n_tokens), dtype=jnp.bool_)
        remask_flat = remask_flat.at[
            jnp.arange(b)[:, None], top_idx
        ].set(True)
        remask = jnp.reshape(remask_flat, score.shape)

        # 2) Remask selected tokens and re-noise their continuous latents.
        x_masked = jnp.where(remask, self.mask_token_id, x).astype(jnp.int32)

        # Re-noise the remasked positions using the configured continuous path.
        z0 = self.token_embed(x)
        remask_latent = remask[..., None]
        if z.ndim == x.ndim + 2:
            z0 = jnp.broadcast_to(z0[None, ...], z.shape)
            remask_latent = remask[None, ..., None]
        z_noisy = self._sample_masked_continuous_latent(
            rng=rng_eps,
            z0=z0,
            t=jnp.asarray(t),
        )
        z_masked = jnp.where(remask_latent, z_noisy, z)

        # 3) Resample remasked positions (unmask immediately).
        probs2 = self._sampling_probs(
            x_t=x_masked,
            z_t=z_masked,
            t=jnp.asarray(t),
            cond_emb=cond_emb,
        )

        if self.remask_refine_sample_mode == "argmax":
            sampled = jnp.argmax(probs2, axis=-1).astype(jnp.int32)
        elif self.remask_refine_sample_mode == "sample":
            sampled = categorical_sample_from_probs(
                rng_tok,
                probs2,
                policy=self.categorical_sampling_policy,
            )
        else:
            raise NotImplementedError(
                f"Unknown remask_refine_sample_mode={self.remask_refine_sample_mode!r}"
            )

        x_new = jnp.where(remask, sampled, x).astype(jnp.int32)
        z_new_tokens = self.token_embed(x_new)
        if z.ndim == x.ndim + 2:
            z_new_tokens = jnp.broadcast_to(z_new_tokens[None, ...], z.shape)
        z_new = jnp.where(remask_latent, z_new_tokens, z_masked)
        return x_new, z_new

    def sample_step(
        self,
        rng: Array,
        i: int,
        timesteps: int,
        state: tuple[Array, Array],
        *,
        conditioning: Array | None = None,
    ) -> tuple[Array, Array]:
        """One reverse step (t -> s) for the joint (x_t, z_t) state."""
        x_t, z_t = state

        rng_body = jax.random.fold_in(rng, i)
        rng_body, rng_token, rng_latent = jax.random.split(rng_body, 3)

        s, t = self.get_sampling_grid(i, timesteps)
        cond_emb = self.get_cond_embedding(conditioning)

        # Discrete flip/keep probabilities for masked positions.
        alpha_t = self.alpha(jnp.asarray(t))
        alpha_s = self.alpha(jnp.asarray(s))

        rho_flip = (alpha_s - alpha_t) / (1.0 - alpha_t)
        rho_flip = jnp.clip(rho_flip, a_min=0.0, a_max=1.0)

        probs = self._sampling_probs(
            x_t=x_t,
            z_t=z_t,
            t=jnp.asarray(t),
            cond_emb=cond_emb,
        )

        sampled, keep_mask = sample_mixture_categorical(
            rng_token,
            destination_probs=probs,
            stay_prob=1.0 - rho_flip,
            change_prob=rho_flip,
            policy=self.categorical_sampling_policy,
        )
        masked_proposal = jnp.where(
            keep_mask,
            self.mask_token_id,
            sampled,
        ).astype(jnp.int32)
        x_s = masked_core.carry_over_unmasked(
            x_t,
            masked_proposal,
            mask_token_id=self.mask_token_id,
        )

        # Continuous latent update for positions that remain masked.
        stay_mask = x_s == self.mask_token_id

        z0_hat = self._estimate_z0(probs=probs)
        z0_hat_particles = z0_hat
        if z_t.ndim == x_t.ndim + 2:
            z0_hat_particles = jnp.broadcast_to(z0_hat[None, ...], z_t.shape)
        z_cont = self._reverse_masked_continuous_latent(
            rng=rng_latent,
            z_t=z_t,
            z0_hat=z0_hat_particles,
            s=jnp.asarray(s),
            t=jnp.asarray(t),
        )

        # For unmasked positions, z is deterministically the token embedding.
        z_unmasked = self.token_embed(x_s)
        stay_mask_latent = stay_mask[..., None]
        if z_t.ndim == x_t.ndim + 2:
            z_unmasked = jnp.broadcast_to(z_unmasked[None, ...], z_t.shape)
            stay_mask_latent = stay_mask[None, ..., None]
        z_s = jnp.where(stay_mask_latent, z_cont, z_unmasked)

        # Optional remasking-based refinement (keeps time fixed at `s`).
        if self.remask_refine_enabled and (self.remask_refine_steps > 0) and (self.remask_refine_frac > 0.0):
            # Use a deterministic split schedule so the refinement is JIT-friendly.
            for j in range(int(self.remask_refine_steps)):
                rng_body = jax.random.fold_in(rng_body, 10_000 + j)
                x_s, z_s = self._remask_refine_step(
                    rng_body,
                    x=x_s,
                    z=z_s,
                    t=jnp.asarray(s),
                    cond_emb=cond_emb,
                )

        return x_s, z_s

    def decode(
        self,
        state: tuple[Array, Array],
        *,
        conditioning: Array | None = None,
    ) -> Array:
        """Convert a (possibly partially masked) state into final token ids."""
        x, z = state
        mask = x == self.mask_token_id
        if not self.force_decode_at_end:
            return x
        cond_emb = self.get_cond_embedding(conditioning)
        probs = self._sampling_probs(
            x_t=x,
            z_t=z,
            t=jnp.asarray(0.0),
            cond_emb=cond_emb,
        )
        pred = jnp.argmax(probs, axis=-1).astype(jnp.int32)
        return jnp.where(mask, pred, x).astype(jnp.int32)
