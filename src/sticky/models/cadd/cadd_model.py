from __future__ import annotations

import math
from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from sticky.models.architectures.factory import (
    build_image_backbone,
    build_sequence_backbone,
)
from sticky.models.architectures.networks.conditioning import CondEmbedding
from sticky.models.md4 import utils as md4_utils


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
        key = str(self.schedule_fn_type)
        if key == "linear":
            return 1.0 - t
        if key == "cosine":
            # Matches the MD4 cosine schedule shape: alpha(0)=1, alpha(1)=0.
            return 1.0 - jnp.cos(math.pi / 2.0 * (1.0 - t))
        if key.startswith("poly"):
            exponent = float(key.replace("poly", ""))
            return 1.0 - t**exponent
        raise NotImplementedError(f"Unknown schedule_fn_type={self.schedule_fn_type!r}")

    def __call__(self, t: Array) -> Array:
        return (1.0 - 2.0 * float(self.eps)) * self._base(t) + float(self.eps)


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
      * Multi-sample estimation is currently implemented for K=1 only.
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
    K: int = 1
    force_decode_at_end: bool = True

    # Optional sampling-time corrector (in the spirit of Gat et al., 2024).
    # The CADD paper reports applying a corrector for image generation, but does
    # not fully specify hyperparameters; we expose a simple remasking corrector.
    corrector_enabled: bool = False
    corrector_steps: int = 1
    corrector_remask_frac: float = 0.0
    corrector_metric: str = "entropy"  # entropy | neg_entropy
    corrector_sample_mode: str = "sample"  # sample | argmax

    def setup(self):
        if int(self.K) != 1:
            raise NotImplementedError(
                "CADD multi-sample estimation with K>1 is not implemented yet."
            )

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
        return int(self.vocab_size)

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
        t_b = md4_utils.reverse_broadcast(t, x0.ndim)
        keep_prob = self.alpha(t_b)
        un_mask = jax.random.bernoulli(self.make_rng("sample"), keep_prob, x0.shape)
        return jnp.where(un_mask, x0, self.mask_token_id).astype(jnp.int32)

    def _continuous_gamma_bar(self, t: Array, *, target_ndim: int) -> Array:
        gbar = self.gamma_bar(md4_utils.reverse_broadcast(jnp.asarray(t), target_ndim))
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

    def _cross_entropy_on_masked(self, *, logits: Array, x0: Array, x_t: Array) -> tuple[Array, dict[str, Array]]:
        """Compute -log p(x0 | x_t, z_t) restricted to masked positions."""
        log_p = jax.nn.log_softmax(logits, axis=-1)
        # Gather log-prob for the true token at each position.
        x0_gather = jnp.expand_dims(x0, axis=-1)
        lp_true = jnp.take_along_axis(log_p, x0_gather, axis=-1)[..., 0]
        neg_logp = -lp_true

        mask = (x_t == self.mask_token_id).astype(jnp.float32)
        loss_pos = neg_logp * mask

        # Sum over all token positions.
        sum_axes = tuple(range(1, x0.ndim))
        per_ex = jnp.sum(loss_pos, axis=sum_axes)
        loss = jnp.mean(per_ex)

        metrics = {
            "loss_ce": loss,
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

        loss, metrics = self._cross_entropy_on_masked(logits=logits, x0=x, x_t=x_t)
        metrics["loss"] = loss
        metrics["t_mean"] = jnp.mean(t)

        # Match MD4-style logging: convert losses to bits-per-token.
        return md4_utils.loss2bpt(metrics, self.data_shape)

    # ------------------------------- Sampling -------------------------------

    def get_sampling_grid(self, i: int, timesteps: int) -> tuple[Array, Array]:
        """Return (s, t) with t decreasing from ~1 to ~0."""
        t = (timesteps - i) / timesteps
        s = t - 1.0 / timesteps
        if self.sampling_grid == "cosine":
            t = jnp.cos(math.pi / 2.0 * (1.0 - t))
            s = jnp.cos(math.pi / 2.0 * (1.0 - s))
        return s, t

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
        x = self.mask_token_id * jnp.ones((batch_size,) + tuple(self.data_shape), dtype=jnp.int32)
        z = jax.random.normal(
            self.make_rng("sample"),
            (batch_size,) + tuple(self.data_shape) + (int(self.feature_dim),),
        )
        return x, z

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

    def _corrector_step(
        self,
        rng: Array,
        *,
        x: Array,
        z: Array,
        t: Array,
        cond_emb: Array | None,
    ) -> tuple[Array, Array]:
        """A simple remask/resample corrector step at fixed time `t`.

        This is a lightweight, JIT-friendly approximation inspired by
        remasking-based correctors for discrete diffusion/flow models.
        """
        remask_frac = float(self.corrector_remask_frac)
        if remask_frac <= 0.0:
            return x, z

        # 1) Score current tokens by uncertainty (entropy).
        z_disc = self.token_embed(x)
        is_mask = (x == self.mask_token_id)
        z_tilde = z_disc + z * is_mask.astype(jnp.float32)[..., None]
        logits = self.predict_logits(z_tilde, jnp.asarray(t), cond=cond_emb, train=False)
        logits = logits / self.temperature(jnp.asarray(t))
        probs = jax.nn.softmax(logits, axis=-1)

        eps = 1e-20
        ent = -jnp.sum(probs * jnp.log(jnp.clip(probs, eps, 1.0)), axis=-1)
        if self.corrector_metric == "neg_entropy":
            score = -ent
        elif self.corrector_metric == "entropy":
            score = ent
        else:
            raise NotImplementedError(
                f"Unknown corrector_metric={self.corrector_metric!r}"
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
        rng, rng_eps, rng_tok = jax.random.split(rng, 3)
        z_noisy = self._sample_masked_continuous_latent(
            rng=rng_eps,
            z0=z0,
            t=jnp.asarray(t),
        )
        z_masked = jnp.where(remask[..., None], z_noisy, z)

        # 3) Resample remasked positions (unmask immediately).
        z_disc2 = self.token_embed(x_masked)
        is_mask2 = x_masked == self.mask_token_id
        z_tilde2 = z_disc2 + z_masked * is_mask2.astype(jnp.float32)[..., None]
        logits2 = self.predict_logits(
            z_tilde2, jnp.asarray(t), cond=cond_emb, train=False
        )
        logits2 = logits2 / self.temperature(jnp.asarray(t))

        if self.corrector_sample_mode == "argmax":
            sampled = jnp.argmax(logits2, axis=-1).astype(jnp.int32)
        elif self.corrector_sample_mode == "sample":
            sampled = jax.random.categorical(rng_tok, logits2, axis=-1).astype(jnp.int32)
        else:
            raise NotImplementedError(
                f"Unknown corrector_sample_mode={self.corrector_sample_mode!r}"
            )

        x_new = jnp.where(remask, sampled, x).astype(jnp.int32)
        z_new = jnp.where(remask[..., None], self.token_embed(x_new), z_masked)
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
        rng_body, rng_flip, rng_token, rng_latent = jax.random.split(rng_body, 4)

        s, t = self.get_sampling_grid(i, timesteps)
        cond_emb = self.get_cond_embedding(conditioning)

        # Discrete flip/keep probabilities for masked positions.
        alpha_t = self.alpha(jnp.asarray(t))
        alpha_s = self.alpha(jnp.asarray(s))

        rho_flip = (alpha_s - alpha_t) / (1.0 - alpha_t)
        rho_flip = jnp.clip(rho_flip, a_min=0.0, a_max=1.0)

        # Model logits p_theta(x0 | x_t, z_t).
        z_disc = self.token_embed(x_t)
        mask = (x_t == self.mask_token_id).astype(jnp.float32)[..., None]
        z_tilde = z_disc + z_t * mask

        logits = self.predict_logits(z_tilde, jnp.asarray(t), cond=cond_emb, train=False)
        logits = logits / self.temperature(jnp.asarray(t))
        probs = jax.nn.softmax(logits, axis=-1)

        # Decide which masked positions to unmask this step.
        to_flip = jax.random.bernoulli(rng_flip, rho_flip, x_t.shape) & (
            x_t == self.mask_token_id
        )

        # Sample new tokens at flip positions.
        sampled = jax.random.categorical(rng_token, logits, axis=-1).astype(jnp.int32)
        x_s = jnp.where(to_flip, sampled, x_t).astype(jnp.int32)

        # Continuous latent update for positions that remain masked.
        stay_mask = x_s == self.mask_token_id

        z0_hat = self._estimate_z0(probs=probs)
        z_cont = self._reverse_masked_continuous_latent(
            rng=rng_latent,
            z_t=z_t,
            z0_hat=z0_hat,
            s=jnp.asarray(s),
            t=jnp.asarray(t),
        )

        # For unmasked positions, z is deterministically the token embedding.
        z_unmasked = self.token_embed(x_s)
        z_s = jnp.where(stay_mask[..., None], z_cont, z_unmasked)

        # Optional remasking-based corrector (keeps time fixed at `s`).
        if self.corrector_enabled and (self.corrector_steps > 0) and (self.corrector_remask_frac > 0.0):
            # Use a deterministic split schedule so the corrector is JIT-friendly.
            for j in range(int(self.corrector_steps)):
                rng_body = jax.random.fold_in(rng_body, 10_000 + j)
                x_s, z_s = self._corrector_step(
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
        z_disc = self.token_embed(x)
        z_tilde = z_disc + z * mask.astype(jnp.float32)[..., None]
        logits = self.predict_logits(z_tilde, jnp.asarray(0.0), cond=cond_emb, train=False)
        pred = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        return jnp.where(mask, pred, x).astype(jnp.int32)
