from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from sticky.models import masked_discrete_core as masked_core
from sticky.models.architectures.factory import build_image_token_backbone
from sticky.models.architectures.networks.conditioning import CondEmbedding
from sticky.models.discrete_mixture import categorical_sample_from_logits


Array = jnp.ndarray


def _loss2bpt(loss_dict: dict[str, Array], data_shape: tuple[int, ...]) -> dict[str, Array]:
    seq_len = jnp.prod(jnp.asarray(data_shape))
    scale = 1.0 / (seq_len * jnp.log(2.0))
    return {
        key: value * scale if "loss" in key else value
        for key, value in loss_dict.items()
    }


class TokenEmbedding(nn.Module):
    """Lightweight embedding table with explicit table access."""

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


class CANDI(nn.Module):
    """Experimental CIFAR-native CANDI adaptation.

    This keeps the paper's hybrid clean/corrupted state semantics, weighted
    cross-entropy objective, and hybrid reverse sampler, but adapts them to the
    repo's discrete-image CIFAR-10 pipeline and shared ADM 5D image-token path.
    """

    data_shape: tuple[int, ...]
    vocab_size: int = 256

    cont_time: bool = True
    timesteps: int = 256
    representation: str = "embed"  # embed | onehot (debug-only)
    experimental: bool = True

    alpha_schedule_type: str = "linear"
    schedule_eps: float = 0.0

    use_percentile_scheduling: bool = True
    min_percentile: float = 0.01
    max_percentile: float = 0.45
    sigma_min: float = 0.2
    sigma_max: float = 4.0
    ode_step_scale: float = 1.0

    feature_dim: int = 96
    num_heads: int = 12
    n_layers: int = 32
    n_dit_layers: int = 0
    dit_num_heads: int = 12
    dit_hidden_size: int = 768
    ch_mult: Sequence[int] = (3, 4, 4)

    dropout_rate: float = 0.1
    use_attn_dropout: bool = True
    mlp_type: str = "swiglu"
    depth_scaled_init: bool = False
    cond_type: str = "adaln"
    model_sharding: bool = False
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

    sampler: str = "continuous"  # continuous | hybrid_cache | hybrid_expected | hybrid_exact(onehot-only)
    sampling_grid: str = "cosine"
    categorical_sampling_policy: str = "legacy_low"
    guidance_scale: float = 0.0  # Placeholder for future external guidance.

    def setup(self):
        if len(tuple(self.data_shape)) != 3:
            raise ValueError(
                "CANDI expects image data_shape=(H, W, C), "
                f"got data_shape={self.data_shape}."
            )
        if int(self.timesteps) <= 0:
            raise ValueError("CANDI requires timesteps > 0.")
        if int(self.vocab_size) <= 1:
            raise ValueError("CANDI requires vocab_size > 1.")
        if int(self.classes) > 0:
            raise NotImplementedError(
                "This experimental CIFAR CANDI path is unconditional only."
            )
        if str(self.representation).lower() not in {"embed", "onehot"}:
            raise ValueError(
                f"Unsupported CANDI representation={self.representation!r}. "
                "Expected one of: embed, onehot."
            )
        sampler_key = self._sampler_mode()
        if sampler_key not in {"continuous", "hybrid_cache", "hybrid_expected", "hybrid_exact"}:
            raise ValueError(
                f"Unsupported CANDI sampler={self.sampler!r}. "
                "Expected one of: continuous, hybrid_cache, hybrid_expected, hybrid_exact."
            )
        if str(self.sampling_grid).lower() not in {"uniform", "cosine"}:
            raise ValueError(
                f"Unsupported CANDI sampling_grid={self.sampling_grid!r}. "
                "Expected one of: uniform, cosine."
            )
        if not (0.0 < float(self.min_percentile) < float(self.max_percentile) < 0.5):
            raise ValueError(
                "CANDI percentile scheduling requires "
                "0 < min_percentile < max_percentile < 0.5."
            )
        if float(self.sigma_min) <= 0.0 or float(self.sigma_max) <= 0.0:
            raise ValueError("CANDI sigma_min and sigma_max must be > 0.")
        if float(self.sigma_min) > float(self.sigma_max):
            raise ValueError("CANDI requires sigma_min <= sigma_max.")

        self._repr_dim = (
            int(self.feature_dim)
            if str(self.representation).lower() == "embed"
            else int(self.vocab_size)
        )

        if str(self.representation).lower() == "embed":
            self.token_embed = TokenEmbedding(
                vocab_size=int(self.vocab_size),
                dim=int(self.feature_dim),
                init_std=1.0,
            )
        else:
            self.token_embed = None

        self.corruption_embed = nn.Embed(2, int(self._repr_dim))
        self.time_cond_embed = CondEmbedding(int(self.feature_dim))
        self.backbone = build_image_token_backbone(
            name=self.image_backbone,
            feature_dim=int(self.feature_dim),
            n_layers=int(self.n_layers),
            n_dit_layers=int(self.n_dit_layers),
            dit_num_heads=int(self.dit_num_heads),
            dit_hidden_size=int(self.dit_hidden_size),
            ch_mult=tuple(int(v) for v in self.ch_mult),
            output_channels=int(self.vocab_size),
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

    def _sampler_mode(self) -> str:
        sampler_key = str(self.sampler).lower()
        if sampler_key not in {"continuous", "hybrid_cache", "hybrid_expected", "hybrid_exact"}:
            return sampler_key
        if sampler_key == "hybrid_exact" and str(self.representation).lower() != "onehot":
            raise ValueError(
                "CANDI sampler='hybrid_exact' is reserved for representation='onehot'. "
                "For embed mode, use sampler='hybrid_expected' or 'hybrid_cache'."
            )
        return sampler_key

    def _is_embed_mode(self) -> bool:
        return str(self.representation).lower() == "embed"

    def _validate_conditioning(self, conditioning: Array | None) -> None:
        if conditioning is not None:
            raise NotImplementedError(
                "This experimental CIFAR CANDI path is unconditional only."
            )

    def alpha(self, t: Array) -> Array:
        return masked_core.clipped_schedule_alpha(
            jnp.asarray(t, dtype=jnp.float32),
            schedule_fn_type=self.alpha_schedule_type,
            eps=float(self.schedule_eps),
        )

    def discrete_noise(self, t: Array) -> Array:
        return jnp.clip(1.0 - self.alpha(t), min=1e-6, max=1.0)

    def sigma_from_discrete_noise(self, discrete_noise: Array) -> Array:
        discrete_noise = jnp.clip(jnp.asarray(discrete_noise, dtype=jnp.float32), 0.0, 1.0)
        # The torch reference always uses the VE/log-linear sigma rule for
        # embedding diffusion, even if percentile scheduling is enabled for
        # discrete/onehot variants.
        if self._is_embed_mode():
            sigma = float(self.sigma_min) * (
                float(self.sigma_max) / float(self.sigma_min)
            ) ** discrete_noise
            return jnp.clip(sigma, float(self.sigma_min), float(self.sigma_max))

        if bool(self.use_percentile_scheduling):
            percentile = (
                discrete_noise
                * (float(self.max_percentile) - float(self.min_percentile))
                + float(self.min_percentile)
            )
            percentile = jnp.clip(percentile, 1e-6, 0.499999)
            z = jax.scipy.special.ndtri(percentile)
            sigma = -1.0 / (jnp.sqrt(2.0) * jnp.minimum(z, -1e-6))
            return jnp.clip(sigma, float(self.sigma_min), float(self.sigma_max))

        sigma = float(self.sigma_min) * (
            float(self.sigma_max) / float(self.sigma_min)
        ) ** discrete_noise
        return jnp.clip(sigma, float(self.sigma_min), float(self.sigma_max))

    def _sample_training_times(self, rng: Array, batch_size: int) -> Array:
        if bool(self.cont_time):
            min_t = 1.0 / float(max(int(self.timesteps), 2))
            return jax.random.uniform(
                rng,
                (int(batch_size),),
                minval=min_t,
                maxval=1.0,
                dtype=jnp.float32,
            )

        t_idx = jax.random.randint(
            rng,
            (int(batch_size),),
            minval=1,
            maxval=int(self.timesteps) + 1,
            dtype=jnp.int32,
        )
        return t_idx.astype(jnp.float32) / float(self.timesteps)

    def clean_representation(self, tokens: Array) -> Array:
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        if str(self.representation).lower() == "embed":
            return self.token_embed(tokens)
        return jax.nn.one_hot(tokens, int(self.vocab_size), dtype=jnp.float32)

    def _representation_from_probs(self, probs: Array) -> Array:
        probs = jnp.asarray(probs, dtype=jnp.float32)
        if str(self.representation).lower() == "embed":
            table = self.token_embed.table_float()
            return jnp.einsum("...k,kd->...d", probs, table)
        return probs

    def _sigma_feature(self, sigma: Array, *, target_ndim: int) -> Array:
        return masked_core.reverse_broadcast(
            jnp.asarray(sigma, dtype=jnp.float32),
            target_ndim,
        )[..., None]

    def corrupt_input(
        self,
        x: Array,
        t: Array,
        *,
        rng: Array | None = None,
    ) -> dict[str, Array]:
        if rng is None:
            rng = self.make_rng("sample")
        rng_mask, rng_noise = jax.random.split(rng)

        x = jnp.asarray(x, dtype=jnp.int32)
        clean_repr = self.clean_representation(x)
        alpha_t = self.alpha(t)
        keep_prob = masked_core.reverse_broadcast(alpha_t, x.ndim)
        clean_mask = jax.random.bernoulli(rng_mask, keep_prob, x.shape)
        corrupted_mask = ~clean_mask

        sigma = self.sigma_from_discrete_noise(self.discrete_noise(t))
        sigma_feat = self._sigma_feature(sigma, target_ndim=x.ndim)
        noise = jax.random.normal(rng_noise, clean_repr.shape, dtype=jnp.float32)
        corrupted_repr = clean_repr + sigma_feat * noise
        hybrid_repr = jnp.where(clean_mask[..., None], clean_repr, corrupted_repr)

        return {
            "alpha": alpha_t,
            "sigma": sigma,
            "clean_mask": clean_mask,
            "corrupted_mask": corrupted_mask,
            "clean_repr": clean_repr,
            "corrupted_repr": corrupted_repr,
            "hybrid_repr": hybrid_repr,
        }

    def _build_input_features(
        self,
        *,
        tokens: Array,
        clean_mask: Array,
        continuous: Array,
        sigma: Array,
    ) -> Array:
        exact_repr = self.clean_representation(tokens)
        base = jnp.where(clean_mask[..., None], exact_repr, continuous)
        corruption_ids = (~clean_mask).astype(jnp.int32)
        base = base + self.corruption_embed(corruption_ids)

        sigma_feat = jnp.broadcast_to(
            self._sigma_feature(sigma, target_ndim=tokens.ndim),
            tokens.shape + (1,),
        )
        clean_feat = clean_mask.astype(jnp.float32)[..., None]
        return jnp.concatenate([base, sigma_feat, clean_feat], axis=-1)

    def _normalize_time(self, t: Array | None, *, batch_size: int) -> Array | None:
        if t is None or str(self.time_features).lower() == "none":
            return None
        t = jnp.asarray(t, dtype=jnp.float32)
        if jnp.isscalar(t) or t.ndim == 0:
            t = t * jnp.ones((int(batch_size),), dtype=t.dtype)
        return t

    def predict_logits(
        self,
        z_tilde: Array,
        t: Array | None,
        *,
        cond: Array | None = None,
        train: bool,
    ) -> Array:
        self._validate_conditioning(cond)
        t = self._normalize_time(t, batch_size=int(z_tilde.shape[0]))
        use_adm_time_path = (z_tilde.ndim == 5) and (str(self.image_backbone).lower() == "adm_unet5d")

        cond_in = None
        timesteps = None
        if t is not None:
            if use_adm_time_path:
                timesteps = t * 1000.0
            else:
                cond_in = self.time_cond_embed(t * 1000.0, cond=None)

        return self.backbone(
            z_tilde,
            cond=cond if use_adm_time_path else cond_in,
            timesteps=timesteps,
            train=train,
        )

    def _apply_carry_over_logits(
        self,
        logits: Array,
        *,
        tokens: Array,
        clean_mask: Array,
    ) -> Array:
        carry_logits = jnp.where(
            jax.nn.one_hot(tokens, int(self.vocab_size), dtype=bool),
            jnp.asarray(0.0, dtype=logits.dtype),
            jnp.asarray(-jnp.inf, dtype=logits.dtype),
        )
        return jnp.where(clean_mask[..., None], carry_logits, logits)

    def _predict_state_logits(
        self,
        *,
        tokens: Array,
        clean_mask: Array,
        continuous: Array,
        t: Array,
        train: bool,
        sigma_override: Array | None = None,
    ) -> Array:
        sigma = (
            self.sigma_from_discrete_noise(self.discrete_noise(t))
            if sigma_override is None
            else jnp.asarray(sigma_override, dtype=jnp.float32)
        )
        features = self._build_input_features(
            tokens=tokens,
            clean_mask=clean_mask,
            continuous=continuous,
            sigma=sigma,
        )
        logits = self.predict_logits(features, t, cond=None, train=train)
        return self._apply_carry_over_logits(logits, tokens=tokens, clean_mask=clean_mask)

    def weighted_masked_ce_loss(
        self,
        *,
        logits: Array,
        targets: Array,
        corrupted_mask: Array,
        t: Array,
        eps: float = 1e-4,
    ) -> dict[str, Array]:
        corrupted_mask = jnp.asarray(corrupted_mask, dtype=jnp.float32)
        per_example_ce_sum = masked_core.masked_cross_entropy_sums(
            logits,
            targets,
            mask=corrupted_mask,
        )
        weight = 1.0 / jnp.maximum(self.discrete_noise(t), float(eps))
        loss_ce = jnp.mean(weight * per_example_ce_sum)
        return {
            "loss": loss_ce,
            "loss_ce": loss_ce,
            "mask_frac": jnp.mean(corrupted_mask),
            "per_example_ce_sum": per_example_ce_sum,
            "loss_weight": weight,
        }

    def continuous_target_from_predictions(
        self,
        *,
        probs: Array,
        predicted_tokens: Array,
    ) -> Array:
        sampler_mode = self._sampler_mode()
        if sampler_mode == "hybrid_cache":
            return self.clean_representation(predicted_tokens)
        return self._representation_from_probs(probs)

    def get_sampling_grid(self, i: int, timesteps: int) -> tuple[Array, Array]:
        return masked_core.make_sampling_time_pair(
            i,
            timesteps,
            sampling_grid=self.sampling_grid,
        )

    @nn.compact
    def __call__(
        self,
        x: Array,
        *,
        cond: Array | None = None,
        train: bool = False,
    ) -> dict[str, Array]:
        self._validate_conditioning(cond)
        x = jnp.asarray(x, dtype=jnp.int32)

        rng_t, rng_corrupt = jax.random.split(self.make_rng("sample"))
        t = self._sample_training_times(rng_t, int(x.shape[0]))
        corruption = self.corrupt_input(x, t, rng=rng_corrupt)

        logits = self._predict_state_logits(
            tokens=x,
            clean_mask=corruption["clean_mask"],
            continuous=corruption["hybrid_repr"],
            t=t,
            train=train,
        )

        metrics = self.weighted_masked_ce_loss(
            logits=logits,
            targets=x,
            corrupted_mask=corruption["corrupted_mask"],
            t=t,
        )
        metrics = {
            "loss": metrics["loss"],
            "loss_ce": metrics["loss_ce"],
            "mask_frac": metrics["mask_frac"],
            "sigma_mean": jnp.mean(corruption["sigma"]),
            "t_mean": jnp.mean(t),
        }
        return _loss2bpt(metrics, self.data_shape)

    def prior_sample(self, batch_size: int) -> dict[str, Array]:
        batch_size = int(batch_size)
        if self._is_embed_mode():
            rng_noise = self.make_rng("sample")
            tokens = jnp.zeros((batch_size,) + tuple(self.data_shape), dtype=jnp.int32)
            clean_mask = jnp.zeros(tokens.shape, dtype=bool)
            noise = jax.random.normal(
                rng_noise,
                tokens.shape + (int(self.feature_dim),),
                dtype=jnp.float32,
            )
            # Match the torch embed prior: start from pure Gaussian latent noise,
            # not token embeddings plus noise.
            continuous = float(self.sigma_max) * noise
            return {
                "tokens": tokens,
                "clean_mask": clean_mask,
                "continuous": continuous,
            }

        rng_tokens, rng_noise = jax.random.split(self.make_rng("sample"))
        tokens = jax.random.randint(
            rng_tokens,
            (batch_size,) + tuple(self.data_shape),
            minval=0,
            maxval=int(self.vocab_size),
            dtype=jnp.int32,
        )
        sigma = self.sigma_from_discrete_noise(jnp.ones((batch_size,), dtype=jnp.float32))
        clean_mask = jnp.zeros(tokens.shape, dtype=bool)
        base_repr = self.clean_representation(tokens)
        continuous = base_repr + self._sigma_feature(sigma, target_ndim=tokens.ndim) * jax.random.normal(
            rng_noise,
            base_repr.shape,
            dtype=jnp.float32,
        )
        return {
            "tokens": tokens,
            "clean_mask": clean_mask,
            "continuous": continuous,
        }

    def _continuous_step(
        self,
        *,
        tokens: Array,
        clean_mask: Array,
        continuous: Array,
        t: Array,
        s: Array,
    ) -> dict[str, Array]:
        # The torch reference conditions on the current discrete time but uses
        # the next sigma in the continuous update and sigma feature input.
        sigma_from = self.sigma_from_discrete_noise(self.discrete_noise(t))
        sigma_to = self.sigma_from_discrete_noise(self.discrete_noise(s))
        logits = self._predict_state_logits(
            tokens=tokens,
            clean_mask=clean_mask,
            continuous=continuous,
            t=t,
            train=False,
            sigma_override=sigma_to,
        )
        probs = jax.nn.softmax(logits, axis=-1)
        target_continuous = self._representation_from_probs(probs)

        sigma_to_feat = self._sigma_feature(sigma_to, target_ndim=tokens.ndim)
        sigma_to_sq = jnp.maximum(sigma_to_feat**2, 1e-6)
        delta_sigma = jnp.maximum(
            self._sigma_feature(sigma_from - sigma_to, target_ndim=tokens.ndim),
            0.0,
        )
        score = (continuous - target_continuous) / sigma_to_sq
        continuous_proposal = continuous - float(self.ode_step_scale) * delta_sigma * score

        next_tokens = jnp.where(clean_mask, tokens, jnp.argmax(logits, axis=-1)).astype(
            jnp.int32
        )
        next_exact = self.clean_representation(next_tokens)
        next_continuous = jnp.where(
            clean_mask[..., None],
            next_exact,
            continuous_proposal,
        )
        return {
            "tokens": next_tokens,
            "clean_mask": clean_mask,
            "continuous": next_continuous,
        }

    def _hybrid_step(
        self,
        *,
        rng: Array,
        tokens: Array,
        clean_mask: Array,
        continuous: Array,
        t: Array,
        s: Array,
    ) -> dict[str, Array]:
        logits = self._predict_state_logits(
            tokens=tokens,
            clean_mask=clean_mask,
            continuous=continuous,
            t=t,
            train=False,
        )
        probs = jax.nn.softmax(logits, axis=-1)

        rng_tokens, rng_reveal = jax.random.split(rng)
        predicted_tokens = categorical_sample_from_logits(
            rng_tokens,
            logits,
            policy=self.categorical_sampling_policy,
        )

        target_continuous = self.continuous_target_from_predictions(
            probs=probs,
            predicted_tokens=predicted_tokens,
        )

        alpha_t = self.alpha(t)
        alpha_s = self.alpha(s)
        reveal_prob = jnp.clip(
            (alpha_s - alpha_t) / jnp.maximum(1.0 - alpha_t, 1e-6),
            min=0.0,
            max=1.0,
        )
        newly_revealed = (~clean_mask) & jax.random.bernoulli(
            rng_reveal,
            reveal_prob[:, None, None, None],
            clean_mask.shape,
        )
        next_clean_mask = clean_mask | newly_revealed

        sigma_t = self.sigma_from_discrete_noise(self.discrete_noise(t))
        sigma_s = self.sigma_from_discrete_noise(self.discrete_noise(s))
        sigma_sq = jnp.maximum(self._sigma_feature(sigma_t, target_ndim=tokens.ndim) ** 2, 1e-6)
        delta_sigma = jnp.maximum(
            self._sigma_feature(sigma_t - sigma_s, target_ndim=tokens.ndim),
            0.0,
        )
        score = (continuous - target_continuous) / sigma_sq
        continuous_proposal = continuous - float(self.ode_step_scale) * delta_sigma * score

        next_tokens = jnp.where(clean_mask, tokens, predicted_tokens).astype(jnp.int32)
        next_exact = self.clean_representation(next_tokens)
        next_continuous = jnp.where(
            next_clean_mask[..., None],
            next_exact,
            continuous_proposal,
        )

        return {
            "tokens": next_tokens,
            "clean_mask": next_clean_mask,
            "continuous": next_continuous,
        }

    def sample_step(
        self,
        rng: Array,
        i: int,
        timesteps: int,
        state: dict[str, Array],
        *,
        conditioning: Array | None = None,
    ) -> dict[str, Array]:
        self._validate_conditioning(conditioning)

        tokens = jnp.asarray(state["tokens"], dtype=jnp.int32)
        clean_mask = jnp.asarray(state["clean_mask"], dtype=bool)
        continuous = jnp.asarray(state["continuous"], dtype=jnp.float32)

        s, t = self.get_sampling_grid(i, int(timesteps))
        t_arr = jnp.full((tokens.shape[0],), jnp.asarray(t, dtype=jnp.float32))
        s_arr = jnp.full((tokens.shape[0],), jnp.asarray(s, dtype=jnp.float32))
        rng = jax.random.fold_in(rng, i)
        if self._is_embed_mode():
            return self._continuous_step(
                tokens=tokens,
                clean_mask=clean_mask,
                continuous=continuous,
                t=t_arr,
                s=s_arr,
            )
        return self._hybrid_step(
            rng=rng,
            tokens=tokens,
            clean_mask=clean_mask,
            continuous=continuous,
            t=t_arr,
            s=s_arr,
        )

    def decode(
        self,
        state: dict[str, Array] | Array,
        *,
        conditioning: Array | None = None,
    ) -> Array:
        self._validate_conditioning(conditioning)
        if isinstance(state, dict):
            tokens = state["tokens"]
        else:
            tokens = state
        return jnp.clip(jnp.asarray(tokens, dtype=jnp.int32), 0, int(self.vocab_size) - 1)
