from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from sticky.models.backbones.image import ContinuousADMBackbone


Array = jnp.ndarray


class DDPM(nn.Module):
    """Classical continuous-state Gaussian DDPM for raw images."""

    data_shape: tuple[int, ...]

    timesteps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    prediction_type: str = "eps"
    variance_type: str = "fixed_small"
    clip_x0: bool = True

    feature_dim: int = 96
    ch_mult: Sequence[int] = (3, 4, 4)
    dropout_rate: float = 0.1
    image_backbone: str = "adm_unet2d"

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

    classes: int = -1

    def setup(self):
        if len(tuple(self.data_shape)) != 3:
            raise ValueError(
                "DDPM expects image data_shape=(H, W, C), "
                f"got data_shape={self.data_shape}."
            )
        if int(self.timesteps) <= 0:
            raise ValueError("DDPM requires timesteps > 0.")
        if str(self.beta_schedule).lower() != "linear":
            raise NotImplementedError(
                "DDPM currently supports only beta_schedule='linear'."
            )
        if str(self.prediction_type).lower() != "eps":
            raise NotImplementedError(
                "DDPM currently supports only prediction_type='eps'."
            )
        if str(self.variance_type).lower() != "fixed_small":
            raise NotImplementedError(
                "DDPM currently supports only variance_type='fixed_small'."
            )
        if int(self.classes) > 0:
            raise NotImplementedError(
                "This DDPM baseline is unconditional and does not support classes > 0."
            )

        key = str(self.image_backbone).lower()
        if key not in {"adm", "adm_unet", "adm_unet2d"}:
            raise ValueError(
                f"Unsupported DDPM image_backbone={self.image_backbone!r}. "
                "Expected one of: adm, adm_unet, adm_unet2d."
            )

        channels = int(self.data_shape[-1])
        self.backbone = ContinuousADMBackbone(
            in_channels=channels,
            output_channels=channels,
            feature_dim=int(self.feature_dim),
            num_res_blocks=int(self.adm_num_res_blocks),
            attention_resolutions=tuple(int(v) for v in self.adm_attention_resolutions),
            channel_mult=tuple(int(v) for v in self.ch_mult),
            num_heads=int(self.adm_num_heads),
            num_head_channels=int(self.adm_num_head_channels),
            num_heads_upsample=int(self.adm_num_heads_upsample),
            dropout_rate=float(self.dropout_rate),
            conv_resample=bool(self.adm_conv_resample),
            use_scale_shift_norm=bool(self.adm_use_scale_shift_norm),
            resblock_updown=bool(self.adm_resblock_updown),
            use_conv_skip=bool(self.adm_use_conv_skip),
            use_new_attention_order=bool(self.adm_use_new_attention_order),
        )

        beta_start = float(self.beta_start)
        beta_end = float(self.beta_end)
        if (beta_start <= 0.0) or (beta_end >= 1.0) or (beta_start >= beta_end):
            raise ValueError(
                "DDPM requires betas to lie strictly in (0, 1). "
                f"Got beta_start={self.beta_start}, beta_end={self.beta_end}."
            )
        betas = jnp.linspace(
            beta_start,
            beta_end,
            int(self.timesteps),
            dtype=jnp.float32,
        )

        alphas = 1.0 - betas
        alpha_bars = jnp.cumprod(alphas, axis=0)
        alpha_bars_prev = jnp.concatenate(
            [jnp.ones((1,), dtype=jnp.float32), alpha_bars[:-1]],
            axis=0,
        )

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.alpha_bars_prev = alpha_bars_prev
        self.sqrt_alpha_bars = jnp.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = jnp.sqrt(1.0 - alpha_bars)
        self.sqrt_recip_alpha_bars = jnp.sqrt(1.0 / alpha_bars)
        self.sqrt_recipm1_alpha_bars = jnp.sqrt(1.0 / alpha_bars - 1.0)
        self.posterior_variance = (
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )
        self.posterior_mean_coef1 = (
            betas * jnp.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alpha_bars_prev) * jnp.sqrt(alphas) / (1.0 - alpha_bars)
        )

    def _validate_conditioning(self, conditioning: Array | None) -> None:
        if conditioning is not None:
            raise NotImplementedError(
                "This DDPM baseline is unconditional and does not support conditioning."
            )

    def _normalize_images(self, x: Array) -> Array:
        return jnp.asarray(x, dtype=jnp.float32) / 127.5 - 1.0

    def _denormalize_images(self, x: Array) -> Array:
        x = jnp.clip(x, -1.0, 1.0)
        return jnp.clip((x + 1.0) * 127.5, 0.0, 255.0).astype(jnp.float32)

    def _extract(self, values: Array, t: Array, ndim: int) -> Array:
        out = jnp.take(values, t, axis=0)
        return out.reshape(out.shape + (1,) * (ndim - out.ndim))

    def q_sample(self, x0: Array, t: Array, noise: Array) -> Array:
        return (
            self._extract(self.sqrt_alpha_bars, t, x0.ndim) * x0
            + self._extract(self.sqrt_one_minus_alpha_bars, t, x0.ndim) * noise
        )

    def predict_eps(
        self,
        x_t: Array,
        t: Array,
        *,
        cond: Array | None = None,
        train: bool = False,
    ) -> Array:
        self._validate_conditioning(cond)
        return self.backbone(
            x_t,
            cond=None,
            timesteps=jnp.asarray(t, dtype=jnp.float32),
            train=train,
        )

    def prior_sample(self, batch_size: int) -> Array:
        return jax.random.normal(
            self.make_rng("sample"),
            (int(batch_size),) + tuple(self.data_shape),
            dtype=jnp.float32,
        )

    def __call__(
        self,
        x: Array,
        *,
        cond: Array | None = None,
        train: bool = False,
    ) -> dict[str, Array]:
        self._validate_conditioning(cond)
        x0 = self._normalize_images(x)

        rng_t, rng_noise = jax.random.split(self.make_rng("sample"))
        t = jax.random.randint(
            rng_t,
            (int(x0.shape[0]),),
            minval=0,
            maxval=int(self.timesteps),
            dtype=jnp.int32,
        )
        noise = jax.random.normal(rng_noise, x0.shape, dtype=jnp.float32)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.predict_eps(x_t, t, cond=None, train=train)

        per_example_mse = jnp.mean(
            jnp.square(eps_pred - noise),
            axis=tuple(range(1, noise.ndim)),
        )
        loss_mse = jnp.mean(per_example_mse)

        return {
            "loss": loss_mse,
            "loss_mse": loss_mse,
            "t_mean": jnp.mean(t.astype(jnp.float32)),
        }

    def sample_step(
        self,
        rng: Array,
        i: int,
        timesteps: int,
        state: Array,
        *,
        conditioning: Array | None = None,
    ) -> Array:
        self._validate_conditioning(conditioning)

        x_t = jnp.asarray(state, dtype=jnp.float32)
        if not hasattr(timesteps, "shape") and int(timesteps) != int(self.timesteps):
            raise ValueError(
                "DDPM currently requires sample_timesteps == model.timesteps. "
                f"Got sample_timesteps={int(timesteps)} and model.timesteps={int(self.timesteps)}."
            )
        t_idx = (
            jnp.asarray(timesteps, dtype=jnp.int32)
            - jnp.asarray(i, dtype=jnp.int32)
            - 1
        )

        eps_pred = self.predict_eps(x_t, t_idx, cond=None, train=False)
        pred_x0 = (
            self._extract(self.sqrt_recip_alpha_bars, t_idx, x_t.ndim) * x_t
            - self._extract(self.sqrt_recipm1_alpha_bars, t_idx, x_t.ndim) * eps_pred
        )
        if bool(self.clip_x0):
            pred_x0 = jnp.clip(pred_x0, -1.0, 1.0)

        model_mean = (
            self._extract(self.posterior_mean_coef1, t_idx, x_t.ndim) * pred_x0
            + self._extract(self.posterior_mean_coef2, t_idx, x_t.ndim) * x_t
        )

        rng_noise = jax.random.fold_in(rng, i)
        noise = jax.random.normal(rng_noise, x_t.shape, dtype=jnp.float32)
        nonzero_mask = jnp.asarray(t_idx > 0, dtype=jnp.float32)
        while nonzero_mask.ndim < x_t.ndim:
            nonzero_mask = nonzero_mask[..., None]

        return model_mean + nonzero_mask * jnp.sqrt(
            self._extract(self.posterior_variance, t_idx, x_t.ndim)
        ) * noise

    def decode(
        self,
        x0: Array,
        *,
        conditioning: Array | None = None,
    ) -> Array:
        self._validate_conditioning(conditioning)
        return self._denormalize_images(jnp.asarray(x0, dtype=jnp.float32))
