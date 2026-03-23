from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from sticky.models.architectures.factory import build_image_token_backbone
from sticky.models.architectures.networks.conditioning import CondEmbedding
from sticky.models import continuous_discrete_core as continuous_core


Array = jnp.ndarray


class BitDiffusion(nn.Module):
    """Continuous analog-bit diffusion on discrete image tokens."""

    data_shape: tuple[int, ...]

    cont_time: bool = True
    timesteps: int = 256
    num_bits: int = 8
    encoding: str = "uint8"  # uint8 | gray
    predict_target: str = "x0"  # x0 | eps
    loss_type: str = "mse"
    self_conditioning: bool = True
    self_conditioning_rate: float = 0.5
    analog_bit_scale: float = 1.0
    clip_x0: bool = True
    signal_schedule_type: str = "linear"  # linear | cosine | poly{p}
    schedule_eps: float = 0.0

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

    sampler: str = "ddim"  # ddim | ddpm
    sampling_grid: str = "uniform"  # uniform | cosine
    time_difference: float = 0.0
    stochasticity: float = 0.0

    def setup(self):
        if len(tuple(self.data_shape)) != 3:
            raise ValueError(
                "BitDiffusion expects image data_shape=(H, W, C), "
                f"got data_shape={self.data_shape}."
            )
        if int(self.timesteps) <= 0:
            raise ValueError("BitDiffusion requires timesteps > 0.")
        if int(self.num_bits) <= 0:
            raise ValueError("BitDiffusion requires num_bits > 0.")

        encoding_key = str(self.encoding).lower()
        if encoding_key not in {"uint8", "gray"}:
            raise ValueError(
                f"Unsupported BitDiffusion encoding={self.encoding!r}. "
                "Expected one of: uint8, gray."
            )
        target_key = str(self.predict_target).lower()
        if target_key not in {"x0", "eps"}:
            raise ValueError(
                f"Unsupported BitDiffusion predict_target={self.predict_target!r}. "
                "Expected one of: x0, eps."
            )
        if str(self.loss_type).lower() != "mse":
            raise NotImplementedError(
                "BitDiffusion currently supports only loss_type='mse'."
            )
        if int(self.classes) > 0:
            raise NotImplementedError(
                "This BitDiffusion baseline is unconditional and does not support classes > 0."
            )
        if not (0.0 <= float(self.self_conditioning_rate) <= 1.0):
            raise ValueError(
                "BitDiffusion self_conditioning_rate must lie in [0, 1]. "
                f"Got {self.self_conditioning_rate}."
            )
        if float(self.time_difference) < 0.0:
            raise ValueError(
                "BitDiffusion time_difference must be >= 0. "
                f"Got {self.time_difference}."
            )
        if float(self.stochasticity) < 0.0:
            raise ValueError(
                "BitDiffusion stochasticity must be >= 0. "
                f"Got {self.stochasticity}."
            )
        if str(self.sampler).lower() not in {"ddim", "ddpm"}:
            raise ValueError(
                f"Unsupported BitDiffusion sampler={self.sampler!r}. "
                "Expected one of: ddim, ddpm."
            )
        if (
            str(self.sampler).lower() == "ddpm"
            and not math.isclose(float(self.stochasticity), 1.0, rel_tol=0.0, abs_tol=1e-6)
        ):
            raise ValueError(
                "BitDiffusion sampler='ddpm' is a DDIM-style alias with fixed eta=1.0. "
                "Set stochasticity=1.0 or use sampler='ddim' for custom eta."
            )
        if str(self.sampling_grid).lower() not in {"uniform", "cosine"}:
            raise ValueError(
                f"Unsupported BitDiffusion sampling_grid={self.sampling_grid!r}. "
                "Expected one of: uniform, cosine."
            )

        self.time_cond_embed = CondEmbedding(int(self.feature_dim))
        self.backbone = build_image_token_backbone(
            name=self.image_backbone,
            feature_dim=int(self.feature_dim),
            n_layers=int(self.n_layers),
            n_dit_layers=int(self.n_dit_layers),
            dit_num_heads=int(self.dit_num_heads),
            dit_hidden_size=int(self.dit_hidden_size),
            ch_mult=tuple(int(v) for v in self.ch_mult),
            output_channels=int(self.num_bits),
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
    def gray_code(self) -> bool:
        return str(self.encoding).lower() == "gray"

    @property
    def analog_low(self) -> float:
        return -float(self.analog_bit_scale)

    @property
    def analog_high(self) -> float:
        return float(self.analog_bit_scale)

    def _validate_conditioning(self, conditioning: Array | None) -> None:
        if conditioning is not None:
            raise NotImplementedError(
                "This BitDiffusion baseline is unconditional and does not support conditioning."
            )

    def _sample_training_times(self, rng: Array, batch_size: int) -> Array:
        if bool(self.cont_time):
            return jax.random.uniform(
                rng,
                (int(batch_size),),
                minval=0.0,
                maxval=1.0,
                dtype=jnp.float32,
            )
        t_idx = jax.random.randint(
            rng,
            (int(batch_size),),
            minval=0,
            maxval=int(self.timesteps),
            dtype=jnp.int32,
        )
        return (t_idx.astype(jnp.float32) + 0.5) / float(self.timesteps)

    def _signal_level(self, t: Array) -> Array:
        return continuous_core.continuous_signal_schedule(
            jnp.asarray(t, dtype=jnp.float32),
            schedule_fn_type=self.signal_schedule_type,
            eps=float(self.schedule_eps),
        )

    def _clip_x0(self, x0: Array) -> Array:
        if not bool(self.clip_x0):
            return x0
        return jnp.clip(
            x0,
            min=min(self.analog_low, self.analog_high),
            max=max(self.analog_low, self.analog_high),
        )

    def encode_tokens(self, x: Array) -> Array:
        return continuous_core.uint8_to_analog_bits(
            x,
            num_bits=int(self.num_bits),
            gray_code=bool(self.gray_code),
            low=float(self.analog_low),
            high=float(self.analog_high),
        ).astype(jnp.float32)

    def _decode_analog_bits(self, x: Array) -> Array:
        return continuous_core.analog_bits_to_uint8(
            x,
            gray_code=bool(self.gray_code),
            threshold=0.0,
        ).astype(jnp.int32)

    def _prepare_backbone_input(
        self,
        x_t: Array,
        *,
        self_cond: Array | None,
    ) -> Array:
        x_t = jnp.asarray(x_t, dtype=jnp.float32)
        if not bool(self.self_conditioning):
            return x_t
        if self_cond is None:
            self_cond = jnp.zeros_like(x_t)
        else:
            self_cond = jnp.asarray(self_cond, dtype=x_t.dtype)
        return jnp.concatenate([x_t, self_cond], axis=-1)

    def _normalize_time(self, t: Array | None, *, batch_size: int) -> Array | None:
        if t is None or str(self.time_features).lower() == "none":
            return None
        t = jnp.asarray(t, dtype=jnp.float32)
        if jnp.isscalar(t) or t.ndim == 0:
            t = t * jnp.ones((int(batch_size),), dtype=t.dtype)
        return t

    def predict_raw(
        self,
        x_t: Array,
        t: Array | None,
        *,
        self_cond: Array | None = None,
        cond: Array | None = None,
        train: bool = False,
    ) -> Array:
        self._validate_conditioning(cond)
        z = self._prepare_backbone_input(x_t, self_cond=self_cond)
        t = self._normalize_time(t, batch_size=int(z.shape[0]))
        use_adm_time_path = str(self.image_backbone).lower() == "adm_unet5d"

        cond_in = None
        timesteps = None
        if t is not None:
            if use_adm_time_path:
                timesteps = t * 1000.0
            else:
                cond_in = self.time_cond_embed(t * 1000.0, cond=None)

        return self.backbone(
            z,
            cond=cond_in,
            timesteps=timesteps,
            train=train,
        )

    def _raw_to_x0(self, raw: Array, x_t: Array, t: Array | None) -> Array:
        if str(self.predict_target).lower() == "x0":
            return self._clip_x0(jnp.asarray(raw, dtype=jnp.float32))

        signal = self._signal_level(jnp.asarray(t, dtype=jnp.float32))
        sqrt_signal, sqrt_noise = continuous_core.continuous_signal_coefficients(
            signal,
            target_ndim=x_t.ndim,
        )
        x0 = (jnp.asarray(x_t, dtype=jnp.float32) - sqrt_noise * jnp.asarray(raw, dtype=jnp.float32)) / jnp.maximum(
            sqrt_signal,
            1e-6,
        )
        return self._clip_x0(x0)

    def predict_x0(
        self,
        x_t: Array,
        t: Array | None,
        *,
        self_cond: Array | None = None,
        cond: Array | None = None,
        train: bool = False,
    ) -> Array:
        raw = self.predict_raw(
            x_t,
            t,
            self_cond=self_cond,
            cond=cond,
            train=train,
        )
        return self._raw_to_x0(raw, x_t, t)

    def _predict_target_and_x0(
        self,
        x_t: Array,
        t: Array | None,
        *,
        self_cond: Array | None = None,
        cond: Array | None = None,
        train: bool = False,
    ) -> tuple[Array, Array]:
        raw = self.predict_raw(
            x_t,
            t,
            self_cond=self_cond,
            cond=cond,
            train=train,
        )
        x0_pred = self._raw_to_x0(raw, x_t, t)
        if str(self.predict_target).lower() == "x0":
            return x0_pred, x0_pred
        return raw, x0_pred

    def _sample_self_condition(
        self,
        x_t: Array,
        t: Array,
        *,
        train: bool,
    ) -> Array | None:
        if not bool(self.self_conditioning):
            return None

        zeros = jnp.zeros_like(x_t)
        if not train:
            return zeros

        use_self_cond = jax.random.bernoulli(
            self.make_rng("sample"),
            p=float(self.self_conditioning_rate),
        )

        def _with_self_cond(_: None) -> Array:
            pred_x0 = self.predict_x0(
                x_t,
                t,
                self_cond=zeros,
                cond=None,
                train=False,
            )
            return jax.lax.stop_gradient(pred_x0)

        return jax.lax.cond(use_self_cond, _with_self_cond, lambda _: zeros, operand=None)

    def _sampling_time_pair(self, i: int, timesteps: int) -> tuple[Array, Array]:
        total_steps = jnp.asarray(timesteps, dtype=jnp.float32)
        step = jnp.asarray(i, dtype=jnp.float32)
        t_now = 1.0 - step / total_steps
        t_next = jnp.maximum(
            1.0 - (step + 1.0 + float(self.time_difference)) / total_steps,
            0.0,
        )

        if str(self.sampling_grid).lower() == "cosine":
            t_now = 1.0 - jnp.cos(jnp.asarray(math.pi / 2.0, dtype=jnp.float32) * t_now)
            t_next = 1.0 - jnp.cos(jnp.asarray(math.pi / 2.0, dtype=jnp.float32) * t_next)

        return jnp.asarray(t_now, dtype=jnp.float32), jnp.asarray(t_next, dtype=jnp.float32)

    def _sampling_eta(self) -> float:
        if str(self.sampler).lower() == "ddpm":
            return 1.0
        return float(self.stochasticity)

    def _extract_sample_latent(self, state: Any) -> Array:
        if hasattr(state, "keys") and "latent" in state:
            return jnp.asarray(state["latent"], dtype=jnp.float32)
        return jnp.asarray(state, dtype=jnp.float32)

    def _extract_sample_self_cond(self, state: Any) -> Array | None:
        if hasattr(state, "keys") and "self_cond" in state:
            return jnp.asarray(state["self_cond"], dtype=jnp.float32)
        if bool(self.self_conditioning):
            return None
        return None

    def prior_sample(self, batch_size: int) -> Array:
        return jax.random.normal(
            self.make_rng("sample"),
            (int(batch_size),) + tuple(self.data_shape) + (int(self.num_bits),),
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

        x_bits = self.encode_tokens(x)
        rng_t, rng_noise = jax.random.split(self.make_rng("sample"))
        t = self._sample_training_times(rng_t, int(x_bits.shape[0]))
        noise = jax.random.normal(rng_noise, x_bits.shape, dtype=jnp.float32)
        x_t = continuous_core.continuous_forward_sample(
            rng_noise,
            x_bits,
            gamma_t=self._signal_level(t),
            noise=noise,
        )

        self_cond = self._sample_self_condition(x_t, t, train=train)
        pred_target, pred_x0 = self._predict_target_and_x0(
            x_t,
            t,
            self_cond=self_cond,
            cond=None,
            train=train,
        )

        if str(self.predict_target).lower() == "x0":
            target = x_bits
        else:
            target = noise

        per_example_loss = jnp.mean(
            jnp.square(pred_target - target),
            axis=tuple(range(1, pred_target.ndim)),
        )
        loss = jnp.mean(per_example_loss)

        return {
            "loss": loss,
            "loss_mse": loss,
            "t_mean": jnp.mean(t),
            "x0_abs_mean": jnp.mean(jnp.abs(pred_x0)),
        }

    def sample_step(
        self,
        rng: Array,
        i: int,
        timesteps: int,
        state: Any,
        *,
        conditioning: Array | None = None,
    ) -> Any:
        self._validate_conditioning(conditioning)
        if int(timesteps) <= 0:
            raise ValueError(f"BitDiffusion sample timesteps must be > 0, got {timesteps}.")

        x_t = self._extract_sample_latent(state)
        self_cond = self._extract_sample_self_cond(state)
        t_now, t_next = self._sampling_time_pair(i, timesteps)

        x0_pred = self.predict_x0(
            x_t,
            t_now,
            self_cond=self_cond,
            cond=None,
            train=False,
        )

        eta = self._sampling_eta()
        step_rng = jax.random.fold_in(rng, i)
        x_next = continuous_core.ddim_update_from_x0(
            step_rng if eta > 0.0 else None,
            x_t=x_t,
            x0_pred=x0_pred,
            gamma_t=self._signal_level(t_now),
            gamma_s=self._signal_level(t_next),
            eta=eta,
        )

        if hasattr(state, "keys") and "latent" in state:
            return {
                "latent": x_next,
                "self_cond": jax.lax.stop_gradient(x0_pred),
            }
        return x_next

    def decode(
        self,
        state: Any,
        *,
        conditioning: Array | None = None,
    ) -> Array:
        self._validate_conditioning(conditioning)
        if hasattr(state, "keys") and "self_cond" in state:
            analog_bits = jnp.asarray(state["self_cond"], dtype=jnp.float32)
        else:
            analog_bits = self._extract_sample_latent(state)
        return self._decode_analog_bits(analog_bits)
