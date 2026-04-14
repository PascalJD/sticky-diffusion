from __future__ import annotations

import jax
import jax.numpy as jnp

from sticky.models.common import masked_discrete_core as masked_core


Array = jnp.ndarray


def uint_to_gray(values: Array) -> Array:
    values = jnp.asarray(values, dtype=jnp.int32)
    return jnp.bitwise_xor(values, jnp.right_shift(values, 1))


def gray_to_uint(values: Array) -> Array:
    values = jnp.asarray(values, dtype=jnp.int32)
    decoded = values
    shift = 1
    while shift < 32:
        decoded = jnp.bitwise_xor(decoded, jnp.right_shift(decoded, shift))
        shift *= 2
    return decoded


def _clip_uint_values(values: Array, *, num_bits: int) -> Array:
    max_value = (1 << int(num_bits)) - 1
    values = jnp.asarray(values, dtype=jnp.int32)
    return jnp.clip(values, 0, max_value)


def uint8_to_bit_planes(
    values: Array,
    *,
    num_bits: int = 8,
    gray_code: bool = False,
    msb_first: bool = True,
) -> Array:
    values = _clip_uint_values(values, num_bits=num_bits)
    if gray_code:
        values = uint_to_gray(values)

    if msb_first:
        shifts = jnp.arange(int(num_bits) - 1, -1, -1, dtype=jnp.int32)
    else:
        shifts = jnp.arange(int(num_bits), dtype=jnp.int32)
    return jnp.bitwise_and(jnp.right_shift(values[..., None], shifts), 1).astype(jnp.int32)


def bit_planes_to_uint8(
    bit_planes: Array,
    *,
    gray_code: bool = False,
    threshold: float | None = None,
    msb_first: bool = True,
) -> Array:
    if threshold is None:
        bits = jnp.asarray(bit_planes, dtype=jnp.int32)
    else:
        bits = analog_to_bits(bit_planes, threshold=threshold)

    num_bits = int(bits.shape[-1])
    if msb_first:
        shifts = jnp.arange(num_bits - 1, -1, -1, dtype=jnp.int32)
    else:
        shifts = jnp.arange(num_bits, dtype=jnp.int32)

    values = jnp.sum(jnp.left_shift(bits.astype(jnp.int32), shifts), axis=-1)
    if gray_code:
        values = gray_to_uint(values)
    return _clip_uint_values(values, num_bits=num_bits).astype(jnp.int32)


def bits_to_analog(
    bit_planes: Array,
    *,
    low: float = -1.0,
    high: float = 1.0,
) -> Array:
    bit_planes = jnp.asarray(bit_planes, dtype=jnp.float32)
    low = jnp.asarray(low, dtype=jnp.float32)
    high = jnp.asarray(high, dtype=jnp.float32)
    return jnp.where(bit_planes > 0.5, high, low)


def analog_to_bits(
    analog_bits: Array,
    *,
    threshold: float = 0.0,
) -> Array:
    analog_bits = jnp.asarray(analog_bits, dtype=jnp.float32)
    return (analog_bits >= float(threshold)).astype(jnp.int32)


def uint8_to_analog_bits(
    values: Array,
    *,
    num_bits: int = 8,
    gray_code: bool = False,
    low: float = -1.0,
    high: float = 1.0,
    msb_first: bool = True,
) -> Array:
    bits = uint8_to_bit_planes(
        values,
        num_bits=num_bits,
        gray_code=gray_code,
        msb_first=msb_first,
    )
    return bits_to_analog(bits, low=low, high=high)


def analog_bits_to_uint8(
    analog_bits: Array,
    *,
    gray_code: bool = False,
    threshold: float = 0.0,
    msb_first: bool = True,
) -> Array:
    return bit_planes_to_uint8(
        analog_bits,
        gray_code=gray_code,
        threshold=threshold,
        msb_first=msb_first,
    )


def continuous_signal_schedule(
    t: Array,
    *,
    schedule_fn_type: str = "linear",
    eps: float = 1e-4,
) -> Array:
    return masked_core.clipped_schedule_alpha(
        t,
        schedule_fn_type=schedule_fn_type,
        eps=eps,
    )


def continuous_signal_coefficients(
    gamma: Array,
    *,
    target_ndim: int | None = None,
) -> tuple[Array, Array]:
    gamma = jnp.clip(jnp.asarray(gamma, dtype=jnp.float32), a_min=0.0, a_max=1.0)
    if target_ndim is not None:
        gamma = masked_core.reverse_broadcast(gamma, target_ndim)
    return (
        jnp.sqrt(gamma),
        jnp.sqrt(jnp.maximum(1.0 - gamma, 0.0)),
    )


def continuous_forward_sample(
    rng: Array,
    x0: Array,
    *,
    gamma_t: Array,
    noise: Array | None = None,
) -> Array:
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    if noise is None:
        noise = jax.random.normal(rng, x0.shape, dtype=x0.dtype)
    else:
        noise = jnp.asarray(noise, dtype=x0.dtype)
    sqrt_gamma, sqrt_one_minus_gamma = continuous_signal_coefficients(
        gamma_t,
        target_ndim=x0.ndim,
    )
    return sqrt_gamma * x0 + sqrt_one_minus_gamma * noise


def predict_eps_from_x0(
    x_t: Array,
    x0_pred: Array,
    *,
    gamma_t: Array,
) -> Array:
    x_t = jnp.asarray(x_t, dtype=jnp.float32)
    x0_pred = jnp.asarray(x0_pred, dtype=jnp.float32)
    sqrt_gamma, sqrt_one_minus_gamma = continuous_signal_coefficients(
        gamma_t,
        target_ndim=x_t.ndim,
    )
    return (x_t - sqrt_gamma * x0_pred) / jnp.maximum(sqrt_one_minus_gamma, 1e-6)


def ddim_update_from_x0(
    rng: Array | None,
    *,
    x_t: Array,
    x0_pred: Array,
    gamma_t: Array,
    gamma_s: Array,
    eta: float = 0.0,
) -> Array:
    x_t = jnp.asarray(x_t, dtype=jnp.float32)
    x0_pred = jnp.asarray(x0_pred, dtype=jnp.float32)
    eps_pred = predict_eps_from_x0(x_t, x0_pred, gamma_t=gamma_t)

    gamma_t = jnp.clip(jnp.asarray(gamma_t, dtype=jnp.float32), a_min=1e-6, a_max=1.0)
    gamma_s = jnp.clip(jnp.asarray(gamma_s, dtype=jnp.float32), a_min=1e-6, a_max=1.0)
    gamma_t = masked_core.reverse_broadcast(gamma_t, x_t.ndim)
    gamma_s = masked_core.reverse_broadcast(gamma_s, x_t.ndim)

    sigma = float(eta) * jnp.sqrt(
        jnp.maximum((1.0 - gamma_s) / (1.0 - gamma_t), 0.0)
        * jnp.maximum(1.0 - gamma_t / gamma_s, 0.0)
    )
    mean = (
        jnp.sqrt(gamma_s) * x0_pred
        + jnp.sqrt(jnp.maximum(1.0 - gamma_s - sigma**2, 0.0)) * eps_pred
    )
    if float(eta) == 0.0:
        return mean
    if rng is None:
        raise ValueError("ddim_update_from_x0 requires rng when eta > 0.")
    noise = jax.random.normal(rng, x_t.shape, dtype=x_t.dtype)
    return mean + sigma * noise


def revealed_mask_from_tokens(tokens: Array, *, mask_token_id: int) -> Array:
    return jnp.asarray(tokens != int(mask_token_id))


def carry_over_revealed(
    current_state: Array,
    proposal: Array,
    *,
    revealed_mask: Array,
) -> Array:
    current_state = jnp.asarray(current_state)
    proposal = jnp.asarray(proposal)
    revealed_mask = masked_core.reverse_broadcast(
        jnp.asarray(revealed_mask, dtype=bool),
        proposal.ndim,
    )
    return jnp.where(revealed_mask, current_state, proposal)
