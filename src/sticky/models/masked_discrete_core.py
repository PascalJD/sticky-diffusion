from __future__ import annotations

import math

import jax
import jax.numpy as jnp


Array = jnp.ndarray


def mask_token_id(vocab_size: int) -> int:
    """Reserve `vocab_size` as the absorbing mask token id."""
    return int(vocab_size)


def reverse_broadcast(value: Array, ndim: int) -> Array:
    """Broadcast by adding singleton axes to the right."""
    value = jnp.asarray(value)
    if value.ndim > ndim:
        raise ValueError(
            f"Cannot reverse broadcast a value with {value.ndim} dimensions "
            f"to {ndim} dimensions."
        )
    if value.ndim < ndim:
        return value.reshape(value.shape + (1,) * (ndim - value.ndim))
    return value


def _base_schedule(t: Array, schedule_fn_type: str) -> Array:
    key = str(schedule_fn_type)
    if key == "linear":
        return 1.0 - t
    if key == "cosine":
        return 1.0 - jax.lax.cos(math.pi / 2.0 * (1.0 - t))
    if key.startswith("poly"):
        exponent = float(key.replace("poly", ""))
        return 1.0 - t**exponent
    raise NotImplementedError(f"Unknown schedule_fn_type={schedule_fn_type!r}")


def _base_schedule_derivative(t: Array, schedule_fn_type: str) -> Array:
    key = str(schedule_fn_type)
    if key == "linear":
        return -jnp.ones_like(t)
    if key == "cosine":
        return -math.pi / 2.0 * jax.lax.sin(math.pi / 2.0 * (1.0 - t))
    if key.startswith("poly"):
        exponent = float(key.replace("poly", ""))
        return -exponent * t ** (exponent - 1.0)
    raise NotImplementedError(f"Unknown schedule_fn_type={schedule_fn_type!r}")


def clipped_schedule_alpha(
    t: Array,
    *,
    schedule_fn_type: str = "linear",
    eps: float = 1e-4,
) -> Array:
    return (1.0 - 2.0 * float(eps)) * _base_schedule(t, schedule_fn_type) + float(eps)


def clipped_schedule_dalpha(
    t: Array,
    *,
    schedule_fn_type: str = "linear",
    eps: float = 1e-4,
) -> Array:
    return (1.0 - 2.0 * float(eps)) * _base_schedule_derivative(t, schedule_fn_type)


def masked_logit_schedule(
    t: Array,
    *,
    schedule_fn_type: str = "linear",
    eps: float = 1e-4,
) -> Array:
    alpha = clipped_schedule_alpha(t, schedule_fn_type=schedule_fn_type, eps=eps)
    return jnp.log(alpha / (1.0 - alpha))


def masked_dgamma_times_alpha(
    t: Array,
    *,
    schedule_fn_type: str = "linear",
    eps: float = 1e-4,
) -> Array:
    alpha = clipped_schedule_alpha(t, schedule_fn_type=schedule_fn_type, eps=eps)
    dalpha = clipped_schedule_dalpha(t, schedule_fn_type=schedule_fn_type, eps=eps)
    return dalpha / (1.0 - alpha)


def make_sampling_time_pair(
    i: int,
    timesteps: int,
    *,
    sampling_grid: str = "uniform",
) -> tuple[Array, Array]:
    t = (timesteps - i) / timesteps
    s = t - 1.0 / timesteps
    if str(sampling_grid) == "cosine":
        t = jnp.cos(math.pi / 2.0 * (1.0 - t))
        s = jnp.cos(math.pi / 2.0 * (1.0 - s))
    return s, t


def make_masked_token_prior(
    *,
    batch_size: int,
    data_shape: tuple[int, ...],
    vocab_size: int,
) -> Array:
    mask_id = mask_token_id(vocab_size)
    return mask_id * jnp.ones((int(batch_size),) + tuple(data_shape), dtype=jnp.int32)


def sample_masked_discrete_forward(
    rng: Array,
    x0: Array,
    *,
    keep_prob: Array,
    mask_token_id: int,
) -> Array:
    unmask = jax.random.bernoulli(rng, jnp.asarray(keep_prob, dtype=jnp.float32), x0.shape)
    return jnp.where(unmask, x0, int(mask_token_id)).astype(jnp.int32)


def masked_positions(tokens: Array, *, mask_token_id: int) -> Array:
    return (tokens == int(mask_token_id)).astype(jnp.float32)


def gather_true_log_probs(logits: Array, targets: Array) -> Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(
        log_probs,
        jnp.expand_dims(targets, axis=-1),
        axis=-1,
    )[..., 0]


def masked_logprob_sums(
    logits: Array,
    targets: Array,
    *,
    mask: Array,
) -> Array:
    true_log_probs = gather_true_log_probs(logits, targets)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    sum_axes = tuple(range(1, targets.ndim))
    return jnp.sum(true_log_probs * mask, axis=sum_axes)


def masked_cross_entropy_sums(
    logits: Array,
    targets: Array,
    *,
    mask: Array,
) -> Array:
    return -masked_logprob_sums(logits, targets, mask=mask)


def carry_over_unmasked(
    current_tokens: Array,
    proposal: Array,
    *,
    mask_token_id: int,
) -> Array:
    return jnp.where(current_tokens == int(mask_token_id), proposal, current_tokens).astype(
        jnp.int32
    )
