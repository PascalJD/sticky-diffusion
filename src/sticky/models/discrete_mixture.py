from __future__ import annotations

import jax
import jax.numpy as jnp


Array = jnp.ndarray


def _broadcast_prob(prob: Array, *, like: Array) -> Array:
    prob = jnp.asarray(prob, dtype=jnp.float32)
    while prob.ndim < like.ndim - 1:
        prob = prob[..., None]
    return jnp.broadcast_to(prob, like.shape[:-1])


def normalize_probs(probs: Array) -> Array:
    probs = jnp.asarray(probs, dtype=jnp.float32)
    probs = jnp.clip(probs, min=0.0)
    total = jnp.sum(probs, axis=-1, keepdims=True)
    fallback_idx = jnp.full(probs.shape[:-1], probs.shape[-1] - 1, dtype=jnp.int32)
    fallback = jax.nn.one_hot(fallback_idx, probs.shape[-1], dtype=probs.dtype)
    return jnp.where(total > 0.0, probs / jnp.maximum(total, 1e-20), fallback)


def categorical_sample_from_probs(rng: Array, probs: Array) -> Array:
    probs = normalize_probs(probs)
    logits = jnp.log(jnp.clip(probs, min=1e-20))
    return jax.random.categorical(rng, logits, axis=-1).astype(jnp.int32)


def sample_mixture_categorical(
    rng: Array,
    *,
    destination_probs: Array,
    stay_prob: Array,
    change_prob: Array | None = None,
) -> tuple[Array, Array]:
    """Sample one categorical over destinations plus an explicit stay state.

    We intentionally do not factor this into Bernoulli + categorical. In JAX,
    the single scaled categorical matches the MD4/CADD sampling law and keeps
    the Appendix G behavior from `jax.random.categorical`.
    """
    destination_probs = normalize_probs(destination_probs)
    stay_prob = _broadcast_prob(stay_prob, like=destination_probs)
    if change_prob is None:
        change_prob = 1.0 - stay_prob
    else:
        change_prob = _broadcast_prob(change_prob, like=destination_probs)

    stay_prob = jnp.clip(stay_prob, min=0.0, max=1.0)
    change_prob = jnp.clip(change_prob, min=0.0, max=1.0)

    full_probs = jnp.concatenate(
        [
            change_prob[..., None] * destination_probs,
            stay_prob[..., None],
        ],
        axis=-1,
    )
    choice = categorical_sample_from_probs(rng, full_probs)
    stay_idx = destination_probs.shape[-1]
    is_stay = choice == stay_idx
    dest_idx = jnp.where(is_stay, 0, choice).astype(jnp.int32)
    return dest_idx, is_stay
