from __future__ import annotations

import inspect
from typing import Literal

import jax
import jax.numpy as jnp


Array = jnp.ndarray
CategoricalSamplingPolicy = Literal["legacy_low", "jax_high", "exact"]
_VALID_CATEGORICAL_POLICIES = frozenset(("legacy_low", "jax_high", "exact"))


def _broadcast_prob(prob: Array, *, like: Array) -> Array:
    prob = jnp.asarray(prob, dtype=jnp.float32)
    while prob.ndim < like.ndim - 1:
        prob = prob[..., None]
    return jnp.broadcast_to(prob, like.shape[:-1])


def _canonicalize_categorical_policy(policy: str) -> str:
    key = str(policy).strip().lower()
    if key not in _VALID_CATEGORICAL_POLICIES:
        valid = ", ".join(sorted(_VALID_CATEGORICAL_POLICIES))
        raise ValueError(f"Unknown categorical sampling policy={policy!r}. Expected one of: {valid}.")
    return key


def _fallback_logits_like(logits: Array) -> Array:
    row = jnp.full((logits.shape[-1],), -jnp.inf, dtype=logits.dtype)
    row = row.at[-1].set(jnp.asarray(0.0, dtype=logits.dtype))
    return jnp.broadcast_to(row, logits.shape)


def _invalid_logits_rows(logits: Array) -> Array:
    has_nan = jnp.any(jnp.isnan(logits), axis=-1, keepdims=True)
    has_pos_inf = jnp.any(jnp.isinf(logits) & (logits > 0), axis=-1, keepdims=True)
    has_any_finite = jnp.any(jnp.isfinite(logits), axis=-1, keepdims=True)
    return has_nan | has_pos_inf | (~has_any_finite)


def _sanitize_logits(logits: Array) -> Array:
    logits = jnp.asarray(logits)
    if not jnp.issubdtype(logits.dtype, jnp.floating):
        logits = logits.astype(jnp.float32)
    fallback = _fallback_logits_like(logits)
    return jnp.where(_invalid_logits_rows(logits), fallback, logits)


def _log_prob_from_prob(prob: Array) -> Array:
    prob = jnp.asarray(prob, dtype=jnp.float32)
    return jnp.where(prob > 0.0, jnp.log(prob), -jnp.inf)


def _log_probs_from_logits(logits: Array) -> Array:
    return jax.nn.log_softmax(_sanitize_logits(logits), axis=-1)


def _legacy_log_probs_from_probs(probs: Array) -> Array:
    probs = normalize_probs(probs)
    floor = jnp.asarray(1e-20, dtype=probs.dtype)
    return jnp.log(jnp.maximum(probs, floor))


def _log_probs_from_probs(probs: Array) -> Array:
    probs = normalize_probs(probs)
    return _log_prob_from_prob(probs)


def _sample_gumbel(rng: Array, *, shape: tuple[int, ...], dtype: jnp.dtype, mode: str) -> Array:
    if mode == "low":
        return jax.random.gumbel(rng, shape=shape, dtype=dtype)
    if mode != "high":
        raise ValueError(f"Unknown Gumbel mode={mode!r}. Expected 'low' or 'high'.")

    info = jnp.finfo(dtype)
    uniforms = jax.random.uniform(
        rng,
        shape=(2,) + shape,
        dtype=dtype,
        minval=0.0,
        maxval=1.0,
    )
    high = uniforms[0]
    low = uniforms[1]
    ulp = jnp.asarray(2.0 ** (-info.nmant), dtype=dtype)
    x = jnp.where(high >= 0.5, high, high + ulp * low + info.tiny)
    return -jnp.log(-jnp.log1p(-x))


def _categorical_argmax_with_gumbel(rng: Array, logits: Array, *, mode: str) -> Array:
    logits = _sanitize_logits(logits)
    noise = _sample_gumbel(rng, shape=logits.shape, dtype=logits.dtype, mode=mode)
    return jnp.argmax(logits + noise, axis=-1).astype(jnp.int32)


def _jax_categorical_supports_mode(categorical_fn) -> bool:
    try:
        signature = inspect.signature(categorical_fn)
    except (TypeError, ValueError):
        return False
    return "mode" in signature.parameters


def _categorical_sample_with_explicit_mode(rng: Array, logits: Array, *, mode: str) -> Array:
    categorical_fn = jax.random.categorical
    if _jax_categorical_supports_mode(categorical_fn):
        return categorical_fn(rng, logits, axis=-1, mode=mode).astype(jnp.int32)
    return _categorical_argmax_with_gumbel(rng, logits, mode=mode)


def _categorical_sample_exact_from_logits(rng: Array, logits: Array) -> Array:
    log_probs = _log_probs_from_logits(logits)
    axis = log_probs.ndim - 1
    log_cdf = jax.lax.cumlogsumexp(log_probs, axis=axis)
    u = jax.random.uniform(
        rng,
        shape=log_probs.shape[:-1],
        dtype=log_probs.dtype,
        minval=0.0,
        maxval=1.0,
    )
    log_threshold = jnp.log1p(-u)
    choice = jnp.sum(log_cdf < log_threshold[..., None], axis=-1)
    return jnp.minimum(choice, log_probs.shape[-1] - 1).astype(jnp.int32)


def normalize_probs(probs: Array) -> Array:
    probs = jnp.asarray(probs, dtype=jnp.float32)
    probs = jnp.clip(probs, a_min=0.0)
    total = jnp.sum(probs, axis=-1, keepdims=True)
    fallback_idx = jnp.full(probs.shape[:-1], probs.shape[-1] - 1, dtype=jnp.int32)
    fallback = jax.nn.one_hot(fallback_idx, probs.shape[-1], dtype=probs.dtype)
    return jnp.where(total > 0.0, probs / jnp.maximum(total, 1e-20), fallback)


def categorical_sample_from_logits(
    rng: Array,
    logits: Array,
    *,
    policy: CategoricalSamplingPolicy = "legacy_low",
) -> Array:
    policy_key = _canonicalize_categorical_policy(policy)
    logits = jnp.asarray(logits)

    if policy_key == "legacy_low":
        return _categorical_sample_with_explicit_mode(rng, logits, mode="low")
    if policy_key == "jax_high":
        return _categorical_sample_with_explicit_mode(rng, logits, mode="high")
    return _categorical_sample_exact_from_logits(rng, logits)


def categorical_sample_from_probs(
    rng: Array,
    probs: Array,
    *,
    policy: CategoricalSamplingPolicy = "legacy_low",
) -> Array:
    policy_key = _canonicalize_categorical_policy(policy)
    if policy_key == "legacy_low":
        logits = _legacy_log_probs_from_probs(probs)
    else:
        logits = _log_probs_from_probs(probs)
    return categorical_sample_from_logits(rng, logits, policy=policy_key)


def _prepare_change_and_stay_probs(
    *,
    stay_prob: Array,
    change_prob: Array | None,
    like: Array,
) -> tuple[Array, Array]:
    stay_prob = _broadcast_prob(stay_prob, like=like)
    if change_prob is None:
        change_prob = 1.0 - stay_prob
    else:
        change_prob = _broadcast_prob(change_prob, like=like)

    stay_prob = jnp.clip(stay_prob, a_min=0.0, a_max=1.0)
    change_prob = jnp.clip(change_prob, a_min=0.0, a_max=1.0)
    return stay_prob, change_prob


def _full_mixture_logits_from_destination_logits(
    *,
    destination_logits: Array,
    stay_prob: Array,
    change_prob: Array | None,
) -> Array:
    stay_prob, change_prob = _prepare_change_and_stay_probs(
        stay_prob=stay_prob,
        change_prob=change_prob,
        like=destination_logits,
    )
    invalid_destination = _invalid_logits_rows(jnp.asarray(destination_logits))
    destination_log_probs = _log_probs_from_logits(destination_logits)
    full_logits = jnp.concatenate(
        [
            destination_log_probs + _log_prob_from_prob(change_prob)[..., None],
            _log_prob_from_prob(stay_prob)[..., None],
        ],
        axis=-1,
    )
    return jnp.where(invalid_destination, _fallback_logits_like(full_logits), full_logits)


def _full_mixture_probs_from_destination_probs(
    *,
    destination_probs: Array,
    stay_prob: Array,
    change_prob: Array | None,
) -> Array:
    destination_probs = normalize_probs(destination_probs)
    stay_prob, change_prob = _prepare_change_and_stay_probs(
        stay_prob=stay_prob,
        change_prob=change_prob,
        like=destination_probs,
    )
    return jnp.concatenate(
        [
            change_prob[..., None] * destination_probs,
            stay_prob[..., None],
        ],
        axis=-1,
    )


def sample_mixture_categorical(
    rng: Array,
    *,
    destination_probs: Array | None = None,
    destination_logits: Array | None = None,
    stay_prob: Array,
    change_prob: Array | None = None,
    policy: CategoricalSamplingPolicy = "legacy_low",
) -> tuple[Array, Array]:
    """Sample one categorical over destinations plus an explicit stay state.

    We intentionally do not factor this into Bernoulli + categorical. In JAX,
    the single scaled categorical matches the MD4/CADD sampling law and keeps
    the Appendix G behavior from `jax.random.categorical` when
    `policy="legacy_low"`. Other policies centralize the alternative JAX-safe
    implementations here.
    """
    policy_key = _canonicalize_categorical_policy(policy)
    if (destination_probs is None) == (destination_logits is None):
        raise ValueError("Provide exactly one of destination_probs or destination_logits.")

    if destination_logits is not None:
        stay_idx = int(destination_logits.shape[-1])
        if policy_key == "legacy_low":
            full_probs = _full_mixture_probs_from_destination_probs(
                destination_probs=jax.nn.softmax(jnp.asarray(destination_logits), axis=-1),
                stay_prob=stay_prob,
                change_prob=change_prob,
            )
            choice = categorical_sample_from_probs(rng, full_probs, policy=policy_key)
        else:
            full_logits = _full_mixture_logits_from_destination_logits(
                destination_logits=destination_logits,
                stay_prob=stay_prob,
                change_prob=change_prob,
            )
            choice = categorical_sample_from_logits(rng, full_logits, policy=policy_key)
    else:
        stay_idx = int(destination_probs.shape[-1])
        full_probs = _full_mixture_probs_from_destination_probs(
            destination_probs=destination_probs,
            stay_prob=stay_prob,
            change_prob=change_prob,
        )
        choice = categorical_sample_from_probs(rng, full_probs, policy=policy_key)

    is_stay = choice == stay_idx
    dest_idx = jnp.where(is_stay, 0, choice).astype(jnp.int32)
    return dest_idx, is_stay
