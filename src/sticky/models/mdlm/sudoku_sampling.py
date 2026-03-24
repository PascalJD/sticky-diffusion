from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp

from sticky.models.discrete_mixture import categorical_sample_from_logits

from .sampling import _get_params, _initialize_sampling_state


Array = jnp.ndarray


def _flatten_mask(mask: Array) -> Array:
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    return mask.reshape((mask.shape[0], -1))


def _reveal_counts(masked_unknown_mask: Array, reveal_prob: Array) -> Array:
    flat_mask = _flatten_mask(masked_unknown_mask)
    num_masked = jnp.sum(flat_mask.astype(jnp.int32), axis=-1)
    reveal_prob = jnp.asarray(reveal_prob, dtype=jnp.float32)
    raw = jnp.rint(num_masked.astype(jnp.float32) * reveal_prob).astype(jnp.int32)
    raw = jnp.clip(raw, 0, num_masked)
    needs_one = (num_masked > 0) & (reveal_prob > 0.0) & (raw == 0)
    return jnp.where(needs_one, 1, raw)


def _topk_mask_from_scores(scores: Array, *, k: Array, valid_mask: Array) -> Array:
    order = jnp.argsort(scores, axis=-1)[:, ::-1]
    ranks = jnp.argsort(order, axis=-1)
    return valid_mask & (ranks < k[:, None])


def select_uniform_reveal_positions(
    rng: Array,
    masked_unknown_mask: Array,
    *,
    reveal_prob: Array,
) -> Array:
    flat_mask = _flatten_mask(masked_unknown_mask)
    k = _reveal_counts(masked_unknown_mask, reveal_prob)
    scores = jax.random.uniform(rng, shape=flat_mask.shape, dtype=jnp.float32)
    scores = jnp.where(flat_mask, scores, -jnp.inf)
    selected = _topk_mask_from_scores(scores, k=k, valid_mask=flat_mask)
    return selected.reshape(masked_unknown_mask.shape)


def select_top_prob_margin_positions(
    log_probs: Array,
    masked_unknown_mask: Array,
    *,
    reveal_prob: Array,
) -> Array:
    flat_mask = _flatten_mask(masked_unknown_mask)
    flat_log_probs = log_probs.reshape((log_probs.shape[0], -1, log_probs.shape[-1]))
    probs = jnp.exp(flat_log_probs)
    if int(probs.shape[-1]) == 1:
        margin = probs[..., 0]
    else:
        top2 = jnp.sort(probs, axis=-1)[..., -2:]
        margin = top2[..., 1] - top2[..., 0]
    k = _reveal_counts(masked_unknown_mask, reveal_prob)
    scores = jnp.where(flat_mask, margin, -jnp.inf)
    selected = _topk_mask_from_scores(scores, k=k, valid_mask=flat_mask)
    return selected.reshape(masked_unknown_mask.shape)


def _clamp_known_tokens_in_state(
    state: Array | tuple[Array, Array, Array],
    *,
    known_tokens: Array,
    known_token_mask: Array,
) -> Array | tuple[Array, Array, Array]:
    if isinstance(state, tuple) and len(state) == 3:
        tokens, cached_log_probs, cache_valid = state
        clamped = jnp.where(known_token_mask, known_tokens, tokens).astype(jnp.int32)
        changed = jnp.any(clamped != tokens)
        cache_valid = jnp.asarray(cache_valid, dtype=jnp.bool_) & (~changed)
        return (clamped, cached_log_probs, cache_valid)
    return jnp.where(known_token_mask, known_tokens, state).astype(jnp.int32)


def conditional_generate(
    rng: Array,
    train_state: Any,
    *,
    model: Any,
    known_tokens: Array,
    known_token_mask: Array,
    timesteps: Optional[int] = None,
    conditioning: Optional[Array] = None,
    use_ema: bool = True,
) -> Array:
    params = _get_params(train_state, use_ema=use_ema)
    variables = {"params": params}

    known_tokens = jnp.asarray(known_tokens, dtype=jnp.int32)
    known_token_mask = jnp.asarray(known_token_mask, dtype=jnp.bool_)
    if known_tokens.shape != known_token_mask.shape:
        raise ValueError(
            "known_tokens and known_token_mask must have matching shapes, got "
            f"{known_tokens.shape} vs {known_token_mask.shape}."
        )

    total_steps = int(model.timesteps if timesteps is None else timesteps)
    prior_tokens = model.apply(variables, int(known_tokens.shape[0]), method=model.prior_sample)
    prior_tokens = jnp.where(known_token_mask, known_tokens, prior_tokens).astype(jnp.int32)
    state = _initialize_sampling_state(model, prior_tokens)

    rng, step_rng = jax.random.split(rng)

    def body_fn(i, st):
        next_state = model.apply(
            variables,
            step_rng,
            i,
            total_steps,
            st,
            conditioning=conditioning,
            known_tokens=known_tokens,
            known_token_mask=known_token_mask,
            method=model.sample_step,
        )
        return _clamp_known_tokens_in_state(
            next_state,
            known_tokens=known_tokens,
            known_token_mask=known_token_mask,
        )

    state = jax.lax.fori_loop(0, total_steps, body_fn, state)
    decoded = model.apply(
        variables,
        state,
        conditioning=conditioning,
        method=model.decode,
    )
    return jnp.where(known_token_mask, known_tokens, decoded).astype(jnp.int32)
