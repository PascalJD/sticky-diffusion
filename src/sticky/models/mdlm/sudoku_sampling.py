from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp

from .sampling import _get_params, _initialize_sampling_state


Array = jnp.ndarray


def _flatten_mask(mask: Array) -> Array:
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    return mask.reshape((mask.shape[0], -1))


def _flatten_position_scores(scores: Array) -> Array:
    scores = jnp.asarray(scores, dtype=jnp.float32)
    return scores.reshape((scores.shape[0], -1))


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


def _sample_gumbel_noise(rng: Array, shape: tuple[int, ...]) -> Array:
    u = jax.random.uniform(
        rng,
        shape=shape,
        dtype=jnp.float32,
        minval=1.0e-6,
        maxval=1.0 - 1.0e-6,
    )
    return -jnp.log(-jnp.log(u))


def position_top_probabilities(log_probs: Array) -> Array:
    probs = jnp.exp(jnp.asarray(log_probs, dtype=jnp.float32))
    return jnp.max(probs, axis=-1)


def position_probability_margins(log_probs: Array) -> Array:
    probs = jnp.exp(jnp.asarray(log_probs, dtype=jnp.float32))
    if int(probs.shape[-1]) == 1:
        return probs[..., 0]
    top2 = jnp.sort(probs, axis=-1)[..., -2:]
    return top2[..., 1] - top2[..., 0]


def _apply_oracle_noise(
    rng: Array | None,
    scores: Array,
    *,
    valid_mask: Array,
    oracle_noise_type: str = "none",
    oracle_noise_scale: float = 0.0,
) -> Array:
    key = str(oracle_noise_type).strip().lower()
    if key in {"", "none"} or float(oracle_noise_scale) == 0.0:
        return jnp.where(valid_mask, scores, -jnp.inf)
    if key != "gumbel":
        raise ValueError(
            f"Unknown oracle_noise_type={oracle_noise_type!r}. "
            "Expected one of {'none', 'gumbel'}."
        )
    if rng is None:
        raise ValueError("Gumbel oracle noise requires an RNG key.")
    noise = _sample_gumbel_noise(rng, scores.shape)
    return jnp.where(valid_mask, scores + float(oracle_noise_scale) * noise, -jnp.inf)


def _select_top_scored_positions(
    scores: Array,
    masked_unknown_mask: Array,
    *,
    reveal_prob: Array,
    rng: Array | None = None,
    oracle_noise_type: str = "none",
    oracle_noise_scale: float = 0.0,
) -> Array:
    flat_mask = _flatten_mask(masked_unknown_mask)
    flat_scores = _flatten_position_scores(scores)
    k = _reveal_counts(masked_unknown_mask, reveal_prob)
    noisy_scores = _apply_oracle_noise(
        rng,
        flat_scores,
        valid_mask=flat_mask,
        oracle_noise_type=oracle_noise_type,
        oracle_noise_scale=float(oracle_noise_scale),
    )
    selected = _topk_mask_from_scores(noisy_scores, k=k, valid_mask=flat_mask)
    return selected.reshape(masked_unknown_mask.shape)


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


def select_top_probability_positions(
    log_probs: Array,
    masked_unknown_mask: Array,
    *,
    reveal_prob: Array,
    rng: Array | None = None,
    oracle_noise_type: str = "none",
    oracle_noise_scale: float = 0.0,
) -> Array:
    return _select_top_scored_positions(
        position_top_probabilities(log_probs),
        masked_unknown_mask,
        reveal_prob=reveal_prob,
        rng=rng,
        oracle_noise_type=oracle_noise_type,
        oracle_noise_scale=float(oracle_noise_scale),
    )


def select_top_prob_margin_positions(
    log_probs: Array,
    masked_unknown_mask: Array,
    *,
    reveal_prob: Array,
    rng: Array | None = None,
    oracle_noise_type: str = "none",
    oracle_noise_scale: float = 0.0,
) -> Array:
    return _select_top_scored_positions(
        position_probability_margins(log_probs),
        masked_unknown_mask,
        reveal_prob=reveal_prob,
        rng=rng,
        oracle_noise_type=oracle_noise_type,
        oracle_noise_scale=float(oracle_noise_scale),
    )


def _state_tokens(state: Array | tuple[Array, Array, Array]) -> Array:
    if isinstance(state, tuple) and len(state) == 3:
        return jnp.asarray(state[0], dtype=jnp.int32)
    return jnp.asarray(state, dtype=jnp.int32)


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
    return_diagnostics: bool = False,
) -> Array | tuple[Array, dict[str, Array]]:
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

    if not bool(return_diagnostics):
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

    zero = jnp.asarray(0.0, dtype=jnp.float32)
    diag0 = {
        "masked_unknown_total_across_steps": zero,
        "selected_count_total_across_steps": zero,
        "selected_margin_sum_total": zero,
        "selected_margin_count_total": zero,
        "selected_row_total_across_steps": zero,
        "selected_col_total_across_steps": zero,
        "selected_value_total_across_steps": zero,
    }

    def body_fn(i, carry):
        st, diag = carry
        next_state, step_info = model.apply(
            variables,
            step_rng,
            i,
            total_steps,
            st,
            conditioning=conditioning,
            known_tokens=known_tokens,
            known_token_mask=known_token_mask,
            return_info=True,
            method=model.sample_step,
        )
        next_state = _clamp_known_tokens_in_state(
            next_state,
            known_tokens=known_tokens,
            known_token_mask=known_token_mask,
        )
        diag = {
            "masked_unknown_total_across_steps": (
                diag["masked_unknown_total_across_steps"] + step_info["masked_unknown_total"]
            ),
            "selected_count_total_across_steps": (
                diag["selected_count_total_across_steps"] + step_info["selected_count_total"]
            ),
            "selected_margin_sum_total": (
                diag["selected_margin_sum_total"] + step_info["selected_margin_sum_total"]
            ),
            "selected_margin_count_total": (
                diag["selected_margin_count_total"] + step_info["selected_margin_count_total"]
            ),
            "selected_row_total_across_steps": (
                diag["selected_row_total_across_steps"] + step_info["selected_row_total"]
            ),
            "selected_col_total_across_steps": (
                diag["selected_col_total_across_steps"] + step_info["selected_col_total"]
            ),
            "selected_value_total_across_steps": (
                diag["selected_value_total_across_steps"] + step_info["selected_value_total"]
            ),
        }
        return next_state, diag

    state, diag = jax.lax.fori_loop(0, total_steps, body_fn, (state, diag0))
    tokens_before_decode = _state_tokens(state)
    decoded = model.apply(
        variables,
        state,
        conditioning=conditioning,
        method=model.decode,
    )
    decoded = jnp.where(known_token_mask, known_tokens, decoded).astype(jnp.int32)

    unknown_token_mask = ~known_token_mask
    diag = dict(diag)
    diag["example_step_count"] = jnp.asarray(
        int(total_steps) * int(known_tokens.shape[0]),
        dtype=jnp.float32,
    )
    diag["unknown_token_total"] = jnp.sum(unknown_token_mask.astype(jnp.float32))
    diag["final_masked_unknown_total"] = jnp.sum(
        ((tokens_before_decode == int(model.mask_token_id)) & unknown_token_mask).astype(jnp.float32)
    )
    return decoded, diag
