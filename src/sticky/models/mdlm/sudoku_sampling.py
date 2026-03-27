from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp

from sticky.models import masked_discrete_core as masked_core
from sticky.models.discrete_mixture import categorical_sample_from_logits

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


def _zero_sampling_info(dtype: Any = jnp.float32) -> dict[str, Array]:
    zero = jnp.asarray(0.0, dtype=dtype)
    return {
        "masked_unknown_total": zero,
        "selected_count_total": zero,
        "selected_margin_sum_total": zero,
        "selected_margin_count_total": zero,
        "selected_row_total": zero,
        "selected_col_total": zero,
        "selected_value_total": zero,
    }


def is_reveal_order_sampler(sampler: str) -> bool:
    return sampler in {"uniform", "top_probability", "top_prob_margin"}


def _normalize_token_mask(
    mask: Array | None,
    *,
    ref: Array,
    name: str,
) -> Array | None:
    if mask is None:
        return None
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    if mask.shape != ref.shape:
        raise ValueError(f"{name} must match reference shape {ref.shape}, got {mask.shape}.")
    return mask


def clamp_known_tokens(
    tokens: Array,
    *,
    current_tokens: Array,
    known_token_mask: Array | None = None,
    known_tokens: Array | None = None,
) -> Array:
    known_token_mask = _normalize_token_mask(
        known_token_mask,
        ref=tokens,
        name="known_token_mask",
    )
    if known_token_mask is None:
        return tokens
    source = current_tokens if known_tokens is None else jnp.asarray(known_tokens, dtype=jnp.int32)
    return jnp.where(known_token_mask, source, tokens).astype(jnp.int32)


def masked_unknown_positions(
    tokens: Array,
    *,
    mask_token_id: int,
    known_token_mask: Array | None = None,
) -> Array:
    known_token_mask = _normalize_token_mask(
        known_token_mask,
        ref=tokens,
        name="known_token_mask",
    )
    if known_token_mask is None:
        known_token_mask = jnp.zeros_like(tokens, dtype=jnp.bool_)
    return (tokens == mask_token_id) & (~known_token_mask)


def sampling_step_info(
    *,
    masked_unknown: Array,
    reveal_positions: Array | None = None,
    margins: Array | None = None,
) -> dict[str, Array]:
    info = _zero_sampling_info()
    info["masked_unknown_total"] = jnp.sum(masked_unknown.astype(jnp.float32))
    if reveal_positions is None:
        return info

    reveal_positions = jnp.asarray(reveal_positions, dtype=jnp.bool_)
    selected = reveal_positions.astype(jnp.float32)
    info["selected_count_total"] = jnp.sum(selected)
    info["selected_margin_count_total"] = info["selected_count_total"]
    if margins is not None:
        info["selected_margin_sum_total"] = jnp.sum(jnp.asarray(margins, dtype=jnp.float32) * selected)
    token_pos = jnp.arange(reveal_positions.shape[-1], dtype=jnp.int32)
    token_pos = token_pos.reshape((1,) * (reveal_positions.ndim - 1) + (-1,))
    info["selected_row_total"] = jnp.sum(
        selected * (jnp.mod(token_pos, 3) == 0).astype(jnp.float32)
    )
    info["selected_col_total"] = jnp.sum(
        selected * (jnp.mod(token_pos, 3) == 1).astype(jnp.float32)
    )
    info["selected_value_total"] = jnp.sum(
        selected * (jnp.mod(token_pos, 3) == 2).astype(jnp.float32)
    )
    return info


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


def reveal_order_sample_step(
    model: Any,
    rng: Array,
    i: int,
    timesteps: int,
    state: Array | tuple[Array, Array, Array],
    *,
    conditioning: Array | None = None,
    known_token_mask: Array | None = None,
    known_tokens: Array | None = None,
    method: str,
    return_info: bool = False,
) -> Array | tuple[Array, Array, Array] | tuple[Array | tuple[Array, Array, Array], dict[str, Array]]:
    rng_body = jax.random.fold_in(rng, i)
    rng_select, rng_sample = jax.random.split(rng_body)
    s, t = model.get_sampling_grid(i, timesteps)

    tokens, log_probs, structured = model._sampling_log_probs(
        state,
        t=t,
        conditioning=conditioning,
    )
    masked_unknown = masked_unknown_positions(
        tokens,
        mask_token_id=model.mask_token_id,
        known_token_mask=known_token_mask,
    )
    margins = position_probability_margins(log_probs)
    alpha_t = model.noise_schedule.alpha(t)
    alpha_s = model.noise_schedule.alpha(s)
    reveal_prob = (alpha_s - alpha_t) / (1.0 - alpha_t)

    if method == "uniform":
        reveal_positions = select_uniform_reveal_positions(
            rng_select,
            masked_unknown,
            reveal_prob=reveal_prob,
        )
    elif method == "top_probability":
        reveal_positions = select_top_probability_positions(
            log_probs,
            masked_unknown,
            reveal_prob=reveal_prob,
            rng=rng_select,
            oracle_noise_type=model.oracle_noise_type,
            oracle_noise_scale=model.oracle_noise_scale,
        )
    elif method == "top_prob_margin":
        reveal_positions = select_top_prob_margin_positions(
            log_probs,
            masked_unknown,
            reveal_prob=reveal_prob,
            rng=rng_select,
            oracle_noise_type=model.oracle_noise_type,
            oracle_noise_scale=model.oracle_noise_scale,
        )
    else:
        raise NotImplementedError(f"Unknown reveal-order method={method!r}")

    sample_mode = str(model.revealed_token_sample_mode).strip().lower()
    if sample_mode == "sample":
        proposal = categorical_sample_from_logits(
            rng_sample,
            log_probs,
            policy=model.categorical_sampling_policy,
        )
    elif sample_mode == "argmax":
        proposal = jnp.argmax(log_probs, axis=-1).astype(jnp.int32)
    else:
        raise ValueError(
            "Unknown revealed_token_sample_mode="
            f"{model.revealed_token_sample_mode!r}. Expected 'sample' or 'argmax'."
        )
    proposal = jnp.where(reveal_positions, proposal, model.mask_token_id).astype(jnp.int32)
    next_tokens = masked_core.carry_over_unmasked(
        tokens,
        proposal,
        mask_token_id=model.mask_token_id,
    )
    next_tokens = clamp_known_tokens(
        next_tokens,
        current_tokens=tokens,
        known_token_mask=known_token_mask,
        known_tokens=known_tokens,
    )

    cache_valid = jnp.asarray(model._use_cache()) & (~jnp.any(next_tokens != tokens))
    next_state = model._pack_sampling_state(
        next_tokens,
        log_probs=log_probs,
        cache_valid=cache_valid,
        structured=structured,
    )
    if not bool(return_info):
        return next_state
    return next_state, sampling_step_info(
        masked_unknown=masked_unknown,
        reveal_positions=reveal_positions,
        margins=margins,
    )


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
