from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp

from sticky.core.sampling_loop import get_params
from sticky.models.common import masked_discrete_core as masked_core
from sticky.models.baselines.mdlm.sudoku_sampling import (
    clamp_known_tokens,
    masked_unknown_positions,
    position_probability_margins,
    position_top_probabilities,
    select_top_prob_margin_positions,
    select_top_probability_positions,
    select_uniform_reveal_positions,
)


Array = jnp.ndarray


def _normalize_sampler_method(method: str | None) -> str:
    key = str("uniform" if method is None else method).strip().lower()
    if key == "vanilla":
        return "uniform"
    if key not in {"uniform", "top_probability", "top_prob_margin"}:
        raise ValueError(
            f"Unknown MDM Sudoku sampler method={method!r}. "
            "Expected one of {'vanilla', 'uniform', 'top_probability', 'top_prob_margin'}."
        )
    return key


def _normalize_decoding_style(style: str | None) -> str:
    key = str("monotone_reveal" if style is None else style).strip().lower()
    if key not in {"monotone_reveal", "topk_remask"}:
        raise ValueError(
            f"Unknown MDM decoding_style={style!r}. "
            "Expected one of {'monotone_reveal', 'topk_remask'}."
        )
    return key


def _current_reverse_t(i: int, timesteps: int) -> Array:
    return jnp.asarray(timesteps, dtype=jnp.int32) - jnp.asarray(i, dtype=jnp.int32) - 1


def _reveal_probability(i: int, timesteps: int, batch_size: int) -> Array:
    current_t = _current_reverse_t(i, timesteps)
    reveal_prob = 1.0 / (current_t.astype(jnp.float32) + 1.0)
    return jnp.full((int(batch_size),), reveal_prob, dtype=jnp.float32)


def _linear_rate(i: int, timesteps: int) -> Array:
    current_t = _current_reverse_t(i, timesteps)
    return current_t.astype(jnp.float32) / float(timesteps)


def _sample_gumbel_noise(rng: Array, shape: tuple[int, ...]) -> Array:
    u = jax.random.uniform(
        rng,
        shape=shape,
        dtype=jnp.float32,
        minval=1.0e-6,
        maxval=1.0 - 1.0e-6,
    )
    return -jnp.log(-jnp.log(u))


def _response_positions(
    tokens: Array,
    *,
    known_token_mask: Array | None = None,
) -> Array:
    if known_token_mask is None:
        return jnp.ones_like(tokens, dtype=jnp.bool_)
    return ~jnp.asarray(known_token_mask, dtype=jnp.bool_)


def _remask_counts(
    candidate_mask: Array,
    *,
    rate: Array,
) -> Array:
    flat_mask = jnp.asarray(candidate_mask, dtype=jnp.bool_).reshape(
        (candidate_mask.shape[0], -1)
    )
    num_candidates = jnp.sum(flat_mask.astype(jnp.int32), axis=-1)
    raw = jnp.floor(
        num_candidates.astype(jnp.float32) * jnp.asarray(rate, dtype=jnp.float32)
    ).astype(jnp.int32)
    return jnp.clip(raw, 0, num_candidates)


def _select_low_confidence_positions(
    scores: Array,
    candidate_mask: Array,
    *,
    rate: Array,
    rng: Array | None = None,
    oracle_noise_type: str = "none",
    oracle_noise_scale: float = 0.0,
) -> Array:
    flat_scores = jnp.asarray(scores, dtype=jnp.float32).reshape((scores.shape[0], -1))
    flat_mask = jnp.asarray(candidate_mask, dtype=jnp.bool_).reshape(
        (candidate_mask.shape[0], -1)
    )
    k = _remask_counts(candidate_mask, rate=rate)

    key = str(oracle_noise_type).strip().lower()
    noisy_scores = flat_scores
    if key not in {"", "none"}:
        if key != "gumbel":
            raise ValueError(
                f"Unknown oracle_noise_type={oracle_noise_type!r}. "
                "Expected one of {'none', 'gumbel'}."
            )
        if rng is None:
            raise ValueError("Gumbel oracle noise requires an RNG key.")
        scaled_noise = jnp.asarray(oracle_noise_scale, dtype=jnp.float32) * jnp.asarray(
            rate, dtype=jnp.float32
        )
        noisy_scores = noisy_scores + scaled_noise * _sample_gumbel_noise(
            rng, tuple(noisy_scores.shape)
        )

    noisy_scores = jnp.where(flat_mask, noisy_scores, jnp.inf)
    order = jnp.argsort(noisy_scores, axis=-1)
    ranks = jnp.argsort(order, axis=-1)
    selected = flat_mask & (ranks < k[:, None])
    return selected.reshape(candidate_mask.shape)


def _zero_sampling_info(dtype: Any = jnp.float32) -> dict[str, Array]:
    zero = jnp.asarray(0.0, dtype=dtype)
    return {
        "masked_unknown_total": zero,
        "selected_count_total": zero,
        "selected_top_probability_sum_total": zero,
        "selected_top_probability_count_total": zero,
        "selected_top_prob_margin_sum_total": zero,
        "selected_top_prob_margin_count_total": zero,
        # Backward-compatible alias expected by the shared Sudoku evaluator.
        "selected_margin_sum_total": zero,
        "selected_margin_count_total": zero,
        "selected_row_total": zero,
        "selected_col_total": zero,
        "selected_value_total": zero,
        "selected_eos_total": zero,
    }


def _packed_selection_component_totals(
    selection_mask: Array,
    *,
    known_token_mask: Array | None = None,
) -> dict[str, Array]:
    selected = jnp.asarray(selection_mask, dtype=jnp.bool_)
    selected_f = selected.astype(jnp.float32)
    if known_token_mask is None:
        token_pos = jnp.arange(selected.shape[-1], dtype=jnp.int32)
        token_pos = token_pos.reshape((1,) * (selected.ndim - 1) + (-1,))
        return {
            "selected_row_total": jnp.sum(
                selected_f * (jnp.mod(token_pos, 3) == 0).astype(jnp.float32)
            ),
            "selected_col_total": jnp.sum(
                selected_f * (jnp.mod(token_pos, 3) == 1).astype(jnp.float32)
            ),
            "selected_value_total": jnp.sum(
                selected_f * (jnp.mod(token_pos, 3) == 2).astype(jnp.float32)
            ),
            "selected_eos_total": jnp.asarray(0.0, dtype=jnp.float32),
        }

    known_token_mask = jnp.asarray(known_token_mask, dtype=jnp.bool_)
    token_pos = jnp.arange(selected.shape[-1], dtype=jnp.int32)
    token_pos = token_pos.reshape((1,) * (selected.ndim - 1) + (-1,))
    response_start = jnp.sum(known_token_mask.astype(jnp.int32), axis=-1, keepdims=True)
    sep_index = response_start - 1
    eos_index = jnp.full_like(response_start, selected.shape[-1] - 1)
    is_sep = token_pos == sep_index
    is_eos = token_pos == eos_index
    original_pos = jnp.where(token_pos >= response_start, token_pos - 1, token_pos)
    content_mask = (~is_sep) & (~is_eos)
    return {
        "selected_row_total": jnp.sum(
            selected_f
            * content_mask.astype(jnp.float32)
            * (jnp.mod(original_pos, 3) == 0).astype(jnp.float32)
        ),
        "selected_col_total": jnp.sum(
            selected_f
            * content_mask.astype(jnp.float32)
            * (jnp.mod(original_pos, 3) == 1).astype(jnp.float32)
        ),
        "selected_value_total": jnp.sum(
            selected_f
            * content_mask.astype(jnp.float32)
            * (jnp.mod(original_pos, 3) == 2).astype(jnp.float32)
        ),
        "selected_eos_total": jnp.sum(selected_f * is_eos.astype(jnp.float32)),
    }


def sampling_step_info(
    *,
    masked_unknown: Array,
    reveal_positions: Array | None = None,
    top_probabilities: Array | None = None,
    margins: Array | None = None,
    known_token_mask: Array | None = None,
) -> dict[str, Array]:
    info = _zero_sampling_info()
    info["masked_unknown_total"] = jnp.sum(jnp.asarray(masked_unknown, dtype=jnp.float32))
    if reveal_positions is None:
        return info

    reveal_positions = jnp.asarray(reveal_positions, dtype=jnp.bool_)
    selected = reveal_positions.astype(jnp.float32)
    info["selected_count_total"] = jnp.sum(selected)
    info.update(
        _packed_selection_component_totals(
            reveal_positions,
            known_token_mask=known_token_mask,
        )
    )

    if top_probabilities is not None:
        top_probabilities = jnp.asarray(top_probabilities, dtype=jnp.float32)
        info["selected_top_probability_sum_total"] = jnp.sum(top_probabilities * selected)
        info["selected_top_probability_count_total"] = info["selected_count_total"]

    if margins is not None:
        margins = jnp.asarray(margins, dtype=jnp.float32)
        margin_sum = jnp.sum(margins * selected)
        info["selected_top_prob_margin_sum_total"] = margin_sum
        info["selected_top_prob_margin_count_total"] = info["selected_count_total"]
        info["selected_margin_sum_total"] = margin_sum
        info["selected_margin_count_total"] = info["selected_count_total"]

    return info


def reveal_order_sample_step(
    model: Any,
    rng: Array,
    i: int,
    timesteps: int,
    tokens: Array,
    *,
    conditioning: Array | None = None,
    known_token_mask: Array | None = None,
    known_tokens: Array | None = None,
    method: str | None = None,
    return_info: bool = False,
) -> Array | tuple[Array, dict[str, Array]]:
    del conditioning
    method = _normalize_sampler_method(method)
    decoding_style = _normalize_decoding_style(
        getattr(model, "decoding_style", "monotone_reveal")
    )
    tokens = jnp.asarray(tokens, dtype=jnp.int32)

    rng_body = jax.random.fold_in(rng, i)
    batch_size = int(tokens.shape[0])
    reveal_prob = _reveal_probability(i, timesteps, batch_size)
    current_t = _current_reverse_t(i, timesteps)
    response_positions = _response_positions(tokens, known_token_mask=known_token_mask)

    logits = model.predict_logits(
        tokens,
        jnp.full((batch_size,), current_t, dtype=jnp.float32),
        train=False,
    )
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    masked_unknown = masked_unknown_positions(
        tokens,
        mask_token_id=model.mask_token_id,
        known_token_mask=known_token_mask,
    )
    top_probabilities = position_top_probabilities(log_probs)
    margins = position_probability_margins(log_probs)

    proposal = jnp.argmax(log_probs, axis=-1).astype(jnp.int32)

    if method == "uniform" or decoding_style == "monotone_reveal":
        if method == "uniform":
            selected_positions = select_uniform_reveal_positions(
                rng_body,
                masked_unknown,
                reveal_prob=reveal_prob,
            )
        elif method == "top_probability":
            selected_positions = select_top_probability_positions(
                log_probs,
                masked_unknown,
                reveal_prob=reveal_prob,
                rng=rng_body,
                oracle_noise_type=model.oracle_noise_type,
                oracle_noise_scale=model.oracle_noise_scale,
            )
        else:
            selected_positions = select_top_prob_margin_positions(
                log_probs,
                masked_unknown,
                reveal_prob=reveal_prob,
                rng=rng_body,
                oracle_noise_type=model.oracle_noise_type,
                oracle_noise_scale=model.oracle_noise_scale,
            )

        # Monotone reveal keeps already revealed response tokens fixed and only
        # fills newly selected masked slots with argmax token values.
        proposal_masked = jnp.where(
            selected_positions, proposal, model.mask_token_id
        ).astype(jnp.int32)
        next_tokens = masked_core.carry_over_unmasked(
            tokens,
            proposal_masked,
            mask_token_id=model.mask_token_id,
        )
    else:
        # Ye-style top-k remasking predicts argmax token values for the whole
        # response suffix, then re-masks the lowest-confidence response slots
        # according to the current linear schedule. Already revealed response
        # tokens may therefore be masked again.
        rate = _linear_rate(i, timesteps)
        score_source = top_probabilities if method == "top_probability" else margins
        selected_positions = _select_low_confidence_positions(
            score_source,
            response_positions,
            rate=rate,
            rng=rng_body,
            oracle_noise_type=model.oracle_noise_type,
            oracle_noise_scale=model.oracle_noise_scale,
        )
        next_tokens = jnp.where(selected_positions, model.mask_token_id, proposal).astype(
            jnp.int32
        )

    next_tokens = clamp_known_tokens(
        next_tokens,
        current_tokens=tokens,
        known_token_mask=known_token_mask,
        known_tokens=known_tokens,
    )

    if not bool(return_info):
        return next_tokens
    return next_tokens, sampling_step_info(
        masked_unknown=masked_unknown,
        reveal_positions=selected_positions,
        top_probabilities=top_probabilities,
        margins=margins,
        known_token_mask=known_token_mask,
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
    sampler_override: str | None = None,
) -> Array | tuple[Array, dict[str, Array]]:
    params = get_params(train_state, use_ema=use_ema)
    variables = {"params": params}

    known_tokens = jnp.asarray(known_tokens, dtype=jnp.int32)
    known_token_mask = jnp.asarray(known_token_mask, dtype=jnp.bool_)
    if known_tokens.shape != known_token_mask.shape:
        raise ValueError(
            "known_tokens and known_token_mask must have matching shapes, got "
            f"{known_tokens.shape} vs {known_token_mask.shape}."
        )

    total_steps = int(model.timesteps if timesteps is None else timesteps)
    sampler_method = _normalize_sampler_method(
        model.sampler if sampler_override is None else sampler_override
    )
    prior_tokens = model.apply(variables, int(known_tokens.shape[0]), method=model.prior_sample)
    state = jnp.where(known_token_mask, known_tokens, prior_tokens).astype(jnp.int32)
    rng, step_rng = jax.random.split(rng)

    if not bool(return_diagnostics):

        def body_fn(i, st):
            return model.apply(
                variables,
                step_rng,
                i,
                total_steps,
                st,
                conditioning=conditioning,
                known_tokens=known_tokens,
                known_token_mask=known_token_mask,
                sampler_override=sampler_method,
                method=model.sample_step,
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
        "selected_top_probability_sum_total": zero,
        "selected_top_probability_count_total": zero,
        "selected_top_prob_margin_sum_total": zero,
        "selected_top_prob_margin_count_total": zero,
        "selected_margin_sum_total": zero,
        "selected_margin_count_total": zero,
        "selected_row_total_across_steps": zero,
        "selected_col_total_across_steps": zero,
        "selected_value_total_across_steps": zero,
        "selected_eos_total_across_steps": zero,
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
            sampler_override=sampler_method,
            return_info=True,
            method=model.sample_step,
        )
        diag = {
            "masked_unknown_total_across_steps": (
                diag["masked_unknown_total_across_steps"] + step_info["masked_unknown_total"]
            ),
            "selected_count_total_across_steps": (
                diag["selected_count_total_across_steps"] + step_info["selected_count_total"]
            ),
            "selected_top_probability_sum_total": (
                diag["selected_top_probability_sum_total"]
                + step_info["selected_top_probability_sum_total"]
            ),
            "selected_top_probability_count_total": (
                diag["selected_top_probability_count_total"]
                + step_info["selected_top_probability_count_total"]
            ),
            "selected_top_prob_margin_sum_total": (
                diag["selected_top_prob_margin_sum_total"]
                + step_info["selected_top_prob_margin_sum_total"]
            ),
            "selected_top_prob_margin_count_total": (
                diag["selected_top_prob_margin_count_total"]
                + step_info["selected_top_prob_margin_count_total"]
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
            "selected_eos_total_across_steps": (
                diag["selected_eos_total_across_steps"] + step_info["selected_eos_total"]
            ),
        }
        return next_state, diag

    state, diag = jax.lax.fori_loop(0, total_steps, body_fn, (state, diag0))
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
        ((state == int(model.mask_token_id)) & unknown_token_mask).astype(jnp.float32)
    )
    return decoded, diag
