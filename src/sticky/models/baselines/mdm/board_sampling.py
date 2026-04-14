from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp

from sticky.core.sampling_loop import get_params


Array = jnp.ndarray


def _normalize_sampler_method(method: str | None) -> str:
    key = str("vanilla" if method is None else method).strip().lower()
    if key == "uniform":
        key = "vanilla"
    if key not in {"vanilla", "top_probability", "top_prob_margin"}:
        raise ValueError(
            f"Unknown MDM inpaint sampler method={method!r}. "
            "Expected one of {'vanilla', 'top_probability', 'top_prob_margin'}."
        )
    return key


def _normalize_known_mask(
    known_token_mask: Array | None,
    *,
    ref: Array,
) -> Array:
    if known_token_mask is None:
        return jnp.zeros_like(ref, dtype=jnp.bool_)
    mask = jnp.asarray(known_token_mask, dtype=jnp.bool_)
    if mask.shape != ref.shape:
        raise ValueError(
            "known_token_mask must match token shape, got "
            f"{mask.shape} vs {ref.shape}."
        )
    return mask


def clamp_known_tokens(
    tokens: Array,
    *,
    current_tokens: Array,
    known_token_mask: Array | None = None,
    known_tokens: Array | None = None,
) -> Array:
    known_token_mask = _normalize_known_mask(known_token_mask, ref=tokens)
    if known_tokens is None:
        source = jnp.asarray(current_tokens, dtype=jnp.int32)
    else:
        source = jnp.asarray(known_tokens, dtype=jnp.int32)
        if source.shape != tokens.shape:
            raise ValueError(
                "known_tokens must match token shape, got "
                f"{source.shape} vs {tokens.shape}."
            )
    return jnp.where(known_token_mask, source, tokens).astype(jnp.int32)


def masked_unknown_positions(
    tokens: Array,
    *,
    known_token_mask: Array | None = None,
    mask_token_id: int = 0,
) -> Array:
    known_token_mask = _normalize_known_mask(known_token_mask, ref=tokens)
    return (jnp.asarray(tokens, dtype=jnp.int32) == int(mask_token_id)) & (~known_token_mask)


def _valid_digit_probs(logits: Array) -> Array:
    probs = jax.nn.softmax(jnp.asarray(logits, dtype=jnp.float32), axis=-1)
    if int(probs.shape[-1]) < 10:
        raise ValueError(
            f"Sudoku inpaint logits must include tokens 0..9, got shape {probs.shape}."
        )
    return probs[..., 1:10]


def position_top_probabilities(logits: Array) -> Array:
    valid_digit_probs = _valid_digit_probs(logits)
    return jnp.max(valid_digit_probs, axis=-1)


def position_probability_margins(logits: Array) -> Array:
    valid_digit_probs = _valid_digit_probs(logits)
    top2 = jnp.sort(valid_digit_probs, axis=-1)[..., -2:]
    return top2[..., 1] - top2[..., 0]


def _proposal_digits(logits: Array) -> Array:
    return (jnp.argmax(_valid_digit_probs(logits), axis=-1) + 1).astype(jnp.int32)


def _reveal_counts(masked_unknown_mask: Array, *, remaining_steps: Array | int) -> Array:
    flat_mask = jnp.asarray(masked_unknown_mask, dtype=jnp.bool_).reshape(
        (masked_unknown_mask.shape[0], -1)
    )
    num_masked = jnp.sum(flat_mask.astype(jnp.int32), axis=-1)
    remaining_steps = jnp.maximum(jnp.asarray(remaining_steps, dtype=jnp.int32), 1)
    counts = (num_masked + remaining_steps - 1) // remaining_steps
    return jnp.where(num_masked > 0, jnp.maximum(counts, 1), 0)


def _topk_mask_from_scores(scores: Array, *, k: Array, valid_mask: Array) -> Array:
    order = jnp.argsort(scores, axis=-1)[:, ::-1]
    ranks = jnp.argsort(order, axis=-1)
    return valid_mask & (ranks < k[:, None])


def select_reveal_positions(
    rng: Array,
    masked_unknown_mask: Array,
    *,
    method: str,
    top_probabilities: Array,
    margins: Array,
    remaining_steps: Array | int,
) -> Array:
    flat_mask = jnp.asarray(masked_unknown_mask, dtype=jnp.bool_).reshape(
        (masked_unknown_mask.shape[0], -1)
    )
    k = _reveal_counts(masked_unknown_mask, remaining_steps=remaining_steps)

    if str(method) == "vanilla":
        scores = jax.random.uniform(rng, flat_mask.shape, dtype=jnp.float32)
    elif str(method) == "top_probability":
        scores = jnp.asarray(top_probabilities, dtype=jnp.float32).reshape(flat_mask.shape)
    else:
        scores = jnp.asarray(margins, dtype=jnp.float32).reshape(flat_mask.shape)

    scores = jnp.where(flat_mask, scores, -jnp.inf)
    selected = _topk_mask_from_scores(scores, k=k, valid_mask=flat_mask)
    return selected.reshape(masked_unknown_mask.shape)


def sampling_step_info(
    *,
    masked_unknown: Array,
    reveal_positions: Array | None = None,
    top_probabilities: Array | None = None,
    margins: Array | None = None,
) -> dict[str, Array]:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    info = {
        "masked_unknown_total": jnp.sum(jnp.asarray(masked_unknown, dtype=jnp.float32)),
        "selected_count_total": zero,
        "selected_top_probability_sum_total": zero,
        "selected_top_probability_count_total": zero,
        "selected_top_prob_margin_sum_total": zero,
        "selected_top_prob_margin_count_total": zero,
    }
    if reveal_positions is None:
        return info

    reveal_positions = jnp.asarray(reveal_positions, dtype=jnp.bool_)
    selected = reveal_positions.astype(jnp.float32)
    info["selected_count_total"] = jnp.sum(selected)
    if top_probabilities is not None:
        info["selected_top_probability_sum_total"] = jnp.sum(
            jnp.asarray(top_probabilities, dtype=jnp.float32) * selected
        )
        info["selected_top_probability_count_total"] = info["selected_count_total"]
    if margins is not None:
        info["selected_top_prob_margin_sum_total"] = jnp.sum(
            jnp.asarray(margins, dtype=jnp.float32) * selected
        )
        info["selected_top_prob_margin_count_total"] = info["selected_count_total"]
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
    method = _normalize_sampler_method(method)
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    current_t = (
        jnp.asarray(timesteps, dtype=jnp.float32)
        - jnp.asarray(i, dtype=jnp.float32)
        - 1.0
    )
    batch_size = int(tokens.shape[0])
    known_token_mask = _normalize_known_mask(known_token_mask, ref=tokens)

    logits = model.predict_logits(
        tokens,
        jnp.full((batch_size,), current_t, dtype=jnp.float32),
        cond=conditioning,
        train=False,
    )
    top_probabilities = position_top_probabilities(logits)
    margins = position_probability_margins(logits)
    masked_unknown = masked_unknown_positions(
        tokens,
        known_token_mask=known_token_mask,
        mask_token_id=int(model.mask_token_id),
    )
    remaining_steps = jnp.asarray(timesteps, dtype=jnp.int32) - jnp.asarray(i, dtype=jnp.int32)
    selected_positions = select_reveal_positions(
        rng,
        masked_unknown,
        method=method,
        top_probabilities=top_probabilities,
        margins=margins,
        remaining_steps=remaining_steps,
    )
    proposal = _proposal_digits(logits)
    next_tokens = jnp.where(selected_positions, proposal, tokens).astype(jnp.int32)
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
    prior_tokens = model.apply(
        variables,
        int(known_tokens.shape[0]),
        method=model.prior_sample,
    )
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
