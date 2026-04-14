from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp

from sticky.models.common.discrete_mixture import normalize_probs

from .anchors import AnchorTable, clamp_known_state
from .plugin_intensity import plugin_intensity_and_probs
from .sampler import make_sampling_time_grid
from .sdes import _expand_like, alpha_sigma


Array = jnp.ndarray


_VALID_POLICIES = frozenset(
    {
        "linear_survival",
        "cosine_survival",
        "linear_topk_probability",
        "plugin_hazard",
    }
)


def normalize_policy_name(policy: str | None) -> str:
    key = str("linear_survival" if policy is None else policy).strip().lower()
    if key not in _VALID_POLICIES:
        valid = ", ".join(sorted(_VALID_POLICIES))
        raise ValueError(f"Unknown SJD Sudoku decoding policy={policy!r}. Expected one of: {valid}.")
    return key


def puzzle_digits_to_clean_indices(
    digits: Array,
    *,
    known_mask: Array | None = None,
) -> Array:
    digits = jnp.asarray(digits, dtype=jnp.int32)
    if known_mask is None:
        return jnp.clip(digits - 1, 0, 8)
    mask = jnp.asarray(known_mask, dtype=jnp.bool_)
    if mask.shape != digits.shape:
        raise ValueError(
            "known_mask must match puzzle digit shape, got "
            f"{mask.shape} vs {digits.shape}."
        )
    return jnp.where(mask, jnp.clip(digits - 1, 0, 8), 0).astype(jnp.int32)


def clean_indices_to_board_digits(indices: Array) -> Array:
    return (jnp.asarray(indices, dtype=jnp.int32) + 1).astype(jnp.int32)


def expected_reveal_counts(
    masked_unknown_mask: Array,
    *,
    alpha_t: Array | float,
    alpha_s: Array | float,
    stochastic_k: bool = False,
    rng: Array | None = None,
) -> Array:
    masked_unknown_mask = jnp.asarray(masked_unknown_mask, dtype=jnp.bool_)
    flat_mask = masked_unknown_mask.reshape((masked_unknown_mask.shape[0], -1))
    masked_count = jnp.sum(flat_mask.astype(jnp.int32), axis=-1)

    alpha_t = jnp.asarray(alpha_t, dtype=jnp.float32)
    alpha_s = jnp.asarray(alpha_s, dtype=jnp.float32)
    alpha_t = jnp.broadcast_to(_expand_like(alpha_t, masked_count), masked_count.shape)
    alpha_s = jnp.broadcast_to(_expand_like(alpha_s, masked_count), masked_count.shape)

    delta = jnp.clip(alpha_s - alpha_t, 0.0, 1.0)
    denom = jnp.maximum(1.0 - alpha_t, 1.0e-6)
    expected = masked_count.astype(jnp.float32) * delta / denom
    expected = jnp.clip(expected, 0.0, masked_count.astype(jnp.float32))
    if stochastic_k:
        if rng is None:
            raise ValueError("stochastic_k=True requires an RNG key.")
        frac = jnp.where(
            masked_count > 0,
            expected / jnp.maximum(masked_count.astype(jnp.float32), 1.0),
            0.0,
        )
        bern = jax.random.bernoulli(
            rng,
            p=jnp.clip(frac[:, None], 0.0, 1.0),
            shape=flat_mask.shape,
        )
        counts = jnp.sum(bern & flat_mask, axis=-1)
    else:
        counts = jnp.rint(expected).astype(jnp.int32)
    return jnp.clip(counts, 0, masked_count)


def _topk_mask(scores: Array, *, valid_mask: Array, k: Array) -> Array:
    valid_mask = jnp.asarray(valid_mask, dtype=jnp.bool_)
    scores = jnp.where(valid_mask, jnp.asarray(scores, dtype=jnp.float32), -jnp.inf)
    order = jnp.argsort(scores, axis=-1)[:, ::-1]
    ranks = jnp.argsort(order, axis=-1)
    return valid_mask & (ranks < k[:, None])


def _selection_scores_from_policy(
    *,
    rng: Array,
    policy: str,
    masked_unknown_mask: Array,
    choice_probs: Array,
    p_commit: Array | None,
) -> Array:
    if policy in {"linear_survival", "cosine_survival"}:
        return jax.random.uniform(rng, masked_unknown_mask.shape, dtype=jnp.float32)
    if policy == "linear_topk_probability":
        return jnp.max(choice_probs, axis=-1)
    if policy == "plugin_hazard":
        if p_commit is None:
            raise ValueError("plugin_hazard requires p_commit scores.")
        return jnp.asarray(p_commit, dtype=jnp.float32)
    raise ValueError(f"Unsupported policy={policy!r}.")


def _selection_mask_from_scores(
    *,
    scores: Array,
    masked_unknown_mask: Array,
    reveal_count: Array,
) -> Array:
    flat_mask = jnp.asarray(masked_unknown_mask, dtype=jnp.bool_).reshape(
        (masked_unknown_mask.shape[0], -1)
    )
    flat_scores = jnp.asarray(scores, dtype=jnp.float32).reshape(flat_mask.shape)
    selected = _topk_mask(flat_scores, valid_mask=flat_mask, k=jnp.asarray(reveal_count, dtype=jnp.int32))
    return selected.reshape(masked_unknown_mask.shape)


def _predict_logits(
    *,
    model: Any,
    params: Any,
    y: Array,
    t_img: Array,
) -> Array:
    logits, _ = model.apply({"params": params}, y, t=t_img, train=False)
    return logits.astype(jnp.float32)


def _classifier_mean(
    *,
    probs: Array,
    anchor_table: AnchorTable,
) -> Array:
    flat_probs = probs.reshape((-1, probs.shape[-1]))
    mu = flat_probs @ jnp.asarray(anchor_table.table_float, dtype=jnp.float32)
    return mu.reshape(probs.shape[:-1] + (anchor_table.d,))


def _step_progress(*, t: Array, T: float) -> Array:
    return jnp.clip(1.0 - (jnp.asarray(t, dtype=jnp.float32) / jnp.asarray(float(T), dtype=jnp.float32)), 0.0, 1.0)


def conditional_generate(
    rng: Array,
    *,
    params: Any,
    model: Any,
    anchors: AnchorTable,
    beta: Any,
    hazard: Any,
    jump: Any,
    known_tokens: Array,
    known_token_mask: Array,
    n_steps: int,
    policy: str,
    sampling_grid: str,
    logit_temperature: float = 1.0,
    intensity_mode: str = "full",
    log_ratio_clip: float = 10.0,
    init_std: float = 1.0,
    stochastic_k: bool = False,
    eta: float | None = None,
    return_diagnostics: bool = False,
) -> Array | tuple[Array, dict[str, Array]]:
    policy = normalize_policy_name(policy)
    known_tokens = jnp.asarray(known_tokens, dtype=jnp.int32)
    known_token_mask = jnp.asarray(known_token_mask, dtype=jnp.bool_)
    if known_tokens.shape != known_token_mask.shape:
        raise ValueError(
            "known_tokens and known_token_mask must have matching shapes, got "
            f"{known_tokens.shape} vs {known_token_mask.shape}."
        )

    batch_size = int(known_tokens.shape[0])
    shape = tuple(int(dim) for dim in known_tokens.shape[1:])
    d = int(anchors.d)
    jump_eff = replace(jump, eta=float(eta)) if eta is not None else jump

    known_idx = puzzle_digits_to_clean_indices(known_tokens, known_mask=known_token_mask)
    time_grid = make_sampling_time_grid(
        T=float(getattr(beta, "T", 1.0)),
        n_steps=int(n_steps),
        sampling_grid=str(sampling_grid),
    )

    rng, init_key = jax.random.split(rng)
    y = float(init_std) * jax.random.normal(
        init_key,
        shape=(batch_size,) + shape + (d,),
        dtype=jnp.float32,
    )
    committed_mask = jnp.asarray(known_token_mask, dtype=jnp.bool_)
    committed_idx = jnp.where(committed_mask, known_idx, -jnp.ones_like(known_idx))
    y, committed_mask, committed_idx = clamp_known_state(
        y=y,
        committed=committed_mask,
        k_idx=committed_idx,
        known_mask=known_token_mask,
        known_idx=known_idx,
        a_table=anchors.table_float,
    )

    diag = {
        "example_step_count": jnp.asarray(0.0, dtype=jnp.float32),
        "masked_unknown_total_across_steps": jnp.asarray(0.0, dtype=jnp.float32),
        "selected_count_total_across_steps": jnp.asarray(0.0, dtype=jnp.float32),
        "selected_top_probability_sum_total": jnp.asarray(0.0, dtype=jnp.float32),
        "selected_top_probability_count_total": jnp.asarray(0.0, dtype=jnp.float32),
        "selected_top_prob_margin_sum_total": jnp.asarray(0.0, dtype=jnp.float32),
        "selected_top_prob_margin_count_total": jnp.asarray(0.0, dtype=jnp.float32),
        "unknown_token_total": jnp.sum((~known_token_mask).astype(jnp.float32)),
        "final_masked_unknown_total": jnp.asarray(0.0, dtype=jnp.float32),
    }

    for i in range(int(n_steps)):
        t_scalar = time_grid[i]
        s_scalar = time_grid[i + 1]
        dt = jnp.maximum(t_scalar - s_scalar, 0.0)
        t_img = jnp.full((batch_size,), t_scalar, dtype=jnp.float32)

        logits_score = _predict_logits(model=model, params=params, y=y, t_img=t_img)
        probs_score = normalize_probs(jax.nn.softmax(logits_score, axis=-1))
        mu = _classifier_mean(probs=probs_score, anchor_table=anchors)

        alpha_t, sigma_t = alpha_sigma(beta, t_img)
        alpha_t = alpha_t[:, None, None]
        sigma2_t = jnp.maximum((sigma_t[:, None, None] ** 2), 1.0e-12)
        beta_t = beta(t_img)[:, None, None]
        score = -(y - alpha_t * mu) / sigma2_t
        drift = (0.5 * beta_t) * y + beta_t * score

        rng, noise_key, count_key, score_key = jax.random.split(rng, 4)
        noise = jax.random.normal(noise_key, shape=y.shape, dtype=jnp.float32)
        uncommitted = (~committed_mask)[..., None].astype(jnp.float32)
        y = y + uncommitted * (drift * dt + jnp.sqrt(jnp.maximum(beta_t * dt, 0.0)) * noise)
        y, committed_mask, committed_idx = clamp_known_state(
            y=y,
            committed=committed_mask,
            k_idx=committed_idx,
            known_mask=known_token_mask,
            known_idx=known_idx,
            a_table=anchors.table_float,
        )

        s_img = jnp.full((batch_size,), s_scalar, dtype=jnp.float32)
        logits = _predict_logits(model=model, params=params, y=y, t_img=s_img)
        choice_probs = normalize_probs(jax.nn.softmax(logits, axis=-1))
        masked_unknown = ~committed_mask

        top_probability = jnp.max(choice_probs, axis=-1)
        top2 = jnp.sort(choice_probs, axis=-1)[..., -2:]
        top_prob_margin = top2[..., 1] - top2[..., 0]

        alpha_progress_t = _step_progress(t=t_scalar, T=float(getattr(beta, "T", 1.0)))
        alpha_progress_s = _step_progress(t=s_scalar, T=float(getattr(beta, "T", 1.0)))
        if policy in {"linear_survival", "cosine_survival", "linear_topk_probability"}:
            reveal_count = expected_reveal_counts(
                masked_unknown,
                alpha_t=jnp.asarray(alpha_progress_t, dtype=jnp.float32),
                alpha_s=jnp.asarray(alpha_progress_s, dtype=jnp.float32),
                stochastic_k=bool(stochastic_k),
                rng=count_key,
            )
            p_commit = None
        else:
            lam_total, plugin_probs = plugin_intensity_and_probs(
                logits=logits,
                y=y,
                t_img=s_img,
                anchors=anchors,
                beta=beta,
                hazard=hazard,
                jump=jump_eff,
                logit_temperature=float(logit_temperature),
                intensity_mode=str(intensity_mode),
                log_ratio_clip=float(log_ratio_clip),
            )
            choice_probs = plugin_probs
            p_commit = 1.0 - jnp.exp(-jnp.maximum(lam_total, 0.0) * dt)
            # Even at eta=1 the policy ablation should remain a true plug-in
            # hazard policy with state-dependent site scores. Do not collapse to a
            # time-only surrogate here.
            reveal_count = jnp.rint(
                jnp.sum(jnp.where(masked_unknown, p_commit, 0.0), axis=-1)
            ).astype(jnp.int32)
            reveal_count = jnp.clip(
                reveal_count,
                0,
                jnp.sum(masked_unknown.astype(jnp.int32), axis=-1),
            )

        selection_scores = _selection_scores_from_policy(
            rng=score_key,
            policy=policy,
            masked_unknown_mask=masked_unknown,
            choice_probs=choice_probs,
            p_commit=p_commit,
        )
        selected = _selection_mask_from_scores(
            scores=selection_scores,
            masked_unknown_mask=masked_unknown,
            reveal_count=reveal_count,
        )

        proposal_idx = jnp.argmax(choice_probs, axis=-1).astype(jnp.int32)
        proposal_vec = anchors.table_float[proposal_idx]
        committed_idx = jnp.where(selected, proposal_idx, committed_idx)
        committed_mask = committed_mask | selected
        y = jnp.where(selected[..., None], proposal_vec, y)
        y, committed_mask, committed_idx = clamp_known_state(
            y=y,
            committed=committed_mask,
            k_idx=committed_idx,
            known_mask=known_token_mask,
            known_idx=known_idx,
            a_table=anchors.table_float,
        )

        diag["example_step_count"] = diag["example_step_count"] + jnp.asarray(batch_size, dtype=jnp.float32)
        diag["masked_unknown_total_across_steps"] = (
            diag["masked_unknown_total_across_steps"]
            + jnp.sum(masked_unknown.astype(jnp.float32))
        )
        diag["selected_count_total_across_steps"] = (
            diag["selected_count_total_across_steps"] + jnp.sum(selected.astype(jnp.float32))
        )
        diag["selected_top_probability_sum_total"] = (
            diag["selected_top_probability_sum_total"]
            + jnp.sum(top_probability * selected.astype(jnp.float32))
        )
        diag["selected_top_probability_count_total"] = (
            diag["selected_top_probability_count_total"] + jnp.sum(selected.astype(jnp.float32))
        )
        diag["selected_top_prob_margin_sum_total"] = (
            diag["selected_top_prob_margin_sum_total"]
            + jnp.sum(top_prob_margin * selected.astype(jnp.float32))
        )
        diag["selected_top_prob_margin_count_total"] = (
            diag["selected_top_prob_margin_count_total"] + jnp.sum(selected.astype(jnp.float32))
        )

    # Record unresolved unknowns before the terminal argmax fill used only for
    # producing a board-valued output.
    final_masked_unknown = (~committed_mask) & (~known_token_mask)
    diag["final_masked_unknown_total"] = jnp.sum(final_masked_unknown.astype(jnp.float32))

    final_logits = _predict_logits(
        model=model,
        params=params,
        y=y,
        t_img=jnp.zeros((batch_size,), dtype=jnp.float32),
    )
    final_idx = jnp.argmax(final_logits, axis=-1).astype(jnp.int32)
    committed_idx = jnp.where(committed_mask, committed_idx, final_idx)
    committed_mask = jnp.ones_like(committed_mask, dtype=jnp.bool_)
    pred_digits = clean_indices_to_board_digits(committed_idx)
    pred_digits = jnp.where(known_token_mask, known_tokens, pred_digits).astype(jnp.int32)

    if not bool(return_diagnostics):
        return pred_digits
    return pred_digits, diag
