from __future__ import annotations

# Phase 4 — slack-aware sampler: implemented in `conditional_generate_with_slack`
# below. The cell-only entry point (`conditional_generate`) is unchanged; the
# slack-augmented Sudoku SJD task (model.enable_joint_input=True) goes through
# the new function. Design summary:
#   * Maintain joint state (y_cells, y_slacks). At t=T both are init_std*N(0, I).
#   * At each step: forward pass with slack_y_t=y_slacks; cells follow the
#     classifier-induced reverse VP SDE on uncommitted sites and the existing
#     plugin/hazard policy for sticky commits; slacks follow the *unconditional*
#     reverse VP SDE every step (no anchors, no per-anchor mixture, clean mean
#     is the constant 1 vector — the data slack for any valid solution).
#   * Optional A2 cheap-projection fix: after the Euler step + cell commits,
#     overwrite slacks with `compute_slack_from_cells(soft_cell_state)`.
#   * Read out the predicted Sudoku from the cell sites only.

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp

from sticky.data.sudoku.slack import compute_slack_from_cells
from sticky.models.common.discrete_mixture import normalize_probs

from .anchors import AnchorTable, clamp_known_state
from .plugin_intensity import plugin_hazard_and_allocation
from .sampler import make_sampling_time_grid
from .sdes import _expand_like, alpha_sigma


Array = jnp.ndarray


_VALID_POLICIES = frozenset(
    {
        "linear_survival",
        "cosine_survival",
        "linear_topk_probability",
        "linear_topk_margin",
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
    if policy == "linear_topk_margin":
        top2 = jax.lax.top_k(choice_probs, k=2)[0]
        return top2[..., 0] - top2[..., 1]
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


def _predict_joint_cell_logits(
    *,
    model: Any,
    params: Any,
    y_cells: Array,
    y_slacks: Array,
    t_img: Array,
) -> Array:
    """Forward pass through the slack-augmented joint classifier and slice
    out the per-cell logits. The full classifier output has shape
    (B, 108, vocab_size); only the first 81 positions contribute to the
    cell argmax / score / policy loops below.
    """
    logits, _ = model.apply(
        {"params": params},
        y_cells,
        t=t_img,
        slack_y_t=y_slacks,
        train=False,
    )
    return logits[:, : int(y_cells.shape[1]), :].astype(jnp.float32)


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
    log_ratio_clip: float = 10.0,
    init_std: float = 1.0,
    stochastic_k: bool = False,
    eta: float | None = None,
    return_diagnostics: bool = False,
    tau_grid_size: int = 32,
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
        if policy in {"linear_survival", "cosine_survival", "linear_topk_probability", "linear_topk_margin"}:
            reveal_count = expected_reveal_counts(
                masked_unknown,
                alpha_t=jnp.asarray(alpha_progress_t, dtype=jnp.float32),
                alpha_s=jnp.asarray(alpha_progress_s, dtype=jnp.float32),
                stochastic_k=bool(stochastic_k),
                rng=count_key,
            )
            p_commit = None
        else:
            lam_total, plugin_probs = plugin_hazard_and_allocation(
                logits=logits,
                y=y,
                t_img=s_img,
                anchors=anchors,
                beta=beta,
                hazard=hazard,
                jump=jump_eff,
                logit_temperature=float(logit_temperature),
                log_ratio_clip=float(log_ratio_clip),
                tau_grid_size=int(tau_grid_size),
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


SLACK_SITE_COUNT = 27
SLACK_VOCAB = 9


def conditional_generate_with_slack(
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
    log_ratio_clip: float = 10.0,
    init_std: float = 1.0,
    slack_init_std: float | None = None,
    project_slacks_after_step: bool = True,
    stochastic_k: bool = False,
    eta: float | None = None,
    return_diagnostics: bool = False,
    tau_grid_size: int = 32,
) -> Array | tuple[Array, dict[str, Array]]:
    """Slack-aware predictor for the slack-augmented Sudoku SJD task.

    Mirrors `conditional_generate` for cells (clamped clue mask, classifier-
    induced reverse VP step on uncommitted cells, sticky-commit policies) and
    additionally maintains a slack state `y_slacks: (B, 27, 9)` that:

      * is initialized from `slack_init_std * N(0, I)` at t=T (the VP
        terminal), like cells.
      * follows the *unconditional* reverse VP SDE every step. The slack axis
        is unanchored — there is no per-anchor mixture to take an expectation
        over — so the conditional score reduces to the analytical VP score
        at the data mean, which is the all-ones vector for any valid Sudoku
        solution: `score = (alpha(t) * 1 - y_slacks) / sigma2(t)`.
      * never receives sticky jumps. Slacks have no committed/uncommitted
        distinction.
      * is optionally overwritten after each Euler+commit step with the
        deterministic group-sum readout from the current cell state — a
        cheap "tied-projection" approximation that re-couples slack to
        cells without paying for a joint un-sticking kernel (Phase D).

    The slack final value at t=0 is discarded; the predicted Sudoku is read
    out from the cell argmax exactly as in `conditional_generate`.
    """
    policy = normalize_policy_name(policy)
    known_tokens = jnp.asarray(known_tokens, dtype=jnp.int32)
    known_token_mask = jnp.asarray(known_token_mask, dtype=jnp.bool_)
    if known_tokens.shape != known_token_mask.shape:
        raise ValueError(
            "known_tokens and known_token_mask must have matching shapes, got "
            f"{known_tokens.shape} vs {known_token_mask.shape}."
        )

    batch_size = int(known_tokens.shape[0])
    cell_shape = tuple(int(dim) for dim in known_tokens.shape[1:])
    d = int(anchors.d)
    if d != SLACK_VOCAB:
        raise ValueError(
            "Slack-aware sampler requires anchor dim == 9 (simplex vertices); "
            f"got anchor.d={d}."
        )
    jump_eff = replace(jump, eta=float(eta)) if eta is not None else jump
    slack_init_std_eff = float(init_std if slack_init_std is None else slack_init_std)

    known_idx = puzzle_digits_to_clean_indices(known_tokens, known_mask=known_token_mask)
    time_grid = make_sampling_time_grid(
        T=float(getattr(beta, "T", 1.0)),
        n_steps=int(n_steps),
        sampling_grid=str(sampling_grid),
    )

    rng, init_cell_key, init_slack_key = jax.random.split(rng, 3)
    y_cells = float(init_std) * jax.random.normal(
        init_cell_key,
        shape=(batch_size,) + cell_shape + (d,),
        dtype=jnp.float32,
    )
    y_slacks = slack_init_std_eff * jax.random.normal(
        init_slack_key,
        shape=(batch_size, SLACK_SITE_COUNT, SLACK_VOCAB),
        dtype=jnp.float32,
    )

    committed_mask = jnp.asarray(known_token_mask, dtype=jnp.bool_)
    committed_idx = jnp.where(committed_mask, known_idx, -jnp.ones_like(known_idx))
    y_cells, committed_mask, committed_idx = clamp_known_state(
        y=y_cells,
        committed=committed_mask,
        k_idx=committed_idx,
        known_mask=known_token_mask,
        known_idx=known_idx,
        a_table=anchors.table_float,
    )

    slack_clean_mean = jnp.ones((SLACK_VOCAB,), dtype=jnp.float32)

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
        "slack_residual_mean_total": jnp.asarray(0.0, dtype=jnp.float32),
        "slack_residual_count_total": jnp.asarray(0.0, dtype=jnp.float32),
    }

    for i in range(int(n_steps)):
        t_scalar = time_grid[i]
        s_scalar = time_grid[i + 1]
        dt = jnp.maximum(t_scalar - s_scalar, 0.0)
        t_img = jnp.full((batch_size,), t_scalar, dtype=jnp.float32)

        # --- Score-time forward pass (joint) ---
        cell_logits_score = _predict_joint_cell_logits(
            model=model, params=params, y_cells=y_cells, y_slacks=y_slacks, t_img=t_img,
        )
        probs_score = normalize_probs(jax.nn.softmax(cell_logits_score, axis=-1))
        mu = _classifier_mean(probs=probs_score, anchor_table=anchors)

        alpha_t, sigma_t = alpha_sigma(beta, t_img)
        alpha_t_b = alpha_t[:, None, None]
        sigma2_t = jnp.maximum((sigma_t[:, None, None] ** 2), 1.0e-12)
        beta_t = beta(t_img)[:, None, None]

        # --- Cell SDE step on uncommitted sites ---
        cell_score = -(y_cells - alpha_t_b * mu) / sigma2_t
        cell_drift = (0.5 * beta_t) * y_cells + beta_t * cell_score

        rng, cell_noise_key, slack_noise_key, count_key, score_key = jax.random.split(rng, 5)
        cell_noise = jax.random.normal(cell_noise_key, shape=y_cells.shape, dtype=jnp.float32)
        uncommitted = (~committed_mask)[..., None].astype(jnp.float32)
        y_cells = y_cells + uncommitted * (
            cell_drift * dt + jnp.sqrt(jnp.maximum(beta_t * dt, 0.0)) * cell_noise
        )
        y_cells, committed_mask, committed_idx = clamp_known_state(
            y=y_cells,
            committed=committed_mask,
            k_idx=committed_idx,
            known_mask=known_token_mask,
            known_idx=known_idx,
            a_table=anchors.table_float,
        )

        # --- Slack VP step on every slack site ---
        # Slacks are unanchored. The reverse score is the unconditional VP
        # score at the data marginal: clean mean is the constant 1 vector
        # (every valid Sudoku has exactly one of each digit per group), so
        #   score = (alpha(t) * 1 - y_slacks) / sigma2(t).
        # Drift and Euler-Maruyama match the cell formula; no commit logic.
        slack_marginal_mean = alpha_t_b * slack_clean_mean[None, None, :]
        slack_score = (slack_marginal_mean - y_slacks) / sigma2_t
        slack_drift = (0.5 * beta_t) * y_slacks + beta_t * slack_score
        slack_noise_step = jax.random.normal(
            slack_noise_key, shape=y_slacks.shape, dtype=jnp.float32
        )
        y_slacks = y_slacks + slack_drift * dt + jnp.sqrt(
            jnp.maximum(beta_t * dt, 0.0)
        ) * slack_noise_step

        # --- Selection-time forward pass (joint, post Euler step) ---
        s_img = jnp.full((batch_size,), s_scalar, dtype=jnp.float32)
        cell_logits = _predict_joint_cell_logits(
            model=model, params=params, y_cells=y_cells, y_slacks=y_slacks, t_img=s_img,
        )
        choice_probs = normalize_probs(jax.nn.softmax(cell_logits, axis=-1))
        masked_unknown = ~committed_mask

        top_probability = jnp.max(choice_probs, axis=-1)
        top2 = jnp.sort(choice_probs, axis=-1)[..., -2:]
        top_prob_margin = top2[..., 1] - top2[..., 0]

        alpha_progress_t = _step_progress(t=t_scalar, T=float(getattr(beta, "T", 1.0)))
        alpha_progress_s = _step_progress(t=s_scalar, T=float(getattr(beta, "T", 1.0)))
        if policy in {
            "linear_survival",
            "cosine_survival",
            "linear_topk_probability",
            "linear_topk_margin",
        }:
            reveal_count = expected_reveal_counts(
                masked_unknown,
                alpha_t=jnp.asarray(alpha_progress_t, dtype=jnp.float32),
                alpha_s=jnp.asarray(alpha_progress_s, dtype=jnp.float32),
                stochastic_k=bool(stochastic_k),
                rng=count_key,
            )
            p_commit = None
        else:
            lam_total, plugin_probs = plugin_hazard_and_allocation(
                logits=cell_logits,
                y=y_cells,
                t_img=s_img,
                anchors=anchors,
                beta=beta,
                hazard=hazard,
                jump=jump_eff,
                logit_temperature=float(logit_temperature),
                log_ratio_clip=float(log_ratio_clip),
                tau_grid_size=int(tau_grid_size),
            )
            choice_probs = plugin_probs
            p_commit = 1.0 - jnp.exp(-jnp.maximum(lam_total, 0.0) * dt)
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
        y_cells = jnp.where(selected[..., None], proposal_vec, y_cells)
        y_cells, committed_mask, committed_idx = clamp_known_state(
            y=y_cells,
            committed=committed_mask,
            k_idx=committed_idx,
            known_mask=known_token_mask,
            known_idx=known_idx,
            a_table=anchors.table_float,
        )

        # --- A2 cheap projection: re-tie slacks to the post-commit cell state ---
        # For committed cells use the one-hot anchor; for uncommitted cells
        # use the classifier softmax (= classifier mean over simplex_vertex
        # anchors). The projection is the deterministic group-sum readout.
        if bool(project_slacks_after_step):
            soft_cells = jnp.where(
                committed_mask[..., None],
                anchors.table_float[committed_idx],
                choice_probs,
            )
            y_slacks = compute_slack_from_cells(soft_cells)

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
        slack_residual = jnp.linalg.norm(
            y_slacks - alpha_t_b * slack_clean_mean[None, None, :], axis=-1
        )
        diag["slack_residual_mean_total"] = (
            diag["slack_residual_mean_total"] + jnp.sum(slack_residual)
        )
        diag["slack_residual_count_total"] = (
            diag["slack_residual_count_total"]
            + jnp.asarray(slack_residual.size, dtype=jnp.float32)
        )

    final_masked_unknown = (~committed_mask) & (~known_token_mask)
    diag["final_masked_unknown_total"] = jnp.sum(final_masked_unknown.astype(jnp.float32))

    final_logits = _predict_joint_cell_logits(
        model=model,
        params=params,
        y_cells=y_cells,
        y_slacks=y_slacks,
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
