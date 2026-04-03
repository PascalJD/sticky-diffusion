from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

import sticky.models.baselines.mdlm.sudoku_sampling as sudoku_sampling_mod
from sticky.models.baselines.mdlm.mdlm_model import MDLM, _selected_log_prob_sums
from sticky.models.baselines.mdlm.sudoku_sampling import (
    conditional_generate,
    select_top_prob_margin_positions,
    select_top_probability_positions,
)


def _make_sequence_model(
    *,
    seq_len: int = 12,
    sampler: str = "uniform",
    noise_schedule_type: str = "loglinear",
    oracle_noise_type: str = "none",
    oracle_noise_scale: float = 0.0,
    feature_dim: int = 8,
    num_heads: int = 2,
    n_layers: int = 1,
    mlp_hidden_dim: int = 32,
    timesteps: int = 4,
) -> MDLM:
    return MDLM(
        data_shape=(seq_len,),
        cont_time=True,
        timesteps=timesteps,
        feature_dim=feature_dim,
        num_heads=num_heads,
        antithetic_time_sampling=False,
        n_layers=n_layers,
        n_dit_layers=0,
        dit_num_heads=num_heads,
        dit_hidden_size=feature_dim * num_heads,
        ch_mult=(1,),
        vocab_size=10,
        noise_schedule_type=noise_schedule_type,
        dropout_rate=0.0,
        use_attn_dropout=False,
        mlp_type="gelu",
        depth_scaled_init=False,
        cond_type="adaln",
        outside_embed=False,
        sequence_backbone="gpt2_like",
        sequence_mlp_hidden_dim=mlp_hidden_dim,
        sequence_max_length=seq_len,
        sequence_causal=False,
        image_backbone="auto",
        time_features="none",
        classes=-1,
        sampler=sampler,
        sampling_grid="loglinear",
        topp=0.95,
        oracle_noise_type=oracle_noise_type,
        oracle_noise_scale=oracle_noise_scale,
        categorical_sampling_policy="exact",
        cache_predictions=False,
        model_sharding=False,
    )


def _canonical_sudoku_sequence() -> jnp.ndarray:
    triples = []
    for r in range(9):
        for c in range(9):
            v = ((r * 3 + (r // 3) + c) % 9) + 1
            triples.extend((r, c, v))
    return jnp.asarray(triples, dtype=jnp.int32)


def test_forward_sample_conditional_preserves_known_tokens():
    model = _make_sequence_model()
    x = (jnp.arange(12, dtype=jnp.int32)[None, :] * 3) % 10
    t = jnp.asarray([0.65], dtype=jnp.float32)
    known_token_mask = jnp.arange(12, dtype=jnp.int32)[None, :] < 4

    variables = model.init(
        {"params": jax.random.PRNGKey(0), "sample": jax.random.PRNGKey(1)},
        x,
        t,
        known_token_mask,
        method=model.forward_sample_conditional,
    )
    zt = model.apply(
        variables,
        x,
        t,
        known_token_mask,
        method=model.forward_sample_conditional,
        rngs={"sample": jax.random.PRNGKey(2)},
    )

    np.testing.assert_array_equal(
        np.asarray(zt[known_token_mask]),
        np.asarray(x[known_token_mask]),
    )
    assert zt.shape == x.shape
    assert zt.dtype == jnp.int32


def test_selected_log_prob_sums_respects_suffix_loss_mask():
    log_probs = jnp.log(
        jnp.asarray(
            [[[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]],
            dtype=jnp.float32,
        )
    )
    targets = jnp.asarray([[0, 1, 1, 0]], dtype=jnp.int32)
    suffix_mask = jnp.asarray([[False, False, True, True]], dtype=jnp.bool_)

    masked_sum = _selected_log_prob_sums(log_probs, targets, mask=suffix_mask)
    expected = jnp.log(jnp.asarray(0.3, dtype=jnp.float32)) + jnp.log(
        jnp.asarray(0.4, dtype=jnp.float32)
    )

    assert jnp.allclose(masked_sum, expected)


def test_top_prob_margin_selects_highest_margin_positions():
    log_probs = jnp.log(
        jnp.asarray(
            [
                [
                    [0.90, 0.05, 0.05],
                    [0.60, 0.39, 0.01],
                    [0.80, 0.10, 0.10],
                    [0.51, 0.49, 0.00],
                ]
            ],
            dtype=jnp.float32,
        )
    )
    masked_unknown_mask = jnp.asarray([[True, True, True, False]], dtype=jnp.bool_)

    selected = select_top_prob_margin_positions(
        log_probs,
        masked_unknown_mask,
        reveal_prob=jnp.asarray(0.66, dtype=jnp.float32),
    )

    np.testing.assert_array_equal(
        np.asarray(selected),
        np.asarray([[True, False, True, False]], dtype=np.bool_),
    )


def test_top_probability_selects_highest_top1_positions():
    log_probs = jnp.log(
        jnp.asarray(
            [
                [
                    [0.55, 0.45, 0.00],
                    [0.90, 0.05, 0.05],
                    [0.70, 0.20, 0.10],
                    [0.99, 0.01, 0.00],
                ]
            ],
            dtype=jnp.float32,
        )
    )
    masked_unknown_mask = jnp.asarray([[True, True, True, False]], dtype=jnp.bool_)

    selected = select_top_probability_positions(
        log_probs,
        masked_unknown_mask,
        reveal_prob=jnp.asarray(0.66, dtype=jnp.float32),
    )

    np.testing.assert_array_equal(
        np.asarray(selected),
        np.asarray([[False, True, True, False]], dtype=np.bool_),
    )


def test_top_prob_margin_gumbel_noise_only_affects_masked_unknown_positions():
    log_probs = jnp.log(
        jnp.asarray(
            [
                [
                    [0.60, 0.20, 0.20],
                    [0.60, 0.20, 0.20],
                    [0.95, 0.03, 0.02],
                    [0.70, 0.30, 0.00],
                ]
            ],
            dtype=jnp.float32,
        )
    )
    masked_unknown_mask = jnp.asarray([[True, True, False, False]], dtype=jnp.bool_)
    selected = select_top_prob_margin_positions(
        log_probs,
        masked_unknown_mask,
        reveal_prob=jnp.asarray(0.5, dtype=jnp.float32),
        rng=jax.random.PRNGKey(7),
        oracle_noise_type="gumbel",
        oracle_noise_scale=0.5,
    )

    assert selected.shape == masked_unknown_mask.shape
    assert np.all(np.asarray(selected[~masked_unknown_mask]) == 0)
    assert int(np.asarray(selected).sum()) == 1


class _ToyConditionalModel:
    timesteps = 3
    vocab_size = 10
    mask_token_id = 10
    cache_predictions = False
    time_features = "none"
    sampler = "top_prob_margin"

    def prior_sample(self, batch_size: int):
        return jnp.full((batch_size, 6), self.mask_token_id, dtype=jnp.int32)

    def sample_step(
        self,
        rng,
        i,
        timesteps,
        state,
        *,
        conditioning=None,
        known_tokens=None,
        known_token_mask=None,
        return_info: bool = False,
    ):
        del rng, i, timesteps, conditioning
        filled = jnp.where(state == self.mask_token_id, 7, state).astype(jnp.int32)
        next_state = jnp.where(known_token_mask, known_tokens, filled).astype(jnp.int32)
        if not bool(return_info):
            return next_state
        masked_unknown = (state == self.mask_token_id) & (~known_token_mask)
        selected = masked_unknown
        token_pos = jnp.arange(state.shape[-1], dtype=jnp.int32)
        return next_state, {
            "masked_unknown_total": jnp.sum(masked_unknown.astype(jnp.float32)),
            "selected_count_total": jnp.sum(selected.astype(jnp.float32)),
            "selected_margin_sum_total": jnp.asarray(0.5, dtype=jnp.float32)
            * jnp.sum(selected.astype(jnp.float32)),
            "selected_margin_count_total": jnp.sum(selected.astype(jnp.float32)),
            "selected_row_total": jnp.sum(
                selected.astype(jnp.float32) * (jnp.mod(token_pos, 3) == 0).astype(jnp.float32)
            ),
            "selected_col_total": jnp.sum(
                selected.astype(jnp.float32) * (jnp.mod(token_pos, 3) == 1).astype(jnp.float32)
            ),
            "selected_value_total": jnp.sum(
                selected.astype(jnp.float32) * (jnp.mod(token_pos, 3) == 2).astype(jnp.float32)
            ),
        }

    def decode(self, state, *, conditioning=None):
        del conditioning
        return state

    def apply(self, variables, *args, method=None, **kwargs):
        del variables
        method_name = getattr(method, "__name__", "")
        if method_name == "prior_sample":
            return self.prior_sample(*args, **kwargs)
        if method_name == "sample_step":
            return self.sample_step(*args, **kwargs)
        if method_name == "decode":
            return self.decode(*args, **kwargs)
        raise AssertionError(f"Unexpected method dispatch: {method_name}")


def test_conditional_generate_preserves_known_tokens_and_reports_diagnostics():
    model = _ToyConditionalModel()
    known_tokens = jnp.asarray([[1, 2, 3, 0, 0, 0]], dtype=jnp.int32)
    known_token_mask = jnp.asarray([[True, True, True, False, False, False]], dtype=jnp.bool_)

    generated, diagnostics = conditional_generate(
        jax.random.PRNGKey(0),
        {"params": {}, "ema_params": None},
        model=model,
        known_tokens=known_tokens,
        known_token_mask=known_token_mask,
        timesteps=3,
        conditioning=None,
        use_ema=False,
        return_diagnostics=True,
    )

    np.testing.assert_array_equal(
        np.asarray(generated[known_token_mask]),
        np.asarray(known_tokens[known_token_mask]),
    )
    np.testing.assert_array_equal(
        np.asarray(generated[~known_token_mask]),
        np.asarray(jnp.full((3,), 7, dtype=jnp.int32)),
    )
    assert float(diagnostics["masked_unknown_total_across_steps"]) == 3.0
    assert float(diagnostics["selected_count_total_across_steps"]) == 3.0
    assert float(diagnostics["selected_margin_sum_total"]) == 1.5
    assert float(diagnostics["selected_row_total_across_steps"]) == 1.0
    assert float(diagnostics["selected_col_total_across_steps"]) == 1.0
    assert float(diagnostics["selected_value_total_across_steps"]) == 1.0
    assert float(diagnostics["example_step_count"]) == 3.0
    assert float(diagnostics["final_masked_unknown_total"]) == 0.0


class _IdentityNoiseSchedule:
    def alpha(self, t):
        return jnp.asarray(t, dtype=jnp.float32)


class _FakeRevealOrderModel:
    def __init__(self, *, log_probs: jnp.ndarray, revealed_token_sample_mode: str):
        self.log_probs = jnp.asarray(log_probs, dtype=jnp.float32)
        self.mask_token_id = 99
        self.categorical_sampling_policy = "exact"
        self.oracle_noise_type = "none"
        self.oracle_noise_scale = 0.0
        self.revealed_token_sample_mode = revealed_token_sample_mode
        self.noise_schedule = _IdentityNoiseSchedule()

    def get_sampling_grid(self, i: int, timesteps: int):
        del i, timesteps
        return jnp.asarray(0.75, dtype=jnp.float32), jnp.asarray(0.5, dtype=jnp.float32)

    def _sampling_log_probs(self, state, *, t, conditioning=None):
        del t, conditioning
        return jnp.asarray(state, dtype=jnp.int32), self.log_probs, False

    def _masked_unknown_positions(self, tokens, *, known_token_mask=None):
        known = (
            jnp.zeros_like(tokens, dtype=jnp.bool_)
            if known_token_mask is None
            else jnp.asarray(known_token_mask, dtype=jnp.bool_)
        )
        return (jnp.asarray(tokens, dtype=jnp.int32) == self.mask_token_id) & (~known)

    def _clamp_known_tokens(
        self,
        tokens,
        *,
        current_tokens,
        known_token_mask=None,
        known_tokens=None,
    ):
        if known_token_mask is None:
            return jnp.asarray(tokens, dtype=jnp.int32)
        source = (
            jnp.asarray(current_tokens, dtype=jnp.int32)
            if known_tokens is None
            else jnp.asarray(known_tokens, dtype=jnp.int32)
        )
        return jnp.where(known_token_mask, source, tokens).astype(jnp.int32)

    def _use_cache(self) -> bool:
        return False

    def _pack_sampling_state(self, tokens, *, log_probs, cache_valid, structured):
        del log_probs, cache_valid, structured
        return jnp.asarray(tokens, dtype=jnp.int32)

    def _sampling_step_info(self, *, masked_unknown, reveal_positions=None, margins=None):
        return MDLM._sampling_step_info(
            self,
            masked_unknown=masked_unknown,
            reveal_positions=reveal_positions,
            margins=margins,
        )


def test_reveal_order_sample_mode_preserves_default_sampling(monkeypatch):
    reveal_positions = jnp.asarray([[True, False, True, False]], dtype=jnp.bool_)
    log_probs = jnp.log(
        jnp.asarray(
            [[[0.1, 0.9, 0.0], [0.8, 0.2, 0.0], [0.05, 0.05, 0.9], [0.6, 0.4, 0.0]]],
            dtype=jnp.float32,
        )
    )
    sampled = jnp.asarray([[7, 7, 7, 7]], dtype=jnp.int32)

    monkeypatch.setattr(
        sudoku_sampling_mod,
        "select_top_prob_margin_positions",
        lambda *args, **kwargs: reveal_positions,
    )
    monkeypatch.setattr(
        sudoku_sampling_mod,
        "categorical_sample_from_logits",
        lambda *args, **kwargs: sampled,
    )

    model = _FakeRevealOrderModel(
        log_probs=log_probs,
        revealed_token_sample_mode="sample",
    )
    state = jnp.asarray([[99, 4, 99, 6]], dtype=jnp.int32)

    next_tokens = MDLM.reveal_order_sample_step(
        model,
        jax.random.PRNGKey(0),
        0,
        4,
        state,
        known_token_mask=jnp.zeros_like(state, dtype=jnp.bool_),
        known_tokens=jnp.zeros_like(state, dtype=jnp.int32),
        method="top_prob_margin",
    )

    np.testing.assert_array_equal(
        np.asarray(next_tokens),
        np.asarray([[7, 4, 7, 6]], dtype=np.int32),
    )


def test_reveal_order_sample_mode_argmax_only_decodes_new_reveals(monkeypatch):
    reveal_positions = jnp.asarray([[True, False, True, False]], dtype=jnp.bool_)
    log_probs = jnp.log(
        jnp.asarray(
            [[[0.1, 0.9, 0.0], [0.8, 0.2, 0.0], [0.05, 0.05, 0.9], [0.6, 0.4, 0.0]]],
            dtype=jnp.float32,
        )
    )

    monkeypatch.setattr(
        sudoku_sampling_mod,
        "select_top_prob_margin_positions",
        lambda *args, **kwargs: reveal_positions,
    )

    def _unexpected_sample(*args, **kwargs):
        raise AssertionError("categorical sampling should not run in argmax reveal mode")

    monkeypatch.setattr(
        sudoku_sampling_mod,
        "categorical_sample_from_logits",
        _unexpected_sample,
    )

    model = _FakeRevealOrderModel(
        log_probs=log_probs,
        revealed_token_sample_mode="argmax",
    )
    state = jnp.asarray([[99, 4, 99, 6]], dtype=jnp.int32)

    next_tokens = MDLM.reveal_order_sample_step(
        model,
        jax.random.PRNGKey(0),
        0,
        4,
        state,
        known_token_mask=jnp.zeros_like(state, dtype=jnp.bool_),
        known_tokens=jnp.zeros_like(state, dtype=jnp.int32),
        method="top_prob_margin",
    )

    np.testing.assert_array_equal(
        np.asarray(next_tokens),
        np.asarray([[1, 4, 2, 6]], dtype=np.int32),
    )


def test_reveal_order_sample_step_clamps_known_tokens_even_if_selector_targets_them(monkeypatch):
    log_probs = jnp.log(
        jnp.asarray(
            [[[0.1, 0.9, 0.0], [0.8, 0.2, 0.0], [0.05, 0.05, 0.9], [0.6, 0.4, 0.0]]],
            dtype=jnp.float32,
        )
    )
    sampled = jnp.asarray([[7, 7, 7, 7]], dtype=jnp.int32)

    monkeypatch.setattr(
        sudoku_sampling_mod,
        "select_top_prob_margin_positions",
        lambda *args, **kwargs: jnp.ones((1, 4), dtype=jnp.bool_),
    )
    monkeypatch.setattr(
        sudoku_sampling_mod,
        "categorical_sample_from_logits",
        lambda *args, **kwargs: sampled,
    )

    model = _FakeRevealOrderModel(
        log_probs=log_probs,
        revealed_token_sample_mode="sample",
    )
    state = jnp.asarray([[5, 99, 8, 99]], dtype=jnp.int32)
    known_token_mask = jnp.asarray([[True, False, True, False]], dtype=jnp.bool_)

    next_tokens = MDLM.reveal_order_sample_step(
        model,
        jax.random.PRNGKey(0),
        0,
        4,
        state,
        known_token_mask=known_token_mask,
        known_tokens=state,
        method="top_prob_margin",
    )

    np.testing.assert_array_equal(
        np.asarray(next_tokens),
        np.asarray([[5, 7, 8, 7]], dtype=np.int32),
    )


def test_tiny_sudoku_overfit_reaches_exact_completion():
    seq = _canonical_sudoku_sequence()
    batch_size = 512
    start_index = 80
    x = jnp.repeat(seq[None, :], batch_size, axis=0)
    start_index_arr = jnp.full((batch_size, 1), start_index, dtype=jnp.int32)
    token_pos = jnp.arange(seq.shape[0], dtype=jnp.int32)[None, :]
    known_token_mask = token_pos < (3 * start_index_arr)
    loss_mask = ~known_token_mask

    model = _make_sequence_model(
        seq_len=243,
        sampler="top_prob_margin",
        feature_dim=8,
        num_heads=2,
        n_layers=1,
        mlp_hidden_dim=64,
        timesteps=8,
    )

    variables = model.init(
        {"params": jax.random.PRNGKey(0), "sample": jax.random.PRNGKey(1)},
        x[:2],
        train=False,
        known_token_mask=known_token_mask[:2],
        loss_mask=loss_mask[:2],
    )
    params = variables["params"]
    tx = optax.adam(3.0e-3)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, rng):
        def loss_fn(p):
            stats = model.apply(
                {"params": p},
                x,
                train=True,
                known_token_mask=known_token_mask,
                loss_mask=loss_mask,
                rngs={"sample": rng},
            )
            return stats["loss"]

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    initial_loss = None
    loss = None
    for step in range(80):
        params, opt_state, loss = train_step(params, opt_state, jax.random.PRNGKey(step + 10))
        if step == 0:
            initial_loss = loss

    assert initial_loss is not None
    assert loss is not None
    assert float(loss) < float(initial_loss)

    known_tokens = jnp.where(known_token_mask, x, 0).astype(jnp.int32)
    generated = conditional_generate(
        jax.random.PRNGKey(999),
        {"params": params, "ema_params": None},
        model=model,
        known_tokens=known_tokens[:32],
        known_token_mask=known_token_mask[:32],
        timesteps=model.timesteps,
        conditioning=None,
        use_ema=False,
    )

    exact = jnp.all(generated == x[:32], axis=-1)
    assert float(jnp.mean(exact.astype(jnp.float32))) >= 0.7
