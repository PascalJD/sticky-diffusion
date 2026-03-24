from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sticky.models.mdlm.mdlm_model import MDLM, _selected_log_prob_sums
from sticky.models.mdlm.sudoku_sampling import (
    conditional_generate,
    select_top_prob_margin_positions,
)


def _make_sequence_model(*, seq_len: int = 12, sampler: str = "uniform") -> MDLM:
    return MDLM(
        data_shape=(seq_len,),
        cont_time=True,
        timesteps=4,
        feature_dim=8,
        num_heads=2,
        antithetic_time_sampling=False,
        n_layers=1,
        n_dit_layers=0,
        dit_num_heads=2,
        dit_hidden_size=16,
        ch_mult=(1,),
        vocab_size=10,
        noise_schedule_type="linear",
        dropout_rate=0.0,
        use_attn_dropout=False,
        mlp_type="gelu",
        depth_scaled_init=False,
        cond_type="adaln",
        outside_embed=False,
        sequence_backbone="gpt2_like",
        sequence_mlp_hidden_dim=32,
        sequence_max_length=seq_len,
        sequence_causal=False,
        image_backbone="auto",
        time_features="none",
        classes=-1,
        sampler=sampler,
        sampling_grid="cosine",
        categorical_sampling_policy="exact",
        cache_predictions=False,
        model_sharding=False,
    )


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


class _ToyConditionalModel:
    timesteps = 3
    vocab_size = 10
    mask_token_id = 10
    cache_predictions = False
    time_features = "none"

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
    ):
        del rng, i, timesteps, conditioning
        filled = jnp.where(state == self.mask_token_id, 7, state).astype(jnp.int32)
        return jnp.where(known_token_mask, known_tokens, filled).astype(jnp.int32)

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


def test_conditional_generate_preserves_known_tokens_exactly():
    model = _ToyConditionalModel()
    known_tokens = jnp.asarray([[1, 2, 3, 0, 0, 0]], dtype=jnp.int32)
    known_token_mask = jnp.asarray([[True, True, True, False, False, False]], dtype=jnp.bool_)

    generated = conditional_generate(
        jax.random.PRNGKey(0),
        {"params": {}, "ema_params": None},
        model=model,
        known_tokens=known_tokens,
        known_token_mask=known_token_mask,
        timesteps=3,
        conditioning=None,
        use_ema=False,
    )

    np.testing.assert_array_equal(
        np.asarray(generated[known_token_mask]),
        np.asarray(known_tokens[known_token_mask]),
    )
    np.testing.assert_array_equal(
        np.asarray(generated[~known_token_mask]),
        np.asarray(jnp.full((3,), 7, dtype=jnp.int32)),
    )
