from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")

from sticky.models.baselines.mdm import board_sampling
from sticky.models.baselines.mdm.mdm_inpaint_model import MDMInpaint


def _solution_board() -> np.ndarray:
    return np.asarray(
        [[((row * 3 + row // 3 + col) % 9) + 1 for row in range(9) for col in range(9)]],
        dtype=np.int32,
    )


def _known_inputs() -> tuple[np.ndarray, np.ndarray]:
    solution = _solution_board()
    clue_mask = np.zeros((1, 81), dtype=np.bool_)
    clue_mask[0, :70] = True
    clue_board = np.where(clue_mask, solution, 0).astype(np.int32)
    return clue_board, clue_mask


def _aligned_logits(solution_board: np.ndarray) -> jnp.ndarray:
    logits = np.full((1, 81, 10), -6.0, dtype=np.float32)
    for pos in range(81):
        correct = int(solution_board[0, pos])
        logits[0, pos, 0] = 7.0
        logits[0, pos, correct] = 6.0 - 0.05 * pos
        logits[0, pos, (correct % 9) + 1] = 1.5 - 0.01 * pos
    return jnp.asarray(logits, dtype=jnp.float32)


class _ToyBoardModel:
    def __init__(self, *, logits, sampler: str = "top_prob_margin", timesteps: int = 4):
        self.aligned_logits = jnp.asarray(logits, dtype=jnp.float32)
        self.vocab_size = int(self.aligned_logits.shape[-1])
        self.mask_token_id = 0
        self.sampler = sampler
        self.timesteps = int(timesteps)

    def prior_sample(self, batch_size: int):
        seq_len = int(self.aligned_logits.shape[1])
        return jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    def predict_logits(self, zt, t=None, *, cond=None, train: bool = False):
        del zt, t, cond, train
        return jnp.broadcast_to(
            self.aligned_logits,
            (1, self.aligned_logits.shape[1], self.aligned_logits.shape[2]),
        )

    def sample_step(
        self,
        rng,
        i,
        timesteps,
        state,
        *,
        conditioning=None,
        known_token_mask=None,
        known_tokens=None,
        sampler_override: str | None = None,
        return_info: bool = False,
    ):
        method = self.sampler if sampler_override is None else sampler_override
        return board_sampling.reveal_order_sample_step(
            self,
            rng,
            i,
            timesteps,
            state,
            conditioning=conditioning,
            known_token_mask=known_token_mask,
            known_tokens=known_tokens,
            method=method,
            return_info=return_info,
        )

    def decode(self, state, *, conditioning=None):
        del conditioning
        return jnp.asarray(state, dtype=jnp.int32)

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


def test_mdm_inpaint_forward_returns_board_logits():
    model = MDMInpaint(
        data_shape=(81,),
        cont_time=False,
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
        noise_schedule_type="loglinear",
        dropout_rate=0.0,
        use_attn_dropout=False,
        mlp_type="gelu",
        depth_scaled_init=False,
        cond_type="adaln",
        outside_embed=False,
        sequence_backbone="gpt2_like",
        sequence_mlp_hidden_dim=32,
        sequence_max_length=81,
        sequence_causal=False,
        image_backbone="auto",
        adm_num_res_blocks=1,
        adm_attention_resolutions=(2, 4),
        adm_num_heads=1,
        adm_num_head_channels=-1,
        adm_num_heads_upsample=-1,
        adm_conv_resample=True,
        adm_use_scale_shift_norm=True,
        adm_resblock_updown=False,
        adm_use_conv_skip=False,
        adm_use_new_attention_order=False,
        time_features="none",
        classes=-1,
        sampler="top_prob_margin",
        sampling_grid="loglinear",
        categorical_sampling_policy="exact",
        model_sharding=False,
    )

    x = jnp.zeros((2, 81), dtype=jnp.int32)
    t = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    variables = model.init({"params": jax.random.PRNGKey(0)}, x, t, train=False)
    outputs = model.apply(variables, x, t, train=False)

    assert outputs["logits"].shape == (2, 81, 10)
    assert model.mask_token_id == 0


def test_clue_positions_remain_fixed_at_every_reverse_step():
    clue_board, clue_mask = _known_inputs()
    model = _ToyBoardModel(logits=_aligned_logits(_solution_board()))
    state = jnp.asarray(clue_board, dtype=jnp.int32)

    for i in range(model.timesteps):
        state = board_sampling.reveal_order_sample_step(
            model,
            jax.random.PRNGKey(10 + i),
            i,
            model.timesteps,
            state,
            known_token_mask=jnp.asarray(clue_mask),
            known_tokens=jnp.asarray(clue_board),
            method="top_prob_margin",
        )
        assert bool(jnp.array_equal(state[jnp.asarray(clue_mask)], jnp.asarray(clue_board)[jnp.asarray(clue_mask)]))


def test_monotone_reveal_never_remasks_a_cell():
    clue_board, clue_mask = _known_inputs()
    model = _ToyBoardModel(logits=_aligned_logits(_solution_board()))
    state = jnp.asarray(clue_board, dtype=jnp.int32)

    for i in range(model.timesteps):
        next_state = board_sampling.reveal_order_sample_step(
            model,
            jax.random.PRNGKey(20 + i),
            i,
            model.timesteps,
            state,
            known_token_mask=jnp.asarray(clue_mask),
            known_tokens=jnp.asarray(clue_board),
            method="top_probability",
        )
        assert int(jnp.sum(next_state == 0)) <= int(jnp.sum(state == 0))
        revealed = (state != 0) & (~jnp.asarray(clue_mask))
        assert bool(jnp.all(next_state[revealed] != 0))
        state = next_state


def test_zero_is_never_committed_as_revealed_value():
    clue_board, clue_mask = _known_inputs()
    solution = _solution_board()
    model = _ToyBoardModel(logits=_aligned_logits(solution))

    output, diag = board_sampling.conditional_generate(
        jax.random.PRNGKey(0),
        {"params": {}, "ema_params": None},
        model=model,
        known_tokens=jnp.asarray(clue_board),
        known_token_mask=jnp.asarray(clue_mask),
        timesteps=model.timesteps,
        use_ema=False,
        return_diagnostics=True,
        sampler_override="top_prob_margin",
    )

    unknown_mask = ~clue_mask
    assert bool(np.all(np.asarray(output)[unknown_mask] >= 1))
    assert float(diag["final_masked_unknown_total"]) == 0.0


def test_all_board_samplers_run_end_to_end():
    clue_board, clue_mask = _known_inputs()
    solution = _solution_board()

    for method in ("vanilla", "top_probability", "top_prob_margin"):
        model = _ToyBoardModel(logits=_aligned_logits(solution), sampler=method)
        output = board_sampling.conditional_generate(
            jax.random.PRNGKey(100),
            {"params": {}, "ema_params": None},
            model=model,
            known_tokens=jnp.asarray(clue_board),
            known_token_mask=jnp.asarray(clue_mask),
            timesteps=model.timesteps,
            use_ema=False,
            return_diagnostics=False,
            sampler_override=method,
        )
        output_np = np.asarray(output)
        np.testing.assert_array_equal(output_np[clue_mask], clue_board[clue_mask])
        assert bool(np.all(output_np[~clue_mask] >= 1))
