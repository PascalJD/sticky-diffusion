from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
np = pytest.importorskip("numpy")

from omegaconf import OmegaConf

from sticky.data.sudoku import pack_sudoku_seq2seq
import sticky.models.baselines.mdm.sudoku_sampling as mdm_sampling_mod
from sticky.models.baselines.mdm import conditional_generate
from sticky.models.baselines.mdm.mdm_model import MDM
from sticky.models.factory import build_model


def _make_model(*, seq_len: int = 245, vocab_size: int = 12, **overrides) -> MDM:
    return MDM(
        data_shape=(seq_len,),
        cont_time=False,
        timesteps=50,
        feature_dim=8,
        num_heads=2,
        antithetic_time_sampling=False,
        n_layers=1,
        n_dit_layers=0,
        dit_num_heads=2,
        dit_hidden_size=16,
        ch_mult=(1,),
        vocab_size=vocab_size,
        noise_schedule_type="loglinear",
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
        oracle_noise_type="gumbel",
        oracle_noise_scale=0.5,
        revealed_token_sample_mode="sample",
        cache_predictions=False,
        model_sharding=False,
        **overrides,
    )


class _ToyMDMConditionalModel:
    def __init__(
        self,
        *,
        aligned_probs,
        sampler: str = "top_prob_margin",
        decoding_style: str = "monotone_reveal",
        oracle_noise_type: str = "none",
        oracle_noise_scale: float = 0.5,
        timesteps: int = 3,
    ):
        probs = jnp.asarray(aligned_probs, dtype=jnp.float32)
        if probs.ndim != 3:
            raise ValueError(f"Expected aligned_probs with shape (B, L, V), got {probs.shape}.")
        self.aligned_logits = jnp.log(probs)
        self.vocab_size = int(probs.shape[-1])
        self.mask_token_id = self.vocab_size
        self.sampler = sampler
        self.decoding_style = decoding_style
        self.oracle_noise_type = oracle_noise_type
        self.oracle_noise_scale = oracle_noise_scale
        self.timesteps = timesteps

    def prior_sample(self, batch_size: int):
        seq_len = int(self.aligned_logits.shape[1])
        return jnp.full((batch_size, seq_len), self.mask_token_id, dtype=jnp.int32)

    def predict_logits(self, zt, t=None, *, cond=None, train: bool = False):
        del t, cond, train
        batch_size = int(jnp.asarray(zt).shape[0])
        return jnp.broadcast_to(
            self.aligned_logits,
            (batch_size, self.aligned_logits.shape[1], self.aligned_logits.shape[2]),
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
        return mdm_sampling_mod.reveal_order_sample_step(
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


def test_mdm_forward_returns_logits_for_packed_sudoku_sequences():
    model = _make_model()
    x = (
        jnp.reshape(jnp.arange(2 * 245, dtype=jnp.int32), (2, 245))
        % jnp.asarray(model.vocab_size, dtype=jnp.int32)
    )

    variables = model.init({"params": jax.random.PRNGKey(0)}, x, train=False)
    outputs = model.apply(variables, x, train=False)
    logits = outputs["logits"]

    assert logits.shape == (2, 245, 12)
    assert model.mask_token_id == 12
    assert model.sequence_backbone == "gpt2_like"
    assert model.sequence_causal is False
    assert model.time_features == "none"

    prior = model.apply(variables, 2, method=model.prior_sample)
    assert prior.shape == (2, 245)
    assert prior.dtype == jnp.int32
    assert bool(jnp.all(prior == model.mask_token_id))

    decoded = model.apply(variables, logits, method=model.decode)
    assert decoded.shape == (2, 245)
    assert decoded.dtype == jnp.int32


def test_build_model_resolves_mdm_name_and_uses_packed_vocab():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "mdm",
                "cont_time": False,
                "timesteps": 50,
                "feature_dim": 32,
                "num_heads": 12,
                "antithetic_time_sampling": False,
                "n_layers": 3,
                "n_dit_layers": 0,
                "dit_num_heads": 12,
                "dit_hidden_size": 384,
                "ch_mult": [1],
                "noise_schedule_type": "loglinear",
                "dropout_rate": 0.1,
                "use_attn_dropout": True,
                "mlp_type": "gelu",
                "depth_scaled_init": False,
                "cond_type": "adaln",
                "outside_embed": False,
                "sequence_backbone": "gpt2_like",
                "sequence_mlp_hidden_dim": 1536,
                "sequence_max_length": 245,
                "sequence_causal": False,
                "image_backbone": "auto",
                "adm_num_res_blocks": 2,
                "adm_attention_resolutions": [2, 4],
                "adm_num_heads": 4,
                "adm_num_head_channels": -1,
                "adm_num_heads_upsample": -1,
                "adm_conv_resample": True,
                "adm_use_scale_shift_norm": True,
                "adm_resblock_updown": False,
                "adm_use_conv_skip": False,
                "adm_use_new_attention_order": False,
                "time_features": "none",
                "classes": -1,
                "cache_predictions": False,
                "token_reweighting": True,
                "alpha": 0.25,
                "gamma": 1.0,
                "time_reweighting": "linear",
                "model_sharding": False,
            },
            "sampler": {
                "method": "top_prob_margin",
                "sampling_grid": "loglinear",
                "categorical_sampling_policy": "exact",
                "decoding_style": "topk_remask",
                "oracle_noise_type": "gumbel",
                "oracle_noise_scale": 0.5,
                "revealed_token_sample_mode": "sample",
                "cache_predictions": False,
            },
        }
    )

    model = build_model(cfg, data_shape=(245,), vocab_size=12)

    assert isinstance(model, MDM)
    assert model.vocab_size == 12
    assert model.mask_token_id == 12
    assert model.sequence_backbone == "gpt2_like"
    assert model.sequence_mlp_hidden_dim == 1536
    assert model.sequence_max_length == 245
    assert model.sequence_causal is False
    assert model.decoding_style == "topk_remask"
    assert model.token_reweighting is True
    assert model.alpha == 0.25
    assert model.gamma == 1.0
    assert model.time_reweighting == "linear"


def test_mdm_reweighting_helpers_match_expected_formulas():
    model = _make_model()
    token_loss = jnp.asarray([[0.5, 1.25]], dtype=jnp.float32)
    t = jnp.asarray([0, 3], dtype=jnp.int32)

    neutral = model.apply_token_reweighting(token_loss)
    neutral_time = model.time_weights(t)

    assert jnp.allclose(neutral, token_loss)
    assert jnp.allclose(neutral_time, jnp.ones_like(t, dtype=jnp.float32))

    weighted_model = _make_model(
        token_reweighting=True,
        alpha=0.25,
        gamma=1.0,
        time_reweighting="linear",
    )

    expected_token = 0.25 * (1.0 - jnp.exp(-token_loss)) * token_loss
    expected_time = jnp.asarray(
        [float(weighted_model.timesteps), float(weighted_model.timesteps - 3)],
        dtype=jnp.float32,
    )

    assert jnp.allclose(weighted_model.apply_token_reweighting(token_loss), expected_token)
    assert jnp.allclose(weighted_model.time_weights(t), expected_time)


def test_mdm_alignment_matches_official_right_shift_convention():
    model = _make_model(seq_len=5, vocab_size=3)
    raw_logits = jnp.asarray(
        [
            [
                [10.0, 10.0, 10.0],
                [20.0, 20.0, 20.0],
                [30.0, 30.0, 30.0],
                [40.0, 40.0, 40.0],
                [50.0, 50.0, 50.0],
            ]
        ],
        dtype=jnp.float32,
    )

    aligned = model.align_right_shifted_logits(raw_logits)

    assert aligned.shape == raw_logits.shape
    assert jnp.allclose(aligned[:, 0], raw_logits[:, 0])
    assert jnp.allclose(aligned[:, 1], raw_logits[:, 0])
    assert jnp.allclose(aligned[:, 2], raw_logits[:, 1])
    assert jnp.allclose(aligned[:, 4], raw_logits[:, 3])


def test_mdm_alignment_changes_loss_relative_to_same_position_logits():
    model = _make_model(seq_len=4, vocab_size=3)
    targets = jnp.asarray([[0, 1, 2, 0]], dtype=jnp.int32)
    raw_logits = -1.0e9 * jnp.ones((1, 4, 3), dtype=jnp.float32)

    # Raw slot i predicts the *next* token perfectly, so same-position loss
    # would be wrong while the official right-shifted loss becomes near-zero.
    raw_logits = raw_logits.at[0, 0, 1].set(0.0)
    raw_logits = raw_logits.at[0, 1, 2].set(0.0)
    raw_logits = raw_logits.at[0, 2, 0].set(0.0)
    raw_logits = raw_logits.at[0, 3, 2].set(0.0)

    mask = jnp.asarray([[False, True, True, True]], dtype=jnp.float32)
    same_position_ce = model.token_cross_entropy(raw_logits, targets)
    aligned_ce = model.token_cross_entropy(
        model.align_right_shifted_logits(raw_logits),
        targets,
    )

    same_position_loss = jnp.sum(same_position_ce * mask) / jnp.sum(mask)
    aligned_loss = jnp.sum(aligned_ce * mask) / jnp.sum(mask)

    assert float(aligned_loss) < 1e-6
    assert float(same_position_loss) > 1.0


def test_mdm_alignment_interacts_correctly_with_packed_prompt_response_masks():
    model = _make_model(seq_len=245, vocab_size=12)
    triplet_seq = (jnp.arange(243, dtype=jnp.int32) % 10)[None, :]
    packed = pack_sudoku_seq2seq(
        triplet_seq=triplet_seq,
        start_index=jnp.asarray([[2]], dtype=jnp.int32),
    )
    seq_len = int(packed["packed_seq"].shape[1])
    raw_logits = jnp.repeat(
        jnp.arange(seq_len, dtype=jnp.float32)[None, :, None],
        repeats=12,
        axis=-1,
    )
    aligned = model.align_right_shifted_logits(raw_logits)

    sep_index = int(packed["sep_index"][0, 0])
    response_start_index = int(packed["response_start_index"][0, 0])
    eos_index = int(packed["eos_index"][0, 0])

    assert bool(packed["prompt_mask"][0, sep_index])
    assert bool(packed["response_mask"][0, response_start_index])
    assert jnp.allclose(aligned[0, response_start_index], raw_logits[0, sep_index])
    assert jnp.allclose(
        aligned[0, response_start_index + 1],
        raw_logits[0, response_start_index],
    )
    assert jnp.allclose(aligned[0, eos_index], raw_logits[0, eos_index - 1])


def test_mdm_reveal_order_decoding_uses_argmax_values(monkeypatch):
    reveal_positions = jnp.asarray([[True, False, True, False]], dtype=jnp.bool_)
    monkeypatch.setattr(
        mdm_sampling_mod,
        "select_top_prob_margin_positions",
        lambda *args, **kwargs: reveal_positions,
    )

    model = _ToyMDMConditionalModel(
        aligned_probs=jnp.asarray(
            [
                [
                    [0.10, 0.90, 0.00],
                    [0.80, 0.20, 0.00],
                    [0.05, 0.05, 0.90],
                    [0.60, 0.40, 0.00],
                ]
            ],
            dtype=jnp.float32,
        ),
    )
    state = jnp.asarray([[model.mask_token_id, 1, model.mask_token_id, 2]], dtype=jnp.int32)

    next_tokens = mdm_sampling_mod.reveal_order_sample_step(
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
        np.asarray([[1, 1, 2, 2]], dtype=np.int32),
    )


def test_mdm_gumbel_noise_changes_positions_but_not_revealed_argmax_values():
    aligned_probs = jnp.asarray(
        [
            [
                [0.52, 0.48, 0.00],
                [0.47, 0.53, 0.00],
                [0.80, 0.20, 0.00],
                [0.90, 0.10, 0.00],
            ]
        ],
        dtype=jnp.float32,
    )
    noiseless_model = _ToyMDMConditionalModel(
        aligned_probs=aligned_probs,
        sampler="top_prob_margin",
        oracle_noise_type="none",
        oracle_noise_scale=0.5,
    )
    noisy_model = _ToyMDMConditionalModel(
        aligned_probs=aligned_probs,
        sampler="top_prob_margin",
        oracle_noise_type="gumbel",
        oracle_noise_scale=0.5,
    )
    state = jnp.asarray(
        [[noiseless_model.mask_token_id, noiseless_model.mask_token_id, 0, 0]],
        dtype=jnp.int32,
    )
    known_token_mask = jnp.zeros_like(state, dtype=jnp.bool_)
    argmax_tokens = jnp.argmax(noiseless_model.aligned_logits, axis=-1).astype(jnp.int32)

    noiseless = mdm_sampling_mod.reveal_order_sample_step(
        noiseless_model,
        jax.random.PRNGKey(0),
        0,
        2,
        state,
        known_token_mask=known_token_mask,
        known_tokens=jnp.zeros_like(state, dtype=jnp.int32),
        method="top_prob_margin",
    )
    noisy = mdm_sampling_mod.reveal_order_sample_step(
        noisy_model,
        jax.random.PRNGKey(0),
        0,
        2,
        state,
        known_token_mask=known_token_mask,
        known_tokens=jnp.zeros_like(state, dtype=jnp.int32),
        method="top_prob_margin",
    )

    noiseless_selected = (noiseless != state) & (state == noiseless_model.mask_token_id)
    noisy_selected = (noisy != state) & (state == noisy_model.mask_token_id)

    assert not jnp.array_equal(noiseless_selected, noisy_selected)
    np.testing.assert_array_equal(
        np.asarray(noiseless[noiseless_selected]),
        np.asarray(argmax_tokens[noiseless_selected]),
    )
    np.testing.assert_array_equal(
        np.asarray(noisy[noisy_selected]),
        np.asarray(argmax_tokens[noisy_selected]),
    )


def test_mdm_top_probability_and_margin_modes_differ_on_controlled_example():
    model = _ToyMDMConditionalModel(
        aligned_probs=jnp.asarray(
            [
                [
                    [0.60, 0.39, 0.01],
                    [0.20, 0.55, 0.25],
                    [0.90, 0.10, 0.00],
                    [0.80, 0.20, 0.00],
                ]
            ],
            dtype=jnp.float32,
        ),
        sampler="top_probability",
    )
    state = jnp.asarray([[model.mask_token_id, model.mask_token_id, 0, 0]], dtype=jnp.int32)
    known_token_mask = jnp.zeros_like(state, dtype=jnp.bool_)

    top_probability = mdm_sampling_mod.reveal_order_sample_step(
        model,
        jax.random.PRNGKey(0),
        0,
        2,
        state,
        known_token_mask=known_token_mask,
        known_tokens=jnp.zeros_like(state, dtype=jnp.int32),
        method="top_probability",
    )
    top_prob_margin = mdm_sampling_mod.reveal_order_sample_step(
        model,
        jax.random.PRNGKey(0),
        0,
        2,
        state,
        known_token_mask=known_token_mask,
        known_tokens=jnp.zeros_like(state, dtype=jnp.int32),
        method="top_prob_margin",
    )

    np.testing.assert_array_equal(
        np.asarray(top_probability),
        np.asarray([[0, model.mask_token_id, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(top_prob_margin),
        np.asarray([[model.mask_token_id, 1, 0, 0]], dtype=np.int32),
    )


def test_mdm_topk_remask_can_remask_already_revealed_response_tokens():
    model = _ToyMDMConditionalModel(
        aligned_probs=jnp.asarray(
            [
                [
                    [0.95, 0.05, 0.00],
                    [0.55, 0.45, 0.00],
                    [0.51, 0.49, 0.00],
                    [0.99, 0.01, 0.00],
                ]
            ],
            dtype=jnp.float32,
        ),
        sampler="top_probability",
        decoding_style="topk_remask",
        timesteps=4,
    )
    state = jnp.asarray([[1, 1, 0, 0]], dtype=jnp.int32)
    known_token_mask = jnp.asarray([[True, False, False, False]], dtype=jnp.bool_)

    next_tokens = mdm_sampling_mod.reveal_order_sample_step(
        model,
        jax.random.PRNGKey(0),
        0,
        4,
        state,
        known_token_mask=known_token_mask,
        known_tokens=jnp.asarray([[1, 0, 0, 0]], dtype=jnp.int32),
        method="top_probability",
    )

    assert int(next_tokens[0, 1]) == model.mask_token_id
    assert int(next_tokens[0, 2]) == model.mask_token_id
    assert int(next_tokens[0, 3]) == 0


def test_mdm_packed_selection_diagnostics_use_original_triplet_coordinates():
    packed = pack_sudoku_seq2seq(
        triplet_seq=(jnp.arange(243, dtype=jnp.int32) % 10)[None, :],
        start_index=jnp.asarray([[1]], dtype=jnp.int32),
    )
    selected = jnp.zeros_like(packed["packed_seq"], dtype=jnp.bool_)
    sep_index = int(packed["sep_index"][0, 0])
    response_start = int(packed["response_start_index"][0, 0])
    eos_index = int(packed["eos_index"][0, 0])

    selected = selected.at[0, sep_index].set(True)
    selected = selected.at[0, response_start].set(True)
    selected = selected.at[0, response_start + 1].set(True)
    selected = selected.at[0, response_start + 2].set(True)
    selected = selected.at[0, eos_index].set(True)

    totals = mdm_sampling_mod._packed_selection_component_totals(
        selected,
        known_token_mask=packed["prompt_mask"],
    )

    assert float(totals["selected_row_total"]) == 1.0
    assert float(totals["selected_col_total"]) == 1.0
    assert float(totals["selected_value_total"]) == 1.0
    assert float(totals["selected_eos_total"]) == 1.0


def test_mdm_conditional_generate_accepts_sampler_override(monkeypatch):
    calls: list[str] = []

    def _override_selector(log_probs, masked_unknown_mask, **kwargs):
        del log_probs, kwargs
        calls.append("top_prob_margin")
        return masked_unknown_mask

    monkeypatch.setattr(
        mdm_sampling_mod,
        "select_top_probability_positions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("sampler_override should bypass the model default sampler")
        ),
    )
    monkeypatch.setattr(
        mdm_sampling_mod,
        "select_top_prob_margin_positions",
        _override_selector,
    )

    model = _ToyMDMConditionalModel(
        aligned_probs=jnp.asarray(
            [[[0.80, 0.20, 0.00], [0.20, 0.80, 0.00], [0.90, 0.10, 0.00]]],
            dtype=jnp.float32,
        ),
        sampler="top_probability",
        timesteps=2,
    )
    known_tokens = jnp.asarray([[1, 0, 0]], dtype=jnp.int32)
    known_token_mask = jnp.asarray([[True, False, False]], dtype=jnp.bool_)

    generated = conditional_generate(
        jax.random.PRNGKey(0),
        {"params": {}, "ema_params": None},
        model=model,
        known_tokens=known_tokens,
        known_token_mask=known_token_mask,
        timesteps=2,
        conditioning=None,
        use_ema=False,
        sampler_override="top_prob_margin",
    )

    assert calls
    np.testing.assert_array_equal(
        np.asarray(generated),
        np.asarray([[1, 1, 0]], dtype=np.int32),
    )


def test_mdm_conditional_generate_terminates_without_masked_response_tokens():
    model = _ToyMDMConditionalModel(
        aligned_probs=jnp.asarray(
            [
                [
                    [0.10, 0.90, 0.00],
                    [0.20, 0.10, 0.70],
                    [0.75, 0.20, 0.05],
                    [0.05, 0.15, 0.80],
                ]
            ],
            dtype=jnp.float32,
        ),
        sampler="vanilla",
        timesteps=3,
    )
    known_tokens = jnp.asarray([[1, 0, 0, 0]], dtype=jnp.int32)
    known_token_mask = jnp.asarray([[True, False, False, False]], dtype=jnp.bool_)

    generated, diagnostics = conditional_generate(
        jax.random.PRNGKey(123),
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
        np.asarray(generated),
        np.asarray([[1, 2, 0, 2]], dtype=np.int32),
    )
    assert not bool(jnp.any(generated[~known_token_mask] == model.mask_token_id))
    assert float(diagnostics["final_masked_unknown_total"]) == 0.0
