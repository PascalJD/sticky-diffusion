from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from sticky.models.sjd.anchors import AnchorTable
from sticky.models.sjd.corrector import pc_gate_from_probs
from sticky.models.sjd.hazard import make_hazard_poly_alpha
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sampler import SamplerConfig, reverse_sample
from sticky.models.sjd.sampling import conditional_generate_board
from sticky.models.sjd.sdes import make_beta


def _anchor_table() -> AnchorTable:
    return AnchorTable(table_float=jnp.eye(4, dtype=jnp.float32))


class _RecordingApply:
    def __init__(self, anchor_table: AnchorTable):
        self.anchor_table = jnp.asarray(anchor_table.table_float, dtype=jnp.float32)
        self.calls: list[np.ndarray] = []

    def apply(self, variables, y, *, t, train=False):
        del variables, t, train
        y = jnp.asarray(y, dtype=jnp.float32)
        jax.debug.callback(lambda arr: self.calls.append(np.asarray(arr)), y)
        dists = jnp.sum(
            (y[..., None, :] - self.anchor_table[None, None, :, :]) ** 2,
            axis=-1,
        )
        return -dists, {}


def _run_sampler(cfg: SamplerConfig, *, key_seed: int = 0):
    anchors = _anchor_table()
    model = _RecordingApply(anchors)
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_poly_alpha(beta, p=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.8)
    known_idx = jnp.asarray([[0, 1, 0, 0, 2, 0]], dtype=jnp.int32)
    known_mask = jnp.asarray([[True, True, False, False, True, False]], dtype=jnp.bool_)
    out = reverse_sample(
        jax.random.PRNGKey(key_seed),
        params={},
        apply_model=lambda params, y, t_img: model.apply(
            {"params": params},
            y,
            t=t_img,
            train=False,
        ),
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        shape=(6,),
        batch_size=1,
        cfg=cfg,
        known_idx=known_idx,
        known_mask=known_mask,
    )
    return out, model, known_idx, known_mask, anchors


def test_pc_disabled_matches_predictor_only_exactly():
    base = SamplerConfig(
        n_steps=4,
        alloc_mode="argmax",
        categorical_sampling_policy="exact",
        pc_enabled=False,
    )
    disabled = SamplerConfig(
        n_steps=4,
        alloc_mode="argmax",
        categorical_sampling_policy="exact",
        pc_enabled=True,
        corrector_substeps=0,
        corrector_step_scale=0.1,
        pc_gate="constant_one",
    )
    out_base, _, _, _, _ = _run_sampler(base)
    out_disabled, _, _, _, _ = _run_sampler(disabled)
    np.testing.assert_array_equal(np.asarray(out_base.k), np.asarray(out_disabled.k))
    np.testing.assert_array_equal(np.asarray(out_base.k_filled), np.asarray(out_disabled.k_filled))
    np.testing.assert_array_equal(np.asarray(out_base.committed), np.asarray(out_disabled.committed))


def test_zero_strength_corrector_matches_predictor_only_exactly():
    base = SamplerConfig(
        n_steps=4,
        alloc_mode="argmax",
        categorical_sampling_policy="exact",
        pc_enabled=False,
    )
    zero = SamplerConfig(
        n_steps=4,
        alloc_mode="argmax",
        categorical_sampling_policy="exact",
        pc_enabled=True,
        corrector_substeps=2,
        corrector_step_scale=0.0,
        pc_gate="margin",
    )
    out_base, _, _, _, _ = _run_sampler(base, key_seed=1)
    out_zero, _, _, _, _ = _run_sampler(zero, key_seed=1)
    np.testing.assert_array_equal(np.asarray(out_base.k_filled), np.asarray(out_zero.k_filled))
    np.testing.assert_array_equal(np.asarray(out_base.committed), np.asarray(out_zero.committed))


def test_known_sites_remain_fixed_during_predictor_and_corrector_calls():
    cfg = SamplerConfig(
        n_steps=3,
        alloc_mode="argmax",
        categorical_sampling_policy="exact",
        pc_enabled=True,
        corrector_substeps=2,
        corrector_step_scale=0.1,
        pc_gate="margin",
        pc_clamp_known=True,
    )
    out, model, known_idx, known_mask, anchors = _run_sampler(cfg, key_seed=2)
    known_rows = np.asarray(anchors.table_float)[np.asarray(known_idx)[known_mask]]
    for seen in model.calls:
        seen_known = seen[0, np.asarray(known_mask)[0], :]
        np.testing.assert_allclose(seen_known, known_rows, atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(out.k_filled)[0, np.asarray(known_mask)[0]],
        np.asarray(known_idx)[0, np.asarray(known_mask)[0]],
    )


def test_pc_allow_unstick_unknown_only_changes_unstick_behavior_but_not_clues():
    enabled = SamplerConfig(
        n_steps=5,
        alloc_mode="argmax",
        categorical_sampling_policy="exact",
        pc_enabled=True,
        corrector_substeps=2,
        corrector_step_scale=0.1,
        pc_gate="constant_one",
        pc_allow_unstick_unknown_only=True,
    )
    disabled = SamplerConfig(
        n_steps=5,
        alloc_mode="argmax",
        categorical_sampling_policy="exact",
        pc_enabled=True,
        corrector_substeps=2,
        corrector_step_scale=0.1,
        pc_gate="constant_one",
        pc_allow_unstick_unknown_only=False,
    )

    out_enabled, _, known_idx, known_mask, _ = _run_sampler(enabled, key_seed=7)
    out_disabled, _, _, _, _ = _run_sampler(disabled, key_seed=7)

    assert float(out_enabled.metrics["sampling/anchor_to_continuous_unstick_attempts_total"]) > 0.0
    assert float(out_disabled.metrics["sampling/anchor_to_continuous_unstick_attempts_total"]) == 0.0
    np.testing.assert_array_equal(
        np.asarray(out_enabled.k_filled)[0, np.asarray(known_mask)[0]],
        np.asarray(known_idx)[0, np.asarray(known_mask)[0]],
    )
    np.testing.assert_array_equal(
        np.asarray(out_disabled.k_filled)[0, np.asarray(known_mask)[0]],
        np.asarray(known_idx)[0, np.asarray(known_mask)[0]],
    )


def test_pc_sampler_has_finite_metrics_and_sane_counters():
    cfg = SamplerConfig(
        n_steps=4,
        alloc_mode="argmax",
        categorical_sampling_policy="exact",
        pc_enabled=True,
        corrector_substeps=2,
        corrector_step_scale=0.1,
        pc_gate="entropy",
    )
    out, _, _, _, _ = _run_sampler(cfg, key_seed=3)
    metrics = jax.device_get(out.metrics)
    for value in metrics.values():
        assert np.isfinite(np.asarray(value)).all()
    assert float(metrics["sampling/anchor_to_continuous_unstick_accepts_total"]) <= float(
        metrics["sampling/anchor_to_continuous_unstick_attempts_total"]
    )
    assert float(metrics["sampling/continuous_to_anchor_commits_total"]) >= 0.0
    assert float(metrics["sampling/langevin_updates_total"]) >= 0.0


def test_pc_nfe_accounting_matches_expected_paths():
    predictor = SamplerConfig(n_steps=4, alloc_mode="argmax", pc_enabled=False)
    pc_constant = SamplerConfig(
        n_steps=4,
        alloc_mode="argmax",
        pc_enabled=True,
        corrector_substeps=2,
        corrector_step_scale=0.1,
        pc_gate="constant_one",
    )
    pc_margin = SamplerConfig(
        n_steps=4,
        alloc_mode="argmax",
        pc_enabled=True,
        corrector_substeps=2,
        corrector_step_scale=0.1,
        pc_gate="margin",
    )

    out_predictor, _, _, _, _ = _run_sampler(predictor, key_seed=4)
    out_constant, _, _, _, _ = _run_sampler(pc_constant, key_seed=4)
    out_margin, _, _, _, _ = _run_sampler(pc_margin, key_seed=4)

    assert float(out_predictor.metrics["sampling/nfe_total"]) == 4.0
    assert float(out_constant.metrics["sampling/nfe_total"]) == 12.0
    assert float(out_margin.metrics["sampling/nfe_total"]) == 20.0


def test_conditional_generate_board_clamps_clues_and_emits_digits():
    anchors = _anchor_table()
    model = _RecordingApply(anchors)
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_poly_alpha(beta, p=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.8)
    known_tokens = jnp.asarray([[1, 2, 0, 0, 3, 0]], dtype=jnp.int32)
    known_mask = known_tokens != 0
    for cfg in (
        SamplerConfig(
            n_steps=3,
            alloc_mode="argmax",
            pc_enabled=False,
        ),
        SamplerConfig(
            n_steps=3,
            alloc_mode="argmax",
            pc_enabled=True,
            corrector_substeps=1,
            corrector_step_scale=0.1,
            pc_gate="constant_one",
        ),
    ):
        board, metrics = conditional_generate_board(
            rng=jax.random.PRNGKey(5),
            params={},
            model=model,
            anchors=anchors,
            beta=beta,
            hazard=hazard,
            jump=jump,
            known_tokens=known_tokens,
            known_token_mask=known_mask,
            cfg=cfg,
        )
        board_np = np.asarray(board)
        assert set(np.unique(board_np)).issubset({1, 2, 3, 4})
        np.testing.assert_array_equal(board_np[known_mask], np.asarray(known_tokens)[known_mask])
        assert float(metrics["sampling/nfe_total"]) >= 3.0


def test_margin_pc_gate_matches_top2_margin_without_full_sort_assumptions():
    probs = jnp.asarray(
        [
            [0.05, 0.10, 0.70, 0.15],
            [0.24, 0.26, 0.25, 0.25],
        ],
        dtype=jnp.float32,
    )
    gate = np.asarray(pc_gate_from_probs(probs, gate="margin"))
    expected = np.asarray(
        [
            1.0 - (0.70 - 0.15),
            1.0 - (0.26 - 0.25),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(gate, expected, atol=1e-6)
