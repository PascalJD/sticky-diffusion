"""End-to-end smoke test for per-cell anchor SJD on Sudoku-shaped data.

Builds a tiny SJD model with an 81-cell, 9-vocab, 4-dim per-position anchor
table, runs init + a forward pass + a 2-step reverse sample, and confirms:
  * the anchor parameter has shape (81, 9, 4),
  * embed() returns a position-aware gather,
  * gradient w.r.t. the anchor table flows from a synthetic loss,
  * the sampler's commit step produces (B, 81, 4) finite outputs.

Also smoke-tests the diagnose_per_cell_anchors CLI script.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

from sticky.models.sjd.anchors import AnchorTable, AnchorTableConfig
from sticky.models.sjd.corruption import classifier_induced_score
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.plugin_intensity import dhm_log_ratio
from sticky.models.sjd.sampler import (
    SamplerConfig,
    reverse_sample,
)
from sticky.models.sjd.sdes import make_beta
from sticky.models.sjd.sjd_model import SJD
from sticky.rng import make_rng


def _build_model(*, V=9, d=4, P=81, feature_dim=8):
    config = AnchorTableConfig(
        family="normal",
        vocab_size=V,
        anchor_dim=d,
        seed=0,
        normalize_at_use=True,
        n_positions=P,
    )
    model = SJD(
        anchor_config=config,
        learnable_anchors=True,
        vocab_size=V,
        feature_dim=feature_dim,
        num_heads=1,
        n_layers=1,
        ch_mult=(1,),
        sequence_backbone="auto",
        sequence_max_length=P,
        image_backbone="auto",
    )
    return model


def test_sudoku_per_cell_init_anchor_table_shape():
    V, d, P = 9, 4, 81
    model = _build_model(V=V, d=d, P=P)
    y_t = jnp.zeros((1, P, d), dtype=jnp.float32)
    t = jnp.zeros((1,), dtype=jnp.float32)
    token_ids = jnp.zeros((1, P), dtype=jnp.int32)
    variables = model.init(
        {"params": make_rng(0)},
        y_t,
        t,
        anchor_token_ids=token_ids,
        train=False,
    )
    assert variables["params"]["anchors"]["table"].shape == (P, V, d)


def test_sudoku_per_cell_embed_is_position_aware():
    V, d, P = 9, 4, 81
    model = _build_model(V=V, d=d, P=P)
    y_t = jnp.zeros((1, P, d), dtype=jnp.float32)
    t = jnp.zeros((1,), dtype=jnp.float32)
    token_ids = jnp.zeros((1, P), dtype=jnp.int32)
    variables = model.init(
        {"params": make_rng(0)},
        y_t,
        t,
        anchor_token_ids=token_ids,
        train=False,
    )
    # Embed all-zeros (every cell picks digit 0). Each cell uses its own
    # anchor block, so the 81 returned vectors should not all be equal —
    # this is the per-position behaviour the global-anchor design cannot
    # produce.
    embedded = model.apply(
        {"params": variables["params"]}, token_ids, method=model.embed
    )
    assert embedded.shape == (1, P, d)
    arr = np.asarray(embedded[0])
    assert not np.allclose(arr, arr[0:1].repeat(P, axis=0))


def test_sudoku_per_cell_dhm_and_score_consume_real_table():
    """Pulled-from-model anchor table should drive both DHM and the score."""
    V, d, P = 9, 4, 8
    model = _build_model(V=V, d=d, P=P)
    y_t = jnp.zeros((1, P, d), dtype=jnp.float32)
    t = jnp.zeros((1,), dtype=jnp.float32)
    token_ids = jnp.zeros((1, P), dtype=jnp.int32)
    variables = model.init(
        {"params": make_rng(0)},
        y_t,
        t,
        anchor_token_ids=token_ids,
        train=False,
    )
    table = model.apply({"params": variables["params"]}, method=model.anchor_table)
    assert tuple(table.shape) == (P, V, d)

    beta = make_beta(beta_min=0.1, beta_max=0.3, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.97, std_floor=1e-3)
    anchors = AnchorTable(table_float=table)

    rng = np.random.default_rng(42)
    y = jnp.asarray(rng.normal(size=(2, P, d)).astype(np.float32))
    t_img = jnp.asarray([0.3, 0.7], dtype=jnp.float32)
    logits = jnp.asarray(rng.normal(size=(2, P, V)).astype(np.float32))

    log_ratio = dhm_log_ratio(
        y=y, t_img=t_img, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump,
    )
    assert log_ratio.shape == (2, P, V)
    assert np.all(np.isfinite(np.asarray(log_ratio)))

    score = classifier_induced_score(
        y, t_img,
        anchor_logits=logits, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump,
    )
    assert score.shape == y.shape
    assert np.all(np.isfinite(np.asarray(score)))


def test_sudoku_per_cell_loss_grad_flows_to_anchors():
    """A synthetic embed-distance loss should produce nonzero gradients on
    the per-cell anchor parameter."""
    V, d, P = 9, 4, 12
    model = _build_model(V=V, d=d, P=P, feature_dim=8)
    y_t = jnp.zeros((1, P, d), dtype=jnp.float32)
    t = jnp.zeros((1,), dtype=jnp.float32)
    token_ids = jnp.asarray(
        np.random.default_rng(0).integers(0, V, size=(2, P)),
        dtype=jnp.int32,
    )
    variables = model.init(
        {"params": make_rng(0)},
        y_t,
        t,
        anchor_token_ids=token_ids[:1],
        train=False,
    )

    def loss_fn(params):
        embedded = model.apply({"params": params}, token_ids, method=model.embed)
        return jnp.sum(embedded * embedded)

    grads = jax.grad(loss_fn)(variables["params"])
    anchor_grad = np.asarray(grads["anchors"]["table"])
    assert anchor_grad.shape == (P, V, d)
    assert np.any(np.abs(anchor_grad) > 0.0)


def test_sudoku_per_cell_reverse_sample_runs_end_to_end():
    """Two-step reverse sample with per-cell anchors produces valid outputs."""
    V, d, P = 9, 4, 9   # 9 cells just for speed
    model = _build_model(V=V, d=d, P=P, feature_dim=8)
    y_t = jnp.zeros((1, P, d), dtype=jnp.float32)
    t = jnp.zeros((1,), dtype=jnp.float32)
    token_ids = jnp.zeros((1, P), dtype=jnp.int32)
    variables = model.init(
        {"params": make_rng(0)},
        y_t,
        t,
        anchor_token_ids=token_ids,
        train=False,
    )
    params = variables["params"]
    table = model.apply({"params": params}, method=model.anchor_table)
    anchors = AnchorTable(table_float=table)

    beta = make_beta(beta_min=0.1, beta_max=0.3, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.97, std_floor=1e-3)

    def apply_model(params, y, t):
        logits, aux = model.apply({"params": params}, y, t, train=False)
        return logits, aux

    cfg = SamplerConfig(
        T=1.0,
        n_steps=2,
        alloc_mode="argmax",
        force_classify_at_end=True,
        metrics_count_nfe=False,
    )
    out = reverse_sample(
        make_rng(7),
        params=params,
        apply_model=apply_model,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        shape=(P,),
        batch_size=1,
        cfg=cfg,
    )
    # k_filled gives the final committed digit per site; values must be in [0, V).
    k = np.asarray(out.k_filled)
    assert k.shape == (1, P)
    assert np.all((k >= 0) & (k < V))


def test_diagnose_per_cell_anchors_script_runs(tmp_path: Path) -> None:
    """The diagnose script should run end-to-end on a freshly-init'd model
    using the per-cell experiment override, and write summary.json + heatmaps.
    """
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "tools" / "diagnose_per_cell_anchors.py"
    out_dir = tmp_path / "diag_out"
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = (
        src_path + os.pathsep + env["PYTHONPATH"]
        if "PYTHONPATH" in env
        else src_path
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--from-init",
            "experiment=sudoku/sjd_sudoku_per_cell",
            "--out",
            str(out_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"diagnose script failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    summary_path = out_dir / "summary.json"
    assert summary_path.exists(), "summary.json was not written"
    import json
    summary = json.loads(summary_path.read_text())
    assert summary["shape"] == [81, 9, 64]
    # Heatmaps for all 9 digits should exist (PNG when matplotlib is
    # available, else .npy fallback).
    for v in range(9):
        png = out_dir / f"cell_cosine_digit_{v}.png"
        npy = out_dir / f"cell_cosine_digit_{v}.npy"
        assert png.exists() or npy.exists(), (
            f"Missing cosine heatmap for digit {v} (looked for both .png and .npy)"
        )
