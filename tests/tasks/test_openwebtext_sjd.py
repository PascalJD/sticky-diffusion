from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import sticky.tasks.openwebtext_sjd as task_mod
from sticky.tasks.openwebtext_sjd import OpenWebTextSJDTask


def _beta(t):
    t = jnp.asarray(t, dtype=jnp.float32)
    return jnp.ones_like(t)


class _DummySJDModel:
    def embed(self, token_ids):
        return jnp.asarray(token_ids, dtype=jnp.float32)[..., None]

    def anchor_table(self):
        return jnp.arange(32, dtype=jnp.float32).reshape(16, 2)

    def apply(self, variables, *args, method=None, **kwargs):
        del variables, kwargs
        if method is not None and getattr(method, "__name__", "") == "embed":
            return self.embed(args[0])
        if method is not None and getattr(method, "__name__", "") == "anchor_table":
            return self.anchor_table()
        xt = jnp.asarray(args[0], dtype=jnp.float32)
        return jnp.zeros(xt.shape[:-1] + (16,), dtype=jnp.float32), {}


def _write_tokens(tmp_path: Path, n: int, seq_len: int, vocab: int) -> Path:
    arr = (np.arange(n * seq_len, dtype=np.int64) % vocab).reshape(n, seq_len)
    p = tmp_path / "tokens.npy"
    np.save(p, arr)
    return p


def test_task_spec_is_text_with_owt_shape(tmp_path):
    path = _write_tokens(tmp_path, n=4, seq_len=128, vocab=50257)
    task = OpenWebTextSJDTask(
        task_name="openwebtext_sjd",
        train_tokens_path=str(path),
        eval_tokens_path=str(path),
        batch_size=2,
        eval_batch_size=2,
        seq_len=128,
        vocab_size=50257,
        beta=_beta,
        hazard=object(),
        jump=object(),
        T=1.0,
        log_state_dependency=False,
    )
    assert task.spec.task_type == "text"
    assert task.spec.data_shape == (128,)
    assert task.spec.vocab_size == 50257


def test_loss_fn_wires_token_ids_and_forwards_schedule(monkeypatch, tmp_path):
    captured = {}

    def _fake_ce(**kwargs):
        captured.update(kwargs)
        return (
            jnp.asarray(0.5, dtype=jnp.float32),
            {"CE/acc_top1_event": jnp.asarray(1.0, dtype=jnp.float32)},
        )

    monkeypatch.setattr(task_mod, "ce_allocation_loss", _fake_ce)

    path = _write_tokens(tmp_path, n=2, seq_len=8, vocab=16)
    task = OpenWebTextSJDTask(
        task_name="openwebtext_sjd",
        train_tokens_path=str(path),
        eval_tokens_path=str(path),
        batch_size=2,
        eval_batch_size=2,
        seq_len=8,
        vocab_size=16,
        beta=_beta,
        hazard=object(),
        jump=object(),
        T=1.0,
        log_state_dependency=False,
        time_sampling="antithetic",
        loss_weighting="alpha_deriv",
    )
    batch = {"image": np.arange(16, dtype=np.int64).reshape(2, 8)}
    loss, metrics = task.loss_fn(
        rng=jax.random.PRNGKey(0),
        model=_DummySJDModel(),
        params={},
        batch=batch,
        train=True,
    )
    np.testing.assert_array_equal(
        np.asarray(captured["x0_idx"]), np.asarray(batch["image"], dtype=np.int32)
    )
    assert captured["given_mask"] is None
    assert captured["time_sampling"] == "antithetic"
    assert captured["loss_weighting"] == "alpha_deriv"
    assert float(loss) == 0.5
    assert float(metrics["clean_index_min"]) == 0.0
    assert float(metrics["clean_index_max"]) == 15.0


def test_loss_fn_traces_under_jit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        task_mod,
        "ce_allocation_loss",
        lambda **kwargs: (
            jnp.asarray(0.25, dtype=jnp.float32),
            {"CE/acc_top1_event": jnp.asarray(1.0, dtype=jnp.float32)},
        ),
    )
    path = _write_tokens(tmp_path, n=2, seq_len=8, vocab=16)
    task = OpenWebTextSJDTask(
        task_name="openwebtext_sjd",
        train_tokens_path=str(path),
        eval_tokens_path=str(path),
        batch_size=2,
        eval_batch_size=2,
        seq_len=8,
        vocab_size=16,
        beta=_beta,
        hazard=object(),
        jump=object(),
        T=1.0,
        log_state_dependency=False,
    )

    @jax.jit
    def run(rng, tokens):
        loss, metrics = task.loss_fn(
            rng=rng,
            model=_DummySJDModel(),
            params={},
            batch={"image": tokens},
            train=True,
        )
        return loss, metrics["clean_index_min"], metrics["clean_index_max"]

    tokens = jnp.arange(16, dtype=jnp.int32).reshape(2, 8)
    loss, cmin, cmax = run(jax.random.PRNGKey(1), tokens)
    assert float(loss) == 0.25
    assert float(cmin) == 0.0
    assert float(cmax) == 15.0
