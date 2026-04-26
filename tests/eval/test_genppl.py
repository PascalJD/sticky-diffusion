from __future__ import annotations

import numpy as np
import pytest

import sticky.eval.genppl as genppl_mod
from sticky.eval.genppl import sample_entropy, score_generative_ppl


def test_sample_entropy_uniform_over_vocab_is_high():
    rng = np.random.default_rng(0)
    samples = rng.integers(0, 50257, size=(256, 1024), dtype=np.int64)
    ent = sample_entropy(samples, vocab_size=50257)
    # Uniform draws over 50257 tokens approach log(50257) ~ 10.83 nats; with
    # 256*1024 samples (partial coverage) we still expect a large entropy.
    assert 9.5 < ent < 11.0


def test_sample_entropy_single_token_is_zero():
    samples = np.zeros((3, 16), dtype=np.int64)
    ent = sample_entropy(samples, vocab_size=50257)
    assert ent == pytest.approx(0.0, abs=1e-6)


def test_score_generative_ppl_uses_mocked_oracle(monkeypatch):
    samples = np.zeros((2, 128), dtype=np.int64)

    def fake_nll(tokens, *, model_name, batch_size=4):
        # 2.0 nats/token everywhere -> PPL = exp(2) ~ 7.389
        return np.full(tokens.shape, 2.0, dtype=np.float32)

    monkeypatch.setattr(genppl_mod, "_oracle_nll_per_token", fake_nll)
    ppl = score_generative_ppl(samples, oracle_name="gpt2-large")
    assert 7.3 < ppl < 7.5


def test_score_generative_ppl_rejects_empty():
    with pytest.raises(ValueError):
        score_generative_ppl(np.zeros((0, 16), dtype=np.int64), oracle_name="gpt2")
