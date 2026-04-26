"""Generative-PPL + sample-entropy scorer for text-diffusion evaluation.

Follows the CANDI (Pynadath et al., 2025, arXiv 2510.22510) protocol: evaluate
samples by scoring them under an off-the-shelf GPT-2 variant and reporting the
(genPPL, entropy) frontier across NFE budgets. This is used for OWT SJD when
deriving a proper ELBO is out of scope.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

_ORACLE_CACHE: dict[str, tuple[Any, Any, Any]] = {}


def sample_entropy(samples: np.ndarray, *, vocab_size: int) -> float:
    """Token-wise Shannon entropy (nats) of the empirical sample distribution.

    Computes H(p) where p is the empirical probability over vocab indices.
    Random-uniform samples give log(vocab_size); degenerate (constant) samples
    give 0.
    """
    samples = np.asarray(samples).reshape(-1)
    if samples.size == 0:
        return 0.0
    counts = np.bincount(samples, minlength=int(vocab_size)).astype(np.float64)
    total = float(counts.sum())
    probs = counts / max(total, 1.0)
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log(nonzero)))


def _get_oracle(model_name: str):
    if model_name in _ORACLE_CACHE:
        return _ORACLE_CACHE[model_name]
    import torch  # lazy import
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    _ORACLE_CACHE[model_name] = (tokenizer, model, device)
    return tokenizer, model, device


def _oracle_nll_per_token(
    tokens: np.ndarray,
    *,
    model_name: str,
    batch_size: int = 4,
) -> np.ndarray:
    """Return NLL (nats) of each token under a causal LM, shape matches `tokens`.

    Uses shifted teacher-forcing: logits at position i predict tokens[i+1].
    Position 0 is assigned the mean NLL of the rest so the output shape matches
    the input.
    """
    import torch

    _, model, device = _get_oracle(model_name)
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 2:
        raise ValueError(f"expected shape (N, S), got {tokens.shape}")
    n, seq_len = tokens.shape
    out = np.zeros_like(tokens, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            batch = torch.from_numpy(tokens[start:stop]).to(device)
            logits = model(batch).logits  # (b, S, V)
            logp = torch.nn.functional.log_softmax(
                logits[:, :-1, :].float(), dim=-1
            )
            targets = batch[:, 1:]
            token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            nll = (-token_logp).cpu().numpy().astype(np.float32)
            # Pad position 0 with the mean of the remaining positions.
            row_mean = nll.mean(axis=1, keepdims=True) if seq_len > 1 else np.zeros(
                (nll.shape[0], 1), dtype=np.float32
            )
            padded = np.concatenate([row_mean, nll], axis=1)
            out[start:stop] = padded
    return out


def score_generative_ppl(
    samples: np.ndarray,
    *,
    oracle_name: str = "gpt2-large",
    batch_size: int = 4,
) -> float:
    """Mean generative perplexity: exp(mean NLL per token under the oracle).

    `samples` is an integer array of shape (N, S) over the GPT-2 vocabulary.
    """
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError(
            f"score_generative_ppl expects nonempty (N, S), got {samples.shape}"
        )
    nll = _oracle_nll_per_token(samples, model_name=oracle_name, batch_size=batch_size)
    return float(math.exp(float(nll.mean())))


def genppl_entropy_report(
    samples_by_nfe: dict[int, np.ndarray],
    *,
    vocab_size: int,
    oracle_name: str = "gpt2-large",
    batch_size: int = 4,
) -> dict[str, float]:
    """Aggregate (genPPL, entropy) metrics per NFE budget into flat scalars.

    Returned keys are `eval/genppl@{nfe}` and `eval/entropy@{nfe}` so they can be
    logged directly to W&B alongside training-loop metrics.
    """
    report: dict[str, float] = {}
    for nfe, samples in sorted(samples_by_nfe.items()):
        report[f"eval/genppl@{int(nfe)}"] = score_generative_ppl(
            samples, oracle_name=oracle_name, batch_size=batch_size
        )
        report[f"eval/entropy@{int(nfe)}"] = sample_entropy(
            samples, vocab_size=vocab_size
        )
    return report


def list_nfe_sweep(cfg: Any, default: Sequence[int] = (16, 32, 64, 128)) -> list[int]:
    """Read the NFE sweep from an eval config, falling back to the default."""
    getter = getattr(cfg, "get", None)
    values = None
    if callable(getter):
        values = getter("nfe_sweep", None)
    elif isinstance(cfg, dict):
        values = cfg.get("nfe_sweep")
    if values is None:
        return [int(v) for v in default]
    return [int(v) for v in values]
