"""Valid-word frontier scorer for Text8 diffusion evaluation.

Replicates CANDI's `text_metrics.py::calculate_word_dictionary_match`: the
reference vocabulary is the set of words (length > 4) appearing in the text8
**test split**, NOT a system dictionary. Each generated sample is scored as

    score = |unique generated words that are in the vocab| / (total tokens)

and the headline is the mean of that ratio over all samples. The temperature x
NFE frontier reports the max-along-temperature mean per NFE.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from sticky.core.paths import resolve_against_original_cwd


def load_test_vocab(path: str) -> set[str]:
    """Load the persisted test-split reference vocabulary (len > 4 words)."""
    resolved = resolve_against_original_cwd(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Text8 valid-word vocabulary not found: {resolved}. "
            "Build it with tools/prepare_text8.py (writes test_vocab_len5.json)."
        )
    words = json.loads(Path(resolved).read_text(encoding="utf-8"))
    return {str(w) for w in words}


def load_char_map(path: str) -> dict[int, str]:
    """Load the persisted char->int map and invert it to int->char."""
    resolved = resolve_against_original_cwd(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Text8 char map not found: {resolved}. "
            "Build it with tools/prepare_text8.py (writes char_map.json)."
        )
    char_to_id = json.loads(Path(resolved).read_text(encoding="utf-8"))
    return {int(i): str(c) for c, i in char_to_id.items()}


def decode_with_map(ids: Sequence[int], id_to_char: Mapping[int, str]) -> str:
    """Map a 1-D id sequence to text via an explicit int->char map."""
    arr = np.asarray(ids).reshape(-1)
    return "".join(id_to_char[int(i)] for i in arr if int(i) in id_to_char)


def word_dictionary_match(text: str, vocab: set[str]) -> float:
    """CANDI per-sample score: unique matched words / total token count.

    The length filter lives on the vocab side (the vocab already contains only
    len>4 words), so short generated tokens simply never match. A sample with no
    tokens scores 0.
    """
    tokens = text.split()
    counter = len(tokens)
    if counter == 0:
        return 0.0
    gen_words = set(tokens)
    matched = len(gen_words & vocab)
    return matched / counter


def score_samples(
    samples: np.ndarray,
    *,
    id_to_char: Mapping[int, str],
    vocab: set[str],
) -> float:
    """Mean per-sample `word_dictionary_match` over a batch of token rows."""
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.shape[0] == 0:
        return 0.0
    scores = [
        word_dictionary_match(decode_with_map(row, id_to_char), vocab)
        for row in samples
    ]
    return float(np.mean(scores))


def valid_word_frontier_report(
    samples_by_nfe_temp: Mapping[tuple[int, float], np.ndarray],
    *,
    id_to_char: Mapping[int, str],
    vocab: set[str],
) -> dict[str, float]:
    """Aggregate the temperature x NFE frontier into flat scalars.

    `samples_by_nfe_temp` maps (nfe, temperature) -> integer sample array
    (N, S). Emits per-gridpoint `eval/valid_word@nfe{N}_temp{T}` and the
    max-along-temperature headline `eval/valid_word_max@nfe{N}`.
    """
    report: dict[str, float] = {}
    per_nfe: dict[int, list[float]] = {}
    for (nfe, temp), samples in sorted(samples_by_nfe_temp.items()):
        mean_score = score_samples(samples, id_to_char=id_to_char, vocab=vocab)
        report[f"eval/valid_word@nfe{int(nfe)}_temp{float(temp):g}"] = mean_score
        per_nfe.setdefault(int(nfe), []).append(mean_score)
    for nfe, scores in sorted(per_nfe.items()):
        report[f"eval/valid_word_max@nfe{int(nfe)}"] = float(max(scores))
    return report


def list_grid(cfg, key: str, default: Sequence) -> list:
    """Read a sweep list from an eval config, falling back to the default."""
    getter = getattr(cfg, "get", None)
    values = getter(key, None) if callable(getter) else None
    if values is None:
        return list(default)
    return list(values)
