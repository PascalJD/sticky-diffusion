"""Loader for pretrained SJD anchor tables stored as .npz files.

Produced by `tools/extract_gpt2_embeddings.py`. The expected payload is an
array under key ``wte`` with shape ``(vocab_size, anchor_dim)``.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray


def load_pretrained_anchor_table(
    *,
    path: str,
    vocab_size: int,
    anchor_dim: int,
    dtype=jnp.float32,
) -> Array:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"pretrained anchor file not found: {p}")
    with np.load(p) as data:
        if "wte" not in data.files:
            raise ValueError(
                f"pretrained anchor file {p} must contain a 'wte' array; "
                f"found {list(data.files)}"
            )
        arr = np.asarray(data["wte"], dtype=np.float32)
    expected = (int(vocab_size), int(anchor_dim))
    if arr.shape != expected:
        raise ValueError(
            f"pretrained anchor shape mismatch: got {arr.shape}, expected {expected}"
        )
    return jnp.asarray(arr, dtype=dtype)
