from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping, Optional

import numpy as np

from sticky.core.paths import resolve_against_original_cwd, resolve_optional_path


def _load_np_tokens(path: Path, *, mmap: bool) -> np.ndarray:
    suffix = path.suffix.lower()
    mmap_mode = "r" if bool(mmap) and suffix == ".npy" else None
    loaded = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)

    if isinstance(loaded, np.ndarray):
        return loaded

    if not hasattr(loaded, "files"):
        raise TypeError(f"Unsupported token file object for {path}.")

    try:
        keys = list(loaded.files)
        preferred_keys = ("tokens", "input_ids")
        for key in preferred_keys:
            if key in loaded:
                return np.asarray(loaded[key])

        if len(keys) == 1:
            return np.asarray(loaded[keys[0]])

        raise ValueError(
            f"Could not infer token array from {path}. "
            f"Expected one array or a key in {preferred_keys}, found keys={keys}."
        )
    finally:
        loaded.close()


def load_token_sequences(
    path_like: str,
    *,
    seq_len: int,
    mmap: bool = True,
    max_examples: int = -1,
) -> np.ndarray:
    path = resolve_against_original_cwd(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Token file not found: {path}")

    tokens = _load_np_tokens(path, mmap=bool(mmap))
    if tokens.ndim != 2:
        raise ValueError(
            f"Expected token array with shape [num_sequences, seq_len], got {tokens.shape}."
        )
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError(f"Expected integer token dtype, got {tokens.dtype}.")
    if int(tokens.shape[1]) != int(seq_len):
        raise ValueError(
            f"Expected seq_len={seq_len}, got token shape {tokens.shape}."
        )

    if int(max_examples) > 0:
        tokens = tokens[: int(max_examples)]
    return tokens


def make_openwebtext_iterator(
    *,
    split: str,
    batch_size: int,
    seq_len: int,
    train_tokens_path: str,
    eval_tokens_path: Optional[str] = None,
    seed: int = 0,
    shuffle: bool = True,
    repeat: bool = False,
    drop_remainder: bool = True,
    mmap: bool = True,
    max_examples: int = -1,
) -> Optional[Iterator[Mapping[str, object]]]:
    """Yield pretokenized fixed-length text batches under the existing batch API."""
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    split_key = str(split).lower()
    if split_key == "train":
        path = resolve_optional_path(
            train_tokens_path,
            nullish=(None, "", "null", "None"),
        )
    elif split_key in {"eval", "validation", "val", "test"}:
        path = resolve_optional_path(
            eval_tokens_path,
            nullish=(None, "", "null", "None"),
        )
    else:
        raise ValueError(f"Unsupported split={split!r}.")

    if path is None:
        return None

    tokens = load_token_sequences(
        str(path),
        seq_len=int(seq_len),
        mmap=bool(mmap),
        max_examples=int(max_examples),
    )

    num_examples = int(tokens.shape[0])
    rng = np.random.default_rng(int(seed))

    def _iterator():
        while True:
            indices = np.arange(num_examples, dtype=np.int64)
            if bool(shuffle):
                rng.shuffle(indices)

            for start in range(0, num_examples, int(batch_size)):
                stop = min(start + int(batch_size), num_examples)
                if (stop - start) < int(batch_size) and bool(drop_remainder):
                    break
                batch_indices = indices[start:stop]
                batch = np.asarray(tokens[batch_indices], dtype=np.int32)
                yield {"image": batch}

            if not bool(repeat):
                break

    return iter(_iterator())
