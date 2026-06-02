from __future__ import annotations

from typing import Iterator, Mapping, Optional

import numpy as np

from sticky.data.openwebtext import make_openwebtext_iterator


# Text8 character vocabulary. Ordering is VERBATIM from the CANDI text8 tokenizer:
# the 26 lowercase letters a..z followed by a single space, so that
# char->int = index into this string (a=0, ..., z=25, space=26). A permuted map
# silently corrupts valid-word decoding, so this constant is the single source of
# truth shared by the data prep tool, the task, and the valid-word eval.
TEXT8_CHARS = "abcdefghijklmnopqrstuvwxyz "
TEXT8_VOCAB_SIZE = len(TEXT8_CHARS)  # 27

CHAR_TO_ID: dict[str, int] = {c: i for i, c in enumerate(TEXT8_CHARS)}
ID_TO_CHAR: dict[int, str] = {i: c for i, c in enumerate(TEXT8_CHARS)}


def encode_text8(text: str) -> np.ndarray:
    """Map a text8 string (lowercase a-z and spaces) to an int32 token array."""
    return np.fromiter(
        (CHAR_TO_ID[c] for c in text if c in CHAR_TO_ID),
        dtype=np.int32,
        count=-1,
    )


def decode_ids(ids) -> str:
    """Map a 1-D sequence of token ids back to a text8 string.

    Out-of-range ids are dropped (they cannot occur for a faithful 27-symbol
    model, but sampling diagnostics may pass padded/unfilled markers).
    """
    arr = np.asarray(ids).reshape(-1)
    return "".join(ID_TO_CHAR[int(i)] for i in arr if int(i) in ID_TO_CHAR)


def make_text8_iterator(
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
    """Yield pretokenized fixed-length text8 batches.

    The on-disk format (``.npy`` arrays of shape ``[num_sequences, seq_len]``)
    and the batch contract (``{"image": batch}``) are identical to OpenWebText,
    so this delegates to the shared iterator rather than duplicating it.
    """
    return make_openwebtext_iterator(
        split=split,
        batch_size=batch_size,
        seq_len=seq_len,
        train_tokens_path=train_tokens_path,
        eval_tokens_path=eval_tokens_path,
        seed=seed,
        shuffle=shuffle,
        repeat=repeat,
        drop_remainder=drop_remainder,
        mmap=mmap,
        max_examples=max_examples,
    )
