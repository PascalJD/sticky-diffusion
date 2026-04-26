#!/usr/bin/env python3
"""Prepare OpenWebText for the sticky-diffusion dataloader.

Downloads HuggingFace OpenWebText, tokenizes with the GPT-2 tokenizer
(vocab_size=50257), concatenates documents separated by <|endoftext|>,
chunks into fixed-length sequences, holds out the last 100k documents as
validation (MDLM convention), and saves train/val splits as .npy files
of shape [num_sequences, seq_len] at data/openwebtext/{train,val}.npy.

Example:
    python tools/prepare_openwebtext.py
    python tools/prepare_openwebtext.py --out-dir data/openwebtext --seq-len 1024
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast


GPT2_VOCAB_SIZE = 50257
DEFAULT_DATASET = "Skylion007/openwebtext"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("data/openwebtext"))
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--val-docs", type=int, default=100_000,
                   help="Number of trailing docs held out as validation.")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="HuggingFace dataset id for OpenWebText.")
    p.add_argument("--tokenize-batch-size", type=int, default=1000)
    # GPT-2 vocab is 50257 > int16 max (32767), so int16 would overflow.
    # uint16 fits but most downstream code assumes signed; stick with int32.
    p.add_argument("--dtype", choices=("int32",), default="int32")
    return p.parse_args()


def encode_range(
    ds,
    tokenizer: GPT2TokenizerFast,
    eos_id: int,
    dtype,
    start: int,
    stop: int,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    """Tokenize docs ds[start:stop], joining with <eos>, return 1-D token array."""
    chunks: list[np.ndarray] = []
    for bstart in tqdm(range(start, stop, batch_size), desc=desc):
        bstop = min(bstart + batch_size, stop)
        texts = ds[bstart:bstop]["text"]
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        total = sum(len(ids) + 1 for ids in encoded)
        buf = np.empty(total, dtype=dtype)
        pos = 0
        for ids in encoded:
            n = len(ids)
            buf[pos : pos + n] = ids
            buf[pos + n] = eos_id
            pos += n + 1
        chunks.append(buf)
    if not chunks:
        return np.empty(0, dtype=dtype)
    return np.concatenate(chunks)


def chunk_into_sequences(tokens: np.ndarray, seq_len: int) -> np.ndarray:
    n_seqs = int(tokens.size) // int(seq_len)
    return tokens[: n_seqs * seq_len].reshape(n_seqs, seq_len)


def main() -> None:
    args = parse_args()
    dtype = np.int32

    print(f"Loading HuggingFace dataset {args.dataset!r}...")
    ds = load_dataset(args.dataset, split="train")
    n_total = len(ds)
    val_start = max(0, n_total - int(args.val_docs))
    print(f"Total docs: {n_total}  train: {val_start}  val: {n_total - val_start}")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.vocab_size != GPT2_VOCAB_SIZE:
        raise RuntimeError(
            f"Expected GPT-2 vocab_size={GPT2_VOCAB_SIZE}, got {tokenizer.vocab_size}."
        )
    eos_id = int(tokenizer.eos_token_id)

    train_tokens = encode_range(
        ds, tokenizer, eos_id, dtype,
        start=0, stop=val_start,
        batch_size=int(args.tokenize_batch_size), desc="tokenize/train",
    )
    val_tokens = encode_range(
        ds, tokenizer, eos_id, dtype,
        start=val_start, stop=n_total,
        batch_size=int(args.tokenize_batch_size), desc="tokenize/val",
    )

    seq_len = int(args.seq_len)
    train_seqs = chunk_into_sequences(train_tokens, seq_len)
    val_seqs = chunk_into_sequences(val_tokens, seq_len)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.npy"
    val_path = out_dir / "val.npy"
    np.save(train_path, train_seqs)
    np.save(val_path, val_seqs)

    print(
        f"Saved {train_path}: shape={tuple(train_seqs.shape)} "
        f"tokens={int(train_seqs.size)}"
    )
    print(
        f"Saved {val_path}: shape={tuple(val_seqs.shape)} "
        f"tokens={int(val_seqs.size)}"
    )


if __name__ == "__main__":
    main()
