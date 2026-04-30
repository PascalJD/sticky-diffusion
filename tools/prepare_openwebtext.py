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
    # Shard-streaming mode: tokenize ds[start:stop] and stream raw int32 tokens
    # (with eos between docs) into --out-bin. Skips chunking/np.save; the merge
    # step is responsible for converting bins to the (N, seq_len) .npy files.
    # Memory bounded to one batch (~6 MB) instead of the full ~36 GB stream.
    p.add_argument("--start", type=int, default=None,
                   help="Shard mode: first doc index (inclusive).")
    p.add_argument("--stop", type=int, default=None,
                   help="Shard mode: doc index upper bound (exclusive).")
    p.add_argument("--out-bin", type=Path, default=None,
                   help="Shard mode: raw int32 bin output path.")
    args = p.parse_args()
    shard_flags = (args.start is not None, args.stop is not None, args.out_bin is not None)
    if any(shard_flags) and not all(shard_flags):
        p.error("--start, --stop, --out-bin must be passed together (shard mode).")
    return args


def encode_range_to_bin(
    ds,
    tokenizer: GPT2TokenizerFast,
    eos_id: int,
    dtype,
    start: int,
    stop: int,
    batch_size: int,
    out_bin: Path,
    desc: str,
) -> int:
    """Tokenize docs ds[start:stop] and stream tokens (each doc + eos) to out_bin.

    Writes to ``${out_bin}.tmp`` and renames on clean exit so a preempted run
    leaves no half-written final file. Returns the total token count written.
    """
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_bin.with_suffix(out_bin.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    total = 0
    with open(tmp_path, "wb") as f:
        for bstart in tqdm(range(start, stop, batch_size), desc=desc):
            bstop = min(bstart + batch_size, stop)
            texts = ds[bstart:bstop]["text"]
            encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
            n = sum(len(ids) + 1 for ids in encoded)
            buf = np.empty(n, dtype=dtype)
            pos = 0
            for ids in encoded:
                k = len(ids)
                buf[pos : pos + k] = ids
                buf[pos + k] = eos_id
                pos += k + 1
            buf.tofile(f)
            total += n
    tmp_path.replace(out_bin)
    return total


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

    if args.out_bin is not None:
        if args.out_bin.exists() and args.out_bin.stat().st_size > 0:
            print(f"Shard already complete at {args.out_bin} "
                  f"({args.out_bin.stat().st_size} bytes) — skipping.")
            return

    print(f"Loading HuggingFace dataset {args.dataset!r}...")
    ds = load_dataset(args.dataset, split="train")
    n_total = len(ds)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.vocab_size != GPT2_VOCAB_SIZE:
        raise RuntimeError(
            f"Expected GPT-2 vocab_size={GPT2_VOCAB_SIZE}, got {tokenizer.vocab_size}."
        )
    eos_id = int(tokenizer.eos_token_id)

    if args.out_bin is not None:
        start, stop = int(args.start), int(args.stop)
        if start < 0 or stop > n_total or start >= stop:
            raise ValueError(
                f"Invalid shard range [{start}, {stop}) for dataset of size {n_total}."
            )
        print(f"Shard mode: docs [{start}, {stop}) -> {args.out_bin}")
        n_written = encode_range_to_bin(
            ds, tokenizer, eos_id, dtype,
            start=start, stop=stop,
            batch_size=int(args.tokenize_batch_size),
            out_bin=args.out_bin,
            desc=f"tokenize/{args.out_bin.stem}",
        )
        print(f"Wrote {args.out_bin}: {n_written} tokens "
              f"({args.out_bin.stat().st_size} bytes)")
        return

    val_start = max(0, n_total - int(args.val_docs))
    print(f"Total docs: {n_total}  train: {val_start}  val: {n_total - val_start}")

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
