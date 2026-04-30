#!/usr/bin/env python3
"""Merge per-shard raw int32 token bins into the (N, seq_len) .npy files
expected by the sticky-diffusion OpenWebText dataloader.

Memory bound is one chunk (default 64 MB), regardless of total split size:
the output .npy is created via ``numpy.lib.format.open_memmap(mode='w+')``
(writes header + mmaps the data section) and shard bytes are streamed in
through that memmap. Process RSS stays small; the OS page cache absorbs the
work.

Example:
    python tools/merge_owt_shards.py \\
        --train-bins shards/train_0.bin shards/train_1.bin \\
                     shards/train_2.bin shards/train_3.bin \\
        --val-bins shards/val.bin \\
        --out-dir data/openwebtext --seq-len 1024
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


DTYPE = np.int32
ITEMSIZE = np.dtype(DTYPE).itemsize  # 4
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024  # 64 MB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-bins", type=Path, nargs="+", required=True,
                   help="Train shard .bin files in concat order.")
    p.add_argument("--val-bins", type=Path, nargs="+", required=True,
                   help="Val shard .bin file(s).")
    p.add_argument("--out-dir", type=Path, default=Path("data/openwebtext"))
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--keep-bins", action="store_true",
                   help="Keep shard .bin files after merge (default: delete).")
    p.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES,
                   help="Streaming copy chunk size (bytes).")
    return p.parse_args()


def merge_split(
    bins: list[Path],
    out_npy: Path,
    seq_len: int,
    chunk_bytes: int,
) -> tuple[int, int]:
    """Merge `bins` into one (N, seq_len) int32 .npy. Returns (n_seqs, dropped_tokens)."""
    for b in bins:
        if not b.exists():
            raise FileNotFoundError(f"Shard bin not found: {b}")
    total_bytes = sum(b.stat().st_size for b in bins)
    if total_bytes % ITEMSIZE != 0:
        raise ValueError(
            f"Total bytes {total_bytes} not divisible by int32 itemsize {ITEMSIZE} "
            f"— a shard write was likely truncated."
        )
    total_tokens = total_bytes // ITEMSIZE
    n_seqs = total_tokens // seq_len
    if n_seqs == 0:
        raise ValueError(
            f"Token count {total_tokens} < seq_len {seq_len} — nothing to write."
        )
    dropped = total_tokens - n_seqs * seq_len
    keep_bytes = n_seqs * seq_len * ITEMSIZE

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    tmp_npy = out_npy.with_suffix(out_npy.suffix + ".tmp")
    if tmp_npy.exists():
        tmp_npy.unlink()

    out = np.lib.format.open_memmap(
        tmp_npy, mode="w+", dtype=DTYPE, shape=(n_seqs, seq_len),
    )
    flat = out.reshape(-1)
    flat_bytes = flat.view(np.uint8)
    write_pos = 0
    chunk_tokens = max(chunk_bytes // ITEMSIZE, seq_len)

    for b in bins:
        size = b.stat().st_size
        readable = size  # bytes available in this shard
        with open(b, "rb") as f:
            while readable > 0 and write_pos < keep_bytes:
                want = min(chunk_bytes, readable, keep_bytes - write_pos)
                buf = f.read(want)
                if not buf:
                    break
                flat_bytes[write_pos : write_pos + len(buf)] = np.frombuffer(buf, dtype=np.uint8)
                write_pos += len(buf)
                readable -= len(buf)
        if write_pos >= keep_bytes:
            break

    out.flush()
    del flat_bytes, flat, out
    tmp_npy.replace(out_npy)
    return n_seqs, dropped


def main() -> None:
    args = parse_args()
    seq_len = int(args.seq_len)

    train_npy = args.out_dir / "openwebtext_gpt2_1024_train.npy"
    eval_npy = args.out_dir / "openwebtext_gpt2_1024_eval.npy"

    print(f"[train] merging {len(args.train_bins)} shard(s) -> {train_npy}")
    n_train, dropped_train = merge_split(
        args.train_bins, train_npy, seq_len, int(args.chunk_bytes),
    )
    print(f"  wrote shape=({n_train}, {seq_len}) int32; dropped {dropped_train} tail tokens")

    print(f"[eval]  merging {len(args.val_bins)} shard(s) -> {eval_npy}")
    n_eval, dropped_eval = merge_split(
        args.val_bins, eval_npy, seq_len, int(args.chunk_bytes),
    )
    print(f"  wrote shape=({n_eval}, {seq_len}) int32; dropped {dropped_eval} tail tokens")

    if not args.keep_bins:
        for b in (*args.train_bins, *args.val_bins):
            try:
                os.remove(b)
                print(f"  removed shard {b}")
            except FileNotFoundError:
                pass

    print("merge complete.")


if __name__ == "__main__":
    main()
