#!/usr/bin/env python3
"""Prepare Text8 for the sticky-diffusion dataloader.

Downloads the Mahoney text8 corpus (100M chars of lowercase a-z + space),
splits it into the standard 90M/5M/5M train/val/test character ranges, encodes
each with the CANDI text8 character map (a=0..space=26), chunks into fixed-length
sequences, and saves train/val/test splits as .npy files of shape
[num_sequences, seq_len] at data/text8/{train,val,test}.npy.

It also writes:
  - data/text8/char_map.json       -- the verbatim char->int map used at train
                                       time (the eval decode must match this).
  - data/text8/test_vocab_len5.json -- the valid-word reference dictionary: the
                                       set of words (length > 4) appearing in the
                                       test split. This IS the metric's vocabulary
                                       (CANDI calculate_word_dictionary_match), not
                                       a system dictionary.

Example:
    python tools/prepare_text8.py
    python tools/prepare_text8.py --out-dir data/text8 --seq-len 128
"""
from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from sticky.data.text8 import TEXT8_CHARS, encode_text8

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"

# Standard text8 split: 90M train / 5M val / 5M test characters.
DEFAULT_TRAIN_CHARS = 90_000_000
DEFAULT_VAL_CHARS = 5_000_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("data/text8"))
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--train-chars", type=int, default=DEFAULT_TRAIN_CHARS)
    p.add_argument("--val-chars", type=int, default=DEFAULT_VAL_CHARS)
    p.add_argument(
        "--text8-path",
        type=Path,
        default=None,
        help="Path to an existing text8 file or text8.zip; downloaded if absent.",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=-1,
        help="Cap on total characters read (for a tiny local fixture); -1 = all.",
    )
    return p.parse_args()


def load_text8(text8_path: Path | None, out_dir: Path) -> str:
    """Return the raw text8 string, downloading/extracting the zip if needed."""
    if text8_path is not None and text8_path.suffix != ".zip" and text8_path.exists():
        return text8_path.read_text(encoding="ascii")

    zip_path = text8_path if (text8_path and text8_path.suffix == ".zip") else out_dir / "text8.zip"
    if not zip_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading text8 from {TEXT8_URL} -> {zip_path} ...")
        urllib.request.urlretrieve(TEXT8_URL, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("text8") as f:
            return f.read().decode("ascii")


def chunk_into_sequences(tokens: np.ndarray, seq_len: int) -> np.ndarray:
    n_seqs = int(tokens.size) // int(seq_len)
    return tokens[: n_seqs * seq_len].reshape(n_seqs, seq_len)


def build_test_vocab(test_text: str, *, min_len_exclusive: int = 4) -> list[str]:
    """Set of words in the test split with length > ``min_len_exclusive`` (>=5).

    This replicates CANDI's calculate_word_dictionary_match dictionary: the
    reference vocabulary is built from the test split itself, keeping only words
    longer than 4 characters.
    """
    vocab = {w for w in test_text.split() if len(w) > min_len_exclusive}
    return sorted(vocab)


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    text = load_text8(args.text8_path, out_dir)
    if int(args.max_chars) > 0:
        text = text[: int(args.max_chars)]
    print(f"Total characters: {len(text)}")

    train_end = int(args.train_chars)
    val_end = train_end + int(args.val_chars)
    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    test_text = text[val_end:]
    print(
        f"Split chars  train: {len(train_text)}  val: {len(val_text)}  "
        f"test: {len(test_text)}"
    )

    seq_len = int(args.seq_len)
    splits = {"train": train_text, "val": val_text, "test": test_text}
    for name, split_text in splits.items():
        tokens = encode_text8(split_text)
        seqs = chunk_into_sequences(tokens, seq_len)
        path = out_dir / f"{name}.npy"
        np.save(path, seqs)
        print(f"Saved {path}: shape={tuple(seqs.shape)} tokens={int(seqs.size)}")

    char_map_path = out_dir / "char_map.json"
    char_map_path.write_text(
        json.dumps({c: i for i, c in enumerate(TEXT8_CHARS)}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {char_map_path} ({len(TEXT8_CHARS)} symbols)")

    vocab = build_test_vocab(test_text)
    vocab_path = out_dir / "test_vocab_len5.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {vocab_path}: {len(vocab)} words (len > 4)")


if __name__ == "__main__":
    main()
