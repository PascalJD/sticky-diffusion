"""Precompute per-anchor frequency weights w(a) = -log p_0(a) for SJD.

Tally token occurrences over a training-split token file (or a glob of files),
apply add-alpha smoothing so every anchor has p_0(a) > 0, optionally clip and
mean-normalize, and save the resulting log_w array (= log w(a)) as a .npz
alongside the embedding table for use by the SJD frequency-weighted hazard
path.

The cached array is `log_w` of shape (vocab_size,) with values
    log_w[a] = log( w(a) )    where    w(a) = -log p_0(a)

**Normalization caveat.** With ``--normalize-mean`` (the default), we divide
by the *uniform* mean ``(1/K) sum_a w(a)`` so ``mean_uniform(w) = 1``. This is
an apples-to-apples ratio across anchors but does *not* preserve the
per-token expected hazard, which is

    E_{a ~ p_0}[ beta(t) w(a) ] = beta(t) * H(p_0)   (entropy in nats)

After dividing by ``mean_uniform(-log p_0)``, the empirical per-token hazard
is rescaled by ``H(p_0) / mean_uniform(-log p_0)`` -- typically in the range
0.5-0.8 for natural-language vocabularies, so a freq-weighted run sees a
~30-50% lower effective hazard than the anchor-agnostic baseline at the same
schedule. If you want ``E_{p_0}[w] = 1`` instead (preserving the empirical
hazard scale at the cost of a less symmetric weight distribution), divide by
``sum_a p_0(a) w(a) = H(p_0)`` manually after running this script, or modify
``compute_log_w`` to expose an entropy-normalization mode.

Example (OpenWebText GPT-2):
    python tools/compute_anchor_frequencies.py \\
        --tokens data/openwebtext/train.npy \\
        --vocab-size 50257 \\
        --out data/anchors/openwebtext_log_w.npz
"""
from __future__ import annotations

import argparse
import glob as _glob
from pathlib import Path
from typing import Iterable

import numpy as np


def _load_tokens(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False, mmap_mode="r")
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as f:
            keys = list(f.files)
            for key in ("tokens", "input_ids"):
                if key in f:
                    return np.asarray(f[key])
            if len(keys) == 1:
                return np.asarray(f[keys[0]])
            raise ValueError(
                f"Could not infer token array from {path}; found keys={keys}."
            )
    raise ValueError(f"Unsupported token file extension: {path}")


_BINCOUNT_CHUNK_TOKENS = 100_000_000


def _tally_counts(paths: Iterable[Path], vocab_size: int) -> np.ndarray:
    counts = np.zeros((int(vocab_size),), dtype=np.int64)
    for p in paths:
        tokens = _load_tokens(p)
        if not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError(f"{p}: expected integer tokens, got {tokens.dtype}")
        flat = np.asarray(tokens).reshape(-1)
        if flat.size == 0:
            continue
        for i in range(0, int(flat.size), _BINCOUNT_CHUNK_TOKENS):
            block = np.asarray(flat[i : i + _BINCOUNT_CHUNK_TOKENS], dtype=np.int64)
            if int(block.min()) < 0 or int(block.max()) >= int(vocab_size):
                raise ValueError(
                    f"{p}: tokens out of range [0, {vocab_size}); "
                    f"chunk min={int(block.min())} max={int(block.max())}"
                )
            counts += np.bincount(block, minlength=int(vocab_size))[: int(vocab_size)]
    return counts


def compute_log_w(
    counts: np.ndarray,
    *,
    smoothing_alpha: float = 1.0,
    normalize_mean: bool = True,
    clip_min: float | None = 0.1,
    clip_max: float | None = 10.0,
) -> dict:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError(f"counts must be 1-D, got shape {counts.shape}")
    if smoothing_alpha < 0.0:
        raise ValueError(f"smoothing_alpha must be >= 0, got {smoothing_alpha}")
    smoothed = counts + float(smoothing_alpha)
    if not np.all(smoothed > 0.0):
        raise ValueError(
            "After smoothing some counts are still <=0. Increase --smoothing-alpha."
        )
    p0 = smoothed / smoothed.sum()
    w = -np.log(p0)  # weight w(a) = -log p_0(a) (positive)

    if clip_min is not None or clip_max is not None:
        lo = float(clip_min) if clip_min is not None else float(w.min())
        hi = float(clip_max) if clip_max is not None else float(w.max())
        w = np.clip(w, lo, hi)

    if normalize_mean:
        mean_w = float(w.mean())
        if mean_w <= 0.0:
            raise ValueError(f"mean(w) <= 0 after clip; got {mean_w}")
        w = w / mean_w

    log_w = np.log(w).astype(np.float32)
    return {
        "log_w": log_w,
        "w": w.astype(np.float32),
        "p0": p0.astype(np.float64),
        "counts_total": np.asarray([int(counts.sum())], dtype=np.int64),
        "smoothing_alpha": np.float32(smoothing_alpha),
        "normalize_mean": np.bool_(normalize_mean),
        "clip_min": np.float32(clip_min) if clip_min is not None else np.float32(np.nan),
        "clip_max": np.float32(clip_max) if clip_max is not None else np.float32(np.nan),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", required=True, type=str,
        help="Path to a .npy/.npz token file or a glob (e.g. 'data/owt/train_*.npy').",
    )
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--no-normalize", dest="normalize_mean", action="store_false")
    parser.add_argument("--clip-min", type=float, default=0.1)
    parser.add_argument("--clip-max", type=float, default=10.0)
    parser.set_defaults(normalize_mean=True)
    args = parser.parse_args()

    matches = sorted(_glob.glob(args.tokens))
    if not matches:
        # Treat as a literal path if glob didn't match.
        candidate = Path(args.tokens)
        if not candidate.exists():
            raise FileNotFoundError(f"No files matched --tokens={args.tokens!r}")
        matches = [str(candidate)]
    paths = [Path(m) for m in matches]
    print(f"counting tokens across {len(paths)} file(s) into {args.vocab_size} bins...")

    counts = _tally_counts(paths, vocab_size=args.vocab_size)
    print(f"  total tokens: {int(counts.sum())}")
    print(f"  unique anchors observed: {int((counts > 0).sum())} / {args.vocab_size}")

    blob = compute_log_w(
        counts,
        smoothing_alpha=args.smoothing_alpha,
        normalize_mean=args.normalize_mean,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )
    log_w = blob["log_w"]
    print(
        f"  log_w summary: shape={log_w.shape} "
        f"min={float(log_w.min()):.4f} max={float(log_w.max()):.4f} "
        f"mean={float(log_w.mean()):.4f}"
    )
    print(
        f"  w summary:     min={float(blob['w'].min()):.4f} "
        f"max={float(blob['w'].max()):.4f} mean={float(blob['w'].mean()):.4f}"
    )

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **blob)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
