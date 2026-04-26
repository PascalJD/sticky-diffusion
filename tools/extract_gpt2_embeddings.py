"""Extract a pretrained GPT-2 token embedding matrix (wte) to an .npz file.

Used as the initial SJD anchor table for OpenWebText SJD training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_wte_from_hf(model_name: str = "gpt2"):
    from transformers import AutoModel  # lazy import

    model = AutoModel.from_pretrained(model_name)
    wte = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)
    return wte, model_name


def save_gpt2_wte(
    out_path: Path,
    model_name: str = "gpt2",
    *,
    expected_vocab_size: int | None = None,
    expected_dim: int | None = None,
) -> Path:
    out_path = Path(out_path)
    wte, resolved_name = _load_wte_from_hf(model_name)
    if expected_vocab_size is not None and wte.shape[0] != int(expected_vocab_size):
        raise ValueError(
            f"loaded wte has vocab_size={wte.shape[0]}, expected {expected_vocab_size}"
        )
    if expected_dim is not None and wte.shape[1] != int(expected_dim):
        raise ValueError(
            f"loaded wte has dim={wte.shape[1]}, expected {expected_dim}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, wte=wte, model_name=np.array(resolved_name))
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/anchors/gpt2_wte.npz"))
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--expected-vocab-size", type=int, default=50257)
    p.add_argument("--expected-dim", type=int, default=768)
    args = p.parse_args()
    out = save_gpt2_wte(
        out_path=args.out,
        model_name=args.model,
        expected_vocab_size=args.expected_vocab_size,
        expected_dim=args.expected_dim,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
