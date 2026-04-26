"""Verification script for the CDCD anchor-normalization fix.

Run from the `sticky` conda env:
    python tools/verify_anchor_normalization.py

Checks:
  1. With normalize_at_use=True, every anchor read has L2 norm sqrt(d).
  2. With normalize_at_use=False, reads are byte-identical to the raw param.
  3. Gradients flow through the normalization (finite, nonzero).
  4. Underlying-parameter scale is absorbed: scaling table by 10x leaves
     the embedding output unchanged. This is the actual mechanism that
     prevents unbounded growth of ||a_k||.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sticky.models.sjd.anchors import AnchorTableConfig, TokenAnchors


def main() -> None:
    vocab, dim = 8, 4
    key = jax.random.PRNGKey(0)
    ids = jnp.arange(vocab)

    cfg_on = AnchorTableConfig(
        family="normal", vocab_size=vocab, anchor_dim=dim,
        init_std=1.0, seed=0, normalize_at_use=True,
    )
    mod_on = TokenAnchors(config=cfg_on)
    params_on = mod_on.init(key, ids)
    embeds = mod_on.apply(params_on, ids)
    table = mod_on.apply(params_on, method=mod_on.table_float)
    expected = float(np.sqrt(dim))
    norms_e = np.linalg.norm(np.asarray(embeds), axis=-1)
    norms_t = np.linalg.norm(np.asarray(table), axis=-1)
    print(f"[on] expected row norm = {expected:.6f}")
    print(f"[on] embed row norms   = {norms_e.round(6)}")
    print(f"[on] table row norms   = {norms_t.round(6)}")
    assert np.allclose(norms_e, expected, atol=1e-5)
    assert np.allclose(norms_t, expected, atol=1e-5)
    assert np.allclose(np.asarray(embeds), np.asarray(table))
    print("[on] OK\n")

    cfg_off = AnchorTableConfig(
        family="normal", vocab_size=vocab, anchor_dim=dim,
        init_std=1.0, seed=0, normalize_at_use=False,
    )
    mod_off = TokenAnchors(config=cfg_off)
    params_off = mod_off.init(key, ids)
    embeds_off = mod_off.apply(params_off, ids)
    table_off = mod_off.apply(params_off, method=mod_off.table_float)
    raw = np.asarray(params_off["params"]["table"])
    assert np.allclose(np.asarray(embeds_off), raw)
    assert np.allclose(np.asarray(table_off), raw)
    print(f"[off] raw row norms = {np.linalg.norm(raw, axis=-1).round(4)}")
    print("[off] OK -- raw table passes through unchanged\n")

    def loss_fn(params):
        e = mod_on.apply(params, ids)
        return jnp.sum(e ** 2)
    grad = jax.grad(loss_fn)(params_on)
    g = np.asarray(grad["params"]["table"])
    print(f"[grad] grad row norms = {np.linalg.norm(g, axis=-1).round(4)}")
    assert np.all(np.isfinite(g))
    assert np.linalg.norm(g) > 0
    print("[grad] OK -- finite, nonzero grads through normalization\n")

    scaled = jax.tree.map(lambda p: p * 10.0, params_on)
    embeds_scaled = mod_on.apply(scaled, ids)
    diff = float(np.max(np.abs(np.asarray(embeds) - np.asarray(embeds_scaled))))
    print(f"[invariance] max |embed - embed_scaled_10x| = {diff:.2e}")
    assert np.allclose(np.asarray(embeds), np.asarray(embeds_scaled), atol=1e-5)
    print("[invariance] OK -- normalization absorbs underlying-parameter scale\n")

    print("All anchor-normalization checks passed.")


if __name__ == "__main__":
    main()
