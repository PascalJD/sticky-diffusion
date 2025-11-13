# src/sticky/core/anchors.py
from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp

Array = jnp.ndarray

@dataclass(frozen=True)
class AnalogBitsAnchors:
    """
    Encode discrete bins {0,...,L-1} into analog bits in {-1, +1}^d.
    """
    L: int
    d: int | None = None
    gray: bool = False

    def __post_init__(self):
        d = int(self.d) if self.d is not None else int(jnp.ceil(jnp.log2(self.L)))
        object.__setattr__(self, "d", d)
        table = self._build_code_table(self.L, d, self.gray)
        object.__setattr__(self, "table", table)

    @staticmethod
    def _build_code_table(L: int, d: int, gray: bool) -> Array:
        idx = jnp.arange(L, dtype=jnp.int32)
        code = idx ^ (idx >> 1) if gray else idx  # Gray vs. binary
        bits = ((code[:, None] >> jnp.arange(d)) & 1).astype(jnp.float32)
        bits = jnp.flip(bits, axis=1)
        # analog mapping: {0,1} -> {-1,+1}
        return 2.0 * bits - 1.0

    def discretize_pixels(self, x: Array, L: int | None = None) -> Array:
        """
        x: (B, H, W) or (B, H*W) in [0,1]; 
        returns int indices in {0,...,L-1} with shape (B, H, W).
        We use uniform bins on [0,1]: k = floor(L * x), clipped to L-1.
        """
        L = L or self.L
        if x.ndim == 2:
            # assume (B, H*W) -> reshape to (B, 28, 28) by default
            B, N = x.shape
            H = W = int(jnp.sqrt(N))
            x = x.reshape(B, H, W)
        k = jnp.floor(x * L).astype(jnp.int32)
        k = jnp.clip(k, 0, L - 1)
        return k  # (B,H,W)

    def indices_to_vectors(self, k: Array) -> Array:
        """
        k: (B,H,W) integer indices -> analog bits (B,H,W,d) in {-1,1}.
        """
        flat = k.reshape(-1)
        vec_flat = self.table[flat]
        return vec_flat.reshape((*k.shape, self.d))

    def encode_from_pixels(self, x: Array) -> tuple[Array, Array]:
        k = self.discretize_pixels(x, L=self.L)
        a = self.indices_to_vectors(k)
        return k, a

    def vectors_to_indices(self, v: Array) -> Array:
        B, H, W, d = v.shape
        table_T = self.table.T  # (d, L)
        sims = jnp.tensordot(v, table_T, axes=[-1, 0])  # (B,H,W,L)
        return jnp.argmax(sims, axis=-1).astype(jnp.int32)

    @property
    def table_float(self) -> Array:
        """(L,d) analog code table in {-1,1}."""
        return self.table

    def indices_to_pixels(self, k: Array) -> Array:
        """Bin centers in [0,1] for integer indices k."""
        return (k.astype(jnp.float32) + 0.5) / float(self.L)

    def vectors_to_pixels(self, v: Array) -> Array:
        """Map analog bits to nearest anchor index, then to bin centers [0,1]."""
        k = self.vectors_to_indices(v)  # (B,H,W)
        return self.indices_to_pixels(k)