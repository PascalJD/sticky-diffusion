from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp

Array = jnp.ndarray


@dataclass(frozen=True)
class AnchorsSpec:
    """Lightweight description of an anchor table."""

    vocab_size: int
    anchor_dim: int


class TokenAnchors(nn.Module):
    """Learnable (or fixed) anchor lookup table.

    Parameters live in the model's `params` collection, so they can be trained
    jointly with the classifier.
    """

    vocab_size: int
    anchor_dim: int

    # If False, we still *store* the table as a parameter (so it's in checkpoints),
    # but stop gradients through it so optimizers won't update it.
    learnable: bool = True

    init_std: float = 1.0

    @nn.compact
    def __call__(self, ids: Array) -> Array:
        """Lookup anchors for integer ids.

        Args:
            ids: int32 array of shape (...)
        Returns:
            anchors: float32 array of shape (..., anchor_dim)
        """
        table = self.param(
            "table",
            nn.initializers.normal(stddev=float(self.init_std)),
            (int(self.vocab_size), int(self.anchor_dim)),
            jnp.float32,
        )
        if not self.learnable:
            table = jax.lax.stop_gradient(table)
        return jnp.take(table, ids, axis=0)

    def table_float(self) -> Array:
        table = self.get_variable("params", "table")
        return table.astype(jnp.float32)



@struct.dataclass
class AnchorTable:
    """Frozen view of an anchor table for sampling.

    This is a PyTree (via flax.struct.dataclass), so it can safely flow through
    `jax.jit`/`jax.lax.fori_loop`.
    """

    table_float: Array

    @property
    def L(self) -> int:
        return int(self.table_float.shape[0])

    @property
    def d(self) -> int:
        return int(self.table_float.shape[1])

