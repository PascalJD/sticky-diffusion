from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp

Array = jnp.ndarray


def _ordered_scalar_table(vocab_size: int, dtype: jnp.dtype) -> Array:
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if vocab_size == 1:
        return jnp.zeros((1, 1), dtype=dtype)

    idx = jnp.arange(vocab_size, dtype=dtype)
    denom = jnp.asarray(vocab_size - 1, dtype=dtype)
    values = -1.0 + (2.0 * idx / denom)
    return values[:, None]


def make_anchor_initializer(
    *,
    name: str,
    vocab_size: int,
    anchor_dim: int,
    init_std: float,
):
    key = str(name).lower()

    if key == "normal":
        return nn.initializers.normal(stddev=float(init_std))

    if key == "ordered_scalar":
        if int(anchor_dim) != 1:
            raise ValueError(
                "ordered_scalar anchors require anchor_dim=1, "
                f"got anchor_dim={anchor_dim}."
            )

        def _init(_key, shape, dtype=jnp.float32):
            expected = (int(vocab_size), int(anchor_dim))
            if tuple(shape) != expected:
                raise ValueError(
                    f"ordered_scalar initializer expected shape {expected}, "
                    f"got {tuple(shape)}."
                )
            return _ordered_scalar_table(int(vocab_size), dtype)

        return _init

    raise ValueError(
        f"Unknown anchor initializer {name!r}. "
        "Expected one of: normal, ordered_scalar."
    )


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
    anchor_init: str = "normal"

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
        init_fn = make_anchor_initializer(
            name=self.anchor_init,
            vocab_size=int(self.vocab_size),
            anchor_dim=int(self.anchor_dim),
            init_std=float(self.init_std),
        )
        table = self.param(
            "table",
            init_fn,
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
