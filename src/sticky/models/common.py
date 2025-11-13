# src/sticky/models/common.py
from __future__ import annotations
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


def timestep_embedding(t: jnp.ndarray, dim: int=64, max_period: int=10_000) -> jnp.ndarray:
    """
    t: shape (B,) in [0, T]; returns (B, dim)
    """
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(0, half) / half)
    args = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        emb = jnp.pad(emb, ((0,0), (0,1)))
    return emb

def _reduce_time(t: jnp.ndarray) -> jnp.ndarray:
    if t.ndim == 1:
        return t
    if t.ndim == 3:
        return jnp.mean(t, axis=(1, 2))
    raise ValueError(f"t must be (B,) or (B,H,W); got {t.shape}.")


class TimeConditioner(nn.Module):

    temb_dim: int = 64
    mlp_mult: int = 4
    out_dim: int = 128

    @nn.compact
    def __call__(self, t_vec: jnp.ndarray) -> jnp.ndarray:
        """
        t_vec: (B,). Returns (B, out_dim).
        """
        temb = timestep_embedding(t_vec, dim=self.temb_dim)
        h = nn.Dense(self.out_dim * self.mlp_mult)(nn.silu(temb))
        h = nn.Dense(self.out_dim)(nn.silu(h))
        return h


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation
    """
    ch: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, tfeat: jnp.ndarray) -> jnp.ndarray:
        """
        x: (B,H,W,C), tfeat: (B, C_cond)
        """
        gb = nn.Dense(2 * self.ch)(nn.silu(tfeat))
        gamma, beta = jnp.split(gb, 2, axis=-1)
        return x * (1.0 + gamma[:, None, None, :]) + beta[:, None, None, :]


class ResBlockFiLM(nn.Module):
    ch: int
    num_groups: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, tfeat: jnp.ndarray) -> jnp.ndarray:
        h = nn.GroupNorm(num_groups=self.num_groups)(x)
        h = nn.silu(h)
        h = nn.Conv(self.ch, kernel_size=(3, 3), padding="SAME")(h)

        # FiLM modulation
        h = nn.GroupNorm(num_groups=self.num_groups)(h)
        h = FiLM(self.ch)(h, tfeat)
        h = nn.silu(h)
        h = nn.Conv(self.ch, kernel_size=(3, 3), padding="SAME")(h)

        # Residual
        if x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, kernel_size=(1, 1), padding="SAME")(x)
        return x + h


class ConvFiLMTrunk(nn.Module):
    ch: int = 64
    depth: int = 3
    num_groups: int = 8
    temb_dim: int = 64
    tfeat_dim: int = 128

    @nn.compact
    def __call__(self, y: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        y: (B,H,W,d) analog bits as channels
        t: (B,) or (B,H,W)
        returns: (B,H,W,ch)
        """
        B = y.shape[0]
        t_vec = _reduce_time(t)
        tfeat = TimeConditioner(self.temb_dim, 4, self.tfeat_dim)(t_vec)

        h = nn.Conv(self.ch, kernel_size=(3, 3), padding="SAME")(y)
        for i in range(self.depth):
            h = ResBlockFiLM(self.ch, self.num_groups, name=f"res{i}")(h, tfeat)
        return h