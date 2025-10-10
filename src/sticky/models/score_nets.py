# src/sticky/models/score_nets.py
from __future__ import annotations
import jax.numpy as jnp
from flax import linen as nn
from ..utils.time_embed import timestep_embedding

class MLPScore1D(nn.Module):
    hidden: int = 128
    tdim: int = 64
    @nn.compact
    def __call__(
        self, x: jnp.ndarray, t: jnp.ndarray, train: bool
    ) -> jnp.ndarray:
        # x:(B,1), t:(B,)
        temb = timestep_embedding(t, self.tdim)
        h = jnp.concatenate([x, temb], axis=-1)
        h = nn.relu(nn.Dense(self.hidden)(h))
        h = nn.relu(nn.Dense(self.hidden)(h))
        out = nn.Dense(1)(h)
        return out

class SmallConvScore(nn.Module):
    """Simple conv-net for MNIST score on [B,28,28,1]."""
    channels: int = 64
    tdim: int = 64
    @nn.compact
    def __call__(
        self, x: jnp.ndarray, t: jnp.ndarray, train: bool
    ) -> jnp.ndarray:
        # x: (B,H,W,1); t:(B,)
        B, H, W, C = x.shape
        temb = timestep_embedding(t, self.tdim)
        temb_map = nn.Dense(H*W)(temb).reshape((B,H,W,1))
        h = jnp.concatenate([x, temb_map], axis=-1)
        h = nn.relu(nn.Conv(self.channels, (3,3), padding='SAME')(h))
        h = nn.relu(nn.Conv(self.channels, (3,3), padding='SAME')(h))
        h = nn.relu(nn.Conv(self.channels, (3,3), padding='SAME')(h))
        out = nn.Conv(1, (3,3), padding='SAME')(h)  # score has same shape as x
        return out