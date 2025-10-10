# src/sticky/models/classifier_heads.py
from __future__ import annotations
import jax.numpy as jnp
from flax import linen as nn
from ..utils.time_embed import timestep_embedding


class CoordClassifier(nn.Module):
    d: int
    L: int
    channels: int = 64
    tdim: int = 64
    @nn.compact
    def __call__(
        self, x: jnp.ndarray, t: jnp.ndarray, train: bool
    ) -> jnp.ndarray:
        # x:(B,H,W,1) or (B,d).
        B, D = x.shape
        temb = timestep_embedding(t, self.tdim)
        temb = nn.Dense(D)(temb)
        h = jnp.concatenate([x, temb], axis=-1)
        h = nn.relu(nn.Dense(2*self.channels)(h))
        h = nn.relu(nn.Dense(2*self.channels)(h))
        logits = nn.Dense(self.d * self.L)(h).reshape((B, self.d, self.L))
        return logits