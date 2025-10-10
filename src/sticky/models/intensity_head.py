# src/sticky/models/intensity_head.py
from __future__ import annotations
import jax.numpy as jnp
from flax import linen as nn
from ..utils.time_embed import timestep_embedding

class IntensityNet(nn.Module):
    hidden: int = 128
    tdim: int = 64
    @nn.compact
    def __call__(
        self, x: jnp.ndarray, t: jnp.ndarray, train: bool
    ) -> jnp.ndarray:
        # x:(B,d) or (B,d_image). Return (B,1) nonnegative.
        temb = timestep_embedding(t, self.tdim)
        h = jnp.concatenate([x, temb], axis=-1)
        h = nn.relu(nn.Dense(self.hidden)(h))
        h = nn.relu(nn.Dense(self.hidden)(h))
        out = nn.Dense(1)(h)
        return nn.softplus(out) + 1e-6