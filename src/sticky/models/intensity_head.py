from __future__ import annotations
import jax.numpy as jnp
import flax.linen as nn
from sticky.models.common import ConvFiLMTrunk

class IntensityHead(nn.Module):
    ch: int = 64
    depth: int = 3
    num_groups: int = 8
    temb_dim: int = 64
    tfeat_dim: int = 128
    min_lambda: float = 1e-6
    max_lambda: float = 1e3 

    @nn.compact
    def __call__(self, y: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        trunk = ConvFiLMTrunk(
            self.ch, self.depth, self.num_groups, self.temb_dim, self.tfeat_dim
        )
        feats = trunk(y, t)
        raw = nn.Conv(1, kernel_size=(1, 1), padding="SAME")(feats)[..., 0]
        # bound between [min_lambda, max_lambda]
        lam = nn.sigmoid(raw) * (self.max_lambda - self.min_lambda) + self.min_lambda
        return lam