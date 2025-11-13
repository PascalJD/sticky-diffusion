from __future__ import annotations
import jax.numpy as jnp
import flax.linen as nn
from sticky.models.common import ConvFiLMTrunk

class IntensityHead(nn.Module):
    """
    Per-pixel non-negative hazard
    Input:  y (B,H,W,d), t (B,) or (B,H,W)
    Output: lambda (B,H,W) via softplus
    """
    ch: int = 64
    depth: int = 3
    num_groups: int = 8
    temb_dim: int = 64
    tfeat_dim: int = 128
    min_lambda: float = 1e-6

    @nn.compact
    def __call__(self, y: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        trunk = ConvFiLMTrunk(
            self.ch, self.depth, self.num_groups, self.temb_dim, self.tfeat_dim
        )
        feats = trunk(y, t)
        lam = nn.Conv(1, kernel_size=(1, 1), padding="SAME")(feats)[..., 0]
        return nn.softplus(lam) + self.min_lambda