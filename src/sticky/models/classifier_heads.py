from __future__ import annotations
import jax.numpy as jnp
import flax.linen as nn
from sticky.models.common import ConvFiLMTrunk

class AllocatorHead(nn.Module):
    """
    Per-pixel classifier over L anchors.
    Input:  y (B,H,W,d), t (B,) or (B,H,W)
    Output: logits (B,H,W,L)
    """
    L: int
    ch: int = 64
    depth: int = 3
    num_groups: int = 8
    temb_dim: int = 64
    tfeat_dim: int = 128

    @nn.compact
    def __call__(self, y: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        trunk = ConvFiLMTrunk(
            self.ch, self.depth, self.num_groups, self.temb_dim, self.tfeat_dim
        )
        feats = trunk(y, t)
        logits = nn.Conv(self.L, kernel_size=(1, 1), padding="SAME")(feats)
        return logits