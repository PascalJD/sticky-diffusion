# src/sticky/models/sticky_head.py
from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.linen as nn

from sticky.models.common import ConvFiLMTrunk


class StickyHead(nn.Module):
    """
    Shared trunk with:
      - Allocator logits over L anchors
      - Scalar hazard lam(y, t) in [min_lambda, max_lambda]

    Input:  y (B,H,W,d), t (B,) or (B,H,W)
    Output: logits (B,H,W,L), lambda (B,H,W)
    """
    L: int
    ch: int = 64
    depth: int = 3
    num_groups: int = 8
    temb_dim: int = 64
    tfeat_dim: int = 128
    min_rho: float = 0.1  # rho should live near 1
    max_rho: float = 10
    use_confidence: bool = True

    @nn.compact
    def __call__(self, y: jnp.ndarray, t: jnp.ndarray):
        # Shared ConvFiLM trunk
        trunk = ConvFiLMTrunk(
            self.ch,
            self.depth,
            self.num_groups,
            self.temb_dim,
            self.tfeat_dim,
        )
        feats = trunk(y, t)  # (B,H,W,C)

        # Allocator head
        logits = nn.Conv(
            self.L,
            kernel_size=(1, 1),
            padding="SAME",
            name="alloc_head",
        )(feats)  # (B,H,W,L)

        if self.use_confidence:
            logits_sg = jax.lax.stop_gradient(logits)
            probs = nn.softmax(logits_sg, axis=-1)
            ent = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1, keepdims=True)
            topv, _ = jax.lax.top_k(logits_sg, k=2)
            margin = (topv[..., 0] - topv[..., 1])[..., None]
            haz_in = jnp.concatenate([feats, ent, margin], axis=-1)
        else:
            haz_in = feats

        haz_feats = nn.swish(
            nn.Conv(
                self.ch, kernel_size=(1, 1), padding="SAME", name="haz_proj"
            )(haz_in)
        )

        raw = nn.Conv(
            1,
            kernel_size=(1, 1),
            padding="SAME",
            name="haz_head",
        )(haz_feats)[..., 0]  # (B,H,W)
        
        raw = jnp.clip(raw, -10.0, 10.0)
        rho = jnp.exp(raw)  # (B,H,W) strictly > 0
        # Enforce E_spatial[rho]=1 per image (proxy for E[rho | t, off]=1)
        rho_mean = jnp.mean(rho, axis=(1, 2), keepdims=True)  # (B,1,1)
        rho_norm = rho / jax.lax.stop_gradient(rho_mean + 1e-12)

        return logits, rho_norm