# src/sticky/utils/time_embed.py
import jax.numpy as jnp

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