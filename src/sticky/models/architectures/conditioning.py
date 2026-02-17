from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def get_timestep_embedding(timesteps, embedding_dim: int, dtype="float"):
    """Build sinusoidal timestep embeddings."""
    assert embedding_dim > 2
    half_dim = embedding_dim // 2
    emb = jnp.log(10_000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype="float32") * -emb)
    emb = timesteps.astype("float32")[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:
        emb = jax.lax.pad(emb, jnp.array(0, dtype), ((0, 0, 0), (0, 1, 0)))
    return emb


class CondEmbedding(nn.Module):
    """Joint time/conditioning embedding used by classifier backbones."""

    embedding_dim: int = 256

    @nn.compact
    def __call__(self, t, cond=None):
        n_embd = int(self.embedding_dim)
        temb = get_timestep_embedding(t, n_embd)
        if cond is None:
            cond_in = temb
        else:
            cond_in = jnp.concatenate([temb, cond], axis=-1)
        h = nn.swish(nn.Dense(features=n_embd * 4, name="dense0")(cond_in))
        h = nn.Dense(n_embd)(h)
        return h
