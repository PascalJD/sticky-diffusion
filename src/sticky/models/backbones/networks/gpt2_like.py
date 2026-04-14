from __future__ import annotations

import dataclasses

import flax.linen as nn
import jax.numpy as jnp


@dataclasses.dataclass(unsafe_hash=True)
class ModelArgs:
    dim: int = 384
    n_layers: int = 3
    n_heads: int = 12
    output_channels: int = 1024
    hidden_dim: int | None = None
    dropout_rate: float = 0.1
    use_attn_dropout: bool = True
    embed_input: bool = False
    n_embed_classes: int = 1024
    max_seq_len: int | None = None
    causal: bool = False


def _normal_init(stddev: float = 0.02):
    return nn.initializers.normal(stddev=stddev)


class GPT2MLP(nn.Module):
    dim: int
    hidden_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            name="fc_in",
        )(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(
            self.dim,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            name="fc_out",
        )(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        return x


class GPT2Block(nn.Module):
    args: ModelArgs

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        norm1 = nn.LayerNorm(epsilon=1e-5, name="ln_1")(x)
        attn_mask = None
        if bool(self.args.causal):
            attn_mask = nn.make_causal_mask(jnp.ones(norm1.shape[:2], dtype=jnp.bool_))
        attn = nn.MultiHeadDotProductAttention(
            num_heads=int(self.args.n_heads),
            qkv_features=int(self.args.dim),
            out_features=int(self.args.dim),
            dropout_rate=(
                float(self.args.dropout_rate) if bool(self.args.use_attn_dropout) else 0.0
            ),
            use_bias=True,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            name="attn",
        )(norm1, norm1, mask=attn_mask, deterministic=not train)
        attn = nn.Dropout(rate=float(self.args.dropout_rate))(attn, deterministic=not train)
        x = x + attn

        norm2 = nn.LayerNorm(epsilon=1e-5, name="ln_2")(x)
        hidden_dim = int(
            self.args.hidden_dim if self.args.hidden_dim is not None else 4 * self.args.dim
        )
        x = x + GPT2MLP(
            dim=int(self.args.dim),
            hidden_dim=hidden_dim,
            dropout_rate=float(self.args.dropout_rate),
            name="mlp",
        )(norm2, train=train)
        return x


class GPT2LikeTransformer(nn.Module):
    args: ModelArgs

    @nn.compact
    def __call__(self, z: jnp.ndarray, *, cond=None, train: bool = False) -> jnp.ndarray:
        if z.ndim != 2 and z.ndim != 3:
            raise ValueError(
                f"GPT2LikeTransformer expects rank-2 or rank-3 inputs, got shape={z.shape}."
            )

        if self.args.embed_input:
            x = nn.Embed(
                num_embeddings=int(self.args.n_embed_classes),
                features=int(self.args.dim),
                embedding_init=_normal_init(),
                name="token_embed",
            )(z.astype(jnp.int32))
        else:
            x = z.astype(jnp.float32)
            if int(x.shape[-1]) != int(self.args.dim):
                x = nn.Dense(
                    int(self.args.dim),
                    kernel_init=_normal_init(),
                    bias_init=nn.initializers.zeros,
                    name="input_proj",
                )(x)

        seq_len = int(x.shape[1])
        max_seq_len = seq_len if self.args.max_seq_len is None else int(self.args.max_seq_len)
        if seq_len > max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured max_seq_len={max_seq_len}."
            )

        pos_embed = self.param(
            "pos_embed",
            _normal_init(),
            (max_seq_len, int(self.args.dim)),
        )
        x = x + pos_embed[None, :seq_len, :]

        if cond is not None:
            cond_proj = nn.Dense(
                int(self.args.dim),
                kernel_init=_normal_init(),
                bias_init=nn.initializers.zeros,
                name="cond_proj",
            )(jnp.asarray(cond, dtype=jnp.float32))
            x = x + cond_proj[:, None, :]

        x = nn.Dropout(rate=float(self.args.dropout_rate))(x, deterministic=not train)

        for layer_idx in range(int(self.args.n_layers)):
            x = GPT2Block(self.args, name=f"block_{layer_idx}")(x, train=train)

        x = nn.LayerNorm(epsilon=1e-5, name="ln_f")(x)
        return nn.Dense(
            int(self.args.output_channels),
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            name="lm_head",
        )(x)
