from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


def _num_groups(num_channels: int, preferred: int = 32) -> int:
    """Choose a valid GroupNorm group count for the given channel width."""
    for groups in (preferred, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


def _group_norm(x: jnp.ndarray, *, num_channels: int, name: str) -> jnp.ndarray:
    return nn.GroupNorm(
        num_groups=_num_groups(int(num_channels)),
        epsilon=1e-5,
        name=name,
    )(x)


class Upsample2D(nn.Module):
    channels: int
    use_conv: bool = True
    out_channels: int | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out_channels = int(self.out_channels or self.channels)
        b, h, w, c = x.shape
        if int(c) != int(self.channels):
            raise ValueError(
                f"Upsample2D expected {self.channels} channels, got {c}."
            )
        x = jax.image.resize(x, (b, h * 2, w * 2, c), method="nearest")
        if self.use_conv:
            x = nn.Conv(
                out_channels,
                kernel_size=(3, 3),
                padding=((1, 1), (1, 1)),
                name="conv",
            )(x)
        return x


class Downsample2D(nn.Module):
    channels: int
    use_conv: bool = True
    out_channels: int | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out_channels = int(self.out_channels or self.channels)
        if int(x.shape[-1]) != int(self.channels):
            raise ValueError(
                f"Downsample2D expected {self.channels} channels, got {x.shape[-1]}."
            )
        if self.use_conv:
            return nn.Conv(
                out_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding=((1, 1), (1, 1)),
                name="conv",
            )(x)
        if out_channels != int(self.channels):
            raise ValueError("Downsample2D without conv requires out_channels == channels.")
        return nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")


class ResBlock(nn.Module):
    out_channels: int
    emb_channels: int
    dropout_rate: float = 0.0
    use_conv_skip: bool = False
    use_scale_shift_norm: bool = False
    up: bool = False
    down: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        in_channels = int(x.shape[-1])
        out_channels = int(self.out_channels)
        if int(emb.shape[-1]) != int(self.emb_channels):
            raise ValueError(
                f"ResBlock expected emb dim {self.emb_channels}, got {emb.shape[-1]}."
            )

        h = _group_norm(x, num_channels=in_channels, name="in_norm")
        h = nn.silu(h)

        if self.up:
            h = Upsample2D(in_channels, use_conv=False, name="h_upd")(h)
            x_skip = Upsample2D(in_channels, use_conv=False, name="x_upd")(x)
        elif self.down:
            h = Downsample2D(in_channels, use_conv=False, name="h_upd")(h)
            x_skip = Downsample2D(in_channels, use_conv=False, name="x_upd")(x)
        else:
            x_skip = x

        h = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            name="in_conv",
        )(h)

        emb_out = nn.Dense(
            2 * out_channels if self.use_scale_shift_norm else out_channels,
            name="emb_proj",
        )(nn.silu(emb))
        emb_out = emb_out[:, None, None, :]

        if self.use_scale_shift_norm:
            scale, shift = jnp.split(emb_out, 2, axis=-1)
            h = _group_norm(h, num_channels=out_channels, name="out_norm")
            h = h * (1.0 + scale) + shift
            h = nn.silu(h)
        else:
            h = h + emb_out
            h = _group_norm(h, num_channels=out_channels, name="out_norm")
            h = nn.silu(h)

        h = nn.Dropout(rate=float(self.dropout_rate))(h, deterministic=not train)
        h = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            kernel_init=nn.initializers.zeros,
            name="out_conv",
        )(h)

        if in_channels == out_channels:
            skip = x_skip
        elif self.use_conv_skip:
            skip = nn.Conv(
                out_channels,
                kernel_size=(3, 3),
                padding=((1, 1), (1, 1)),
                name="skip_conv",
            )(x_skip)
        else:
            skip = nn.Conv(out_channels, kernel_size=(1, 1), name="skip_conv")(x_skip)

        return skip + h


class AttentionBlock(nn.Module):
    num_heads: int = 1
    num_head_channels: int = -1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, h, w, c = x.shape
        channels = int(c)
        if self.num_head_channels > 0:
            if channels % int(self.num_head_channels) != 0:
                raise ValueError(
                    f"channels={channels} not divisible by num_head_channels={self.num_head_channels}"
                )
            num_heads = channels // int(self.num_head_channels)
        else:
            num_heads = int(self.num_heads)

        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} not divisible by num_heads={num_heads}")

        head_dim = channels // num_heads
        n = h * w

        h_in = _group_norm(x, num_channels=channels, name="norm")
        h_in = h_in.reshape(b, n, channels)

        qkv = nn.Dense(3 * channels, name="qkv")(h_in)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(b, n, num_heads, head_dim)
        k = k.reshape(b, n, num_heads, head_dim)
        v = v.reshape(b, n, num_heads, head_dim)

        scale = 1.0 / jnp.sqrt(jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32)))
        w_attn = jnp.einsum("bthd,bshd->bhts", q * scale, k * scale)
        w_attn = nn.softmax(w_attn.astype(jnp.float32), axis=-1).astype(q.dtype)
        a = jnp.einsum("bhts,bshd->bthd", w_attn, v)
        a = a.reshape(b, n, channels)

        a = nn.Dense(
            channels,
            kernel_init=nn.initializers.zeros,
            name="proj_out",
        )(a)
        a = a.reshape(b, h, w, channels)
        return x + a


class ADMUNet2D(nn.Module):
    """ADM-style UNet (Flax, NHWC), close to guided-diffusion UNetModel."""

    in_channels: int
    model_channels: int
    out_channels: int
    num_res_blocks: int
    attention_resolutions: Sequence[int]
    dropout_rate: float = 0.0
    channel_mult: Sequence[int] = (1, 2, 4, 8)
    conv_resample: bool = True
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_conv_skip: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, cond: jnp.ndarray | None = None, train: bool = False) -> jnp.ndarray:
        if x.ndim != 4:
            raise ValueError(f"ADMUNet2D expects NHWC input, got shape {x.shape}.")
        if int(x.shape[-1]) != int(self.in_channels):
            raise ValueError(
                f"ADMUNet2D expected {self.in_channels} input channels, got {x.shape[-1]}."
            )

        num_heads_upsample = (
            int(self.num_heads) if int(self.num_heads_upsample) == -1 else int(self.num_heads_upsample)
        )
        time_embed_dim = int(self.model_channels) * 4

        if cond is None:
            cond_in = jnp.zeros((x.shape[0], int(self.model_channels)), dtype=x.dtype)
        else:
            cond_in = cond

        emb = nn.Dense(time_embed_dim, name="time_embed_0")(cond_in)
        emb = nn.silu(emb)
        emb = nn.Dense(time_embed_dim, name="time_embed_1")(emb)

        ch = int(self.channel_mult[0]) * int(self.model_channels)
        h = nn.Conv(
            ch,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            name="input_conv",
        )(x)

        hs: list[jnp.ndarray] = [h]
        input_block_chans: list[int] = [ch]
        ds = 1
        attn_ds = {int(v) for v in self.attention_resolutions}

        num_levels = len(self.channel_mult)
        for level, mult in enumerate(self.channel_mult):
            out_ch = int(mult) * int(self.model_channels)
            for block_idx in range(int(self.num_res_blocks)):
                h = ResBlock(
                    out_channels=out_ch,
                    emb_channels=time_embed_dim,
                    dropout_rate=float(self.dropout_rate),
                    use_scale_shift_norm=bool(self.use_scale_shift_norm),
                    use_conv_skip=bool(self.use_conv_skip),
                    name=f"input_res_{level}_{block_idx}",
                )(h, emb, train=train)
                ch = out_ch
                if ds in attn_ds:
                    h = AttentionBlock(
                        num_heads=int(self.num_heads),
                        num_head_channels=int(self.num_head_channels),
                        name=f"input_attn_{level}_{block_idx}",
                    )(h)
                hs.append(h)
                input_block_chans.append(ch)

            if level != num_levels - 1:
                if self.resblock_updown:
                    h = ResBlock(
                        out_channels=ch,
                        emb_channels=time_embed_dim,
                        dropout_rate=float(self.dropout_rate),
                        use_scale_shift_norm=bool(self.use_scale_shift_norm),
                        use_conv_skip=bool(self.use_conv_skip),
                        down=True,
                        name=f"input_down_res_{level}",
                    )(h, emb, train=train)
                else:
                    h = Downsample2D(
                        channels=ch,
                        use_conv=bool(self.conv_resample),
                        out_channels=ch,
                        name=f"input_downsample_{level}",
                    )(h)
                ds *= 2
                hs.append(h)
                input_block_chans.append(ch)

        h = ResBlock(
            out_channels=ch,
            emb_channels=time_embed_dim,
            dropout_rate=float(self.dropout_rate),
            use_scale_shift_norm=bool(self.use_scale_shift_norm),
            use_conv_skip=bool(self.use_conv_skip),
            name="middle_res_0",
        )(h, emb, train=train)
        h = AttentionBlock(
            num_heads=int(self.num_heads),
            num_head_channels=int(self.num_head_channels),
            name="middle_attn",
        )(h)
        h = ResBlock(
            out_channels=ch,
            emb_channels=time_embed_dim,
            dropout_rate=float(self.dropout_rate),
            use_scale_shift_norm=bool(self.use_scale_shift_norm),
            use_conv_skip=bool(self.use_conv_skip),
            name="middle_res_1",
        )(h, emb, train=train)

        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            out_ch = int(mult) * int(self.model_channels)
            for block_idx in range(int(self.num_res_blocks) + 1):
                skip = hs.pop()
                input_block_chans.pop()
                h = jnp.concatenate([h, skip], axis=-1)
                h = ResBlock(
                    out_channels=out_ch,
                    emb_channels=time_embed_dim,
                    dropout_rate=float(self.dropout_rate),
                    use_scale_shift_norm=bool(self.use_scale_shift_norm),
                    use_conv_skip=bool(self.use_conv_skip),
                    name=f"output_res_{level}_{block_idx}",
                )(h, emb, train=train)
                ch = out_ch

                if ds in attn_ds:
                    h = AttentionBlock(
                        num_heads=num_heads_upsample,
                        num_head_channels=int(self.num_head_channels),
                        name=f"output_attn_{level}_{block_idx}",
                    )(h)

                if level and block_idx == int(self.num_res_blocks):
                    if self.resblock_updown:
                        h = ResBlock(
                            out_channels=ch,
                            emb_channels=time_embed_dim,
                            dropout_rate=float(self.dropout_rate),
                            use_scale_shift_norm=bool(self.use_scale_shift_norm),
                            use_conv_skip=bool(self.use_conv_skip),
                            up=True,
                            name=f"output_up_res_{level}",
                        )(h, emb, train=train)
                    else:
                        h = Upsample2D(
                            channels=ch,
                            use_conv=bool(self.conv_resample),
                            out_channels=ch,
                            name=f"output_upsample_{level}",
                        )(h)
                    ds //= 2

        h = _group_norm(h, num_channels=ch, name="out_norm")
        h = nn.silu(h)
        h = nn.Conv(
            int(self.out_channels),
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            kernel_init=nn.initializers.zeros,
            name="out_conv",
        )(h)
        return h
