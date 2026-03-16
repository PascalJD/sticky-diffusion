from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn

from .networks.adm_unet import ADMUNet2D
from .networks import unet, uvit


class UNet5DBackbone(nn.Module):
    """Apply a 2D UNet/UViT backbone on flattened 5D inputs.

    Input:
      [B, H, W, C, D]
    Output:
      [B, H, W, C, output_channels]
    """

    feature_dim: int = 128
    n_layers: int = 32
    n_dit_layers: int = 0
    dit_num_heads: int = 12
    dit_hidden_size: int = 768
    ch_mult: Sequence[int] = (1,)
    output_channels: int = 256
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, z, *, cond=None, timesteps=None, train: bool = False):
        del timesteps
        h = z.reshape(list(z.shape)[:-2] + [-1])

        if self.n_dit_layers > 0:
            h = uvit.UNet(
                d_channels=self.feature_dim,
                n_layers=self.n_layers,
                n_dit_layers=self.n_dit_layers,
                dit_num_heads=self.dit_num_heads,
                dit_hidden_size=self.dit_hidden_size,
                ch_mult=self.ch_mult,
                output_channels=self.output_channels * z.shape[-2],
                dropout_rate=self.dropout_rate,
            )(h, cond=cond, train=train)
        else:
            h = unet.UNet(
                d_channels=self.feature_dim,
                n_layers=self.n_layers,
                output_channels=self.output_channels * z.shape[-2],
                dropout_rate=self.dropout_rate,
            )(h, cond=cond, train=train)

        return h.reshape(list(z.shape)[:-1] + [self.output_channels])


class ADMUNet5DBackbone(nn.Module):
    """ADM-style UNet applied on flattened 5D image-token inputs.

    Input:
      [B, H, W, C, D]
    Output:
      [B, H, W, C, output_channels]
    """

    feature_dim: int = 128
    output_channels: int = 256

    num_res_blocks: int = 2
    attention_resolutions: Sequence[int] = (2, 4, 8)
    channel_mult: Sequence[int] = (1, 2, 2, 2)
    num_heads: int = 4
    num_head_channels: int = -1
    num_heads_upsample: int = -1

    dropout_rate: float = 0.0
    conv_resample: bool = True
    use_scale_shift_norm: bool = True
    resblock_updown: bool = False
    use_conv_skip: bool = False
    use_new_attention_order: bool = False

    @nn.compact
    def __call__(self, z, *, cond=None, timesteps=None, train: bool = False):
        h = z.reshape(list(z.shape)[:-2] + [-1])

        h = ADMUNet2D(
            in_channels=int(h.shape[-1]),
            model_channels=int(self.feature_dim),
            out_channels=int(self.output_channels * z.shape[-2]),
            num_res_blocks=int(self.num_res_blocks),
            attention_resolutions=tuple(int(v) for v in self.attention_resolutions),
            dropout_rate=float(self.dropout_rate),
            channel_mult=tuple(int(v) for v in self.channel_mult),
            conv_resample=bool(self.conv_resample),
            num_heads=int(self.num_heads),
            num_head_channels=int(self.num_head_channels),
            num_heads_upsample=int(self.num_heads_upsample),
            use_scale_shift_norm=bool(self.use_scale_shift_norm),
            resblock_updown=bool(self.resblock_updown),
            use_conv_skip=bool(self.use_conv_skip),
            use_new_attention_order=bool(self.use_new_attention_order),
        )(h, timesteps=timesteps, cond=cond, train=train)

        return h.reshape(list(z.shape)[:-1] + [self.output_channels])
