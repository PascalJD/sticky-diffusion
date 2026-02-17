from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn

from sticky.models.md4.networks import unet, uvit


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
    def __call__(self, z, *, cond=None, train: bool = False):
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
