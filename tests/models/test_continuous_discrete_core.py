from __future__ import annotations

import jax
import jax.numpy as jnp

from sticky.models.architectures import build_image_token_backbone
from sticky.models.continuous_discrete_core import (
    analog_bits_to_uint8,
    carry_over_revealed,
    uint8_to_analog_bits,
    uint8_to_bit_planes,
    bit_planes_to_uint8,
)


def test_uint8_bit_roundtrip_supports_binary_and_gray_code():
    x = jnp.asarray(
        [[0, 1, 2, 7], [31, 127, 128, 255]],
        dtype=jnp.int32,
    )

    bits = uint8_to_bit_planes(x)
    restored = bit_planes_to_uint8(bits)
    assert jnp.array_equal(restored, x)

    gray_bits = uint8_to_bit_planes(x, gray_code=True)
    gray_restored = bit_planes_to_uint8(gray_bits, gray_code=True)
    assert jnp.array_equal(gray_restored, x)


def test_analog_bit_threshold_roundtrip():
    x = jnp.asarray([[0, 5, 170, 255]], dtype=jnp.int32)
    analog = uint8_to_analog_bits(x)
    analog = analog + 0.2 * jnp.sign(analog)
    restored = analog_bits_to_uint8(analog, threshold=0.0)

    assert jnp.array_equal(restored, x)


def test_carry_over_revealed_broadcasts_over_feature_dimension():
    current = jnp.asarray([[[[1.0, 1.0], [2.0, 2.0]]]], dtype=jnp.float32)
    proposal = jnp.asarray([[[[9.0, 9.0], [8.0, 8.0]]]], dtype=jnp.float32)
    revealed = jnp.asarray([[[True, False]]])

    merged = carry_over_revealed(current, proposal, revealed_mask=revealed)

    expected = jnp.asarray([[[[1.0, 1.0], [8.0, 8.0]]]], dtype=jnp.float32)
    assert jnp.array_equal(merged, expected)


def test_shared_continuous_5d_adm_backbone_smoke():
    backbone = build_image_token_backbone(
        name="adm_unet5d",
        feature_dim=32,
        n_layers=32,
        n_dit_layers=0,
        dit_num_heads=4,
        dit_hidden_size=128,
        ch_mult=(1, 1),
        output_channels=7,
        dropout_rate=0.0,
        adm_num_res_blocks=1,
        adm_attention_resolutions=(),
        adm_num_heads=1,
        adm_num_head_channels=-1,
        adm_num_heads_upsample=-1,
        adm_conv_resample=True,
        adm_use_scale_shift_norm=True,
        adm_resblock_updown=False,
        adm_use_conv_skip=False,
        adm_use_new_attention_order=False,
    )

    x = jnp.linspace(
        -1.0,
        1.0,
        num=2 * 8 * 8 * 3 * 4,
        dtype=jnp.float32,
    ).reshape(2, 8, 8, 3, 4)
    timesteps = jnp.asarray([125.0, 875.0], dtype=jnp.float32)

    variables = backbone.init(
        {"params": jax.random.PRNGKey(0)},
        x,
        cond=None,
        timesteps=timesteps,
        train=False,
    )
    y = backbone.apply(
        variables,
        x,
        cond=None,
        timesteps=timesteps,
        train=False,
    )

    assert y.shape == (2, 8, 8, 3, 7)
    assert jnp.isfinite(y).all()
