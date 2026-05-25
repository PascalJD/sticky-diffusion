# src/sticky/models/factories/bitdiff.py
"""Builder for the BitDiffusion model family."""
from __future__ import annotations

from omegaconf import DictConfig

from sticky.models.factories._helpers import (
    adm_backbone_kwargs,
    sampler_method,
    sampling_grid,
)


def build_model(cfg: DictConfig, *, data_shape: tuple[int, ...], vocab_size: int):
    del vocab_size
    from sticky.models.baselines.bitdiff.bitdiff_model import BitDiffusion

    sampler_cfg = cfg.get("sampler", {})
    return BitDiffusion(
        data_shape=data_shape,
        cont_time=bool(cfg.model.get("cont_time", True)),
        timesteps=int(cfg.model.get("timesteps", 256)),
        num_bits=int(cfg.model.get("num_bits", 8)),
        encoding=str(cfg.model.get("encoding", "uint8")),
        predict_target=str(cfg.model.get("predict_target", "x0")),
        loss_type=str(cfg.model.get("loss_type", "mse")),
        self_conditioning=bool(cfg.model.get("self_conditioning", True)),
        self_conditioning_rate=float(cfg.model.get("self_conditioning_rate", 0.5)),
        analog_bit_scale=float(cfg.model.get("analog_bit_scale", 1.0)),
        clip_x0=bool(cfg.model.get("clip_x0", True)),
        signal_schedule_type=str(cfg.model.get("signal_schedule_type", "linear")),
        schedule_eps=float(cfg.model.get("schedule_eps", 0.0)),
        feature_dim=int(cfg.model.get("feature_dim", 96)),
        num_heads=int(cfg.model.get("num_heads", 12)),
        n_layers=int(cfg.model.get("n_layers", 32)),
        n_dit_layers=int(cfg.model.get("n_dit_layers", 0)),
        dit_num_heads=int(cfg.model.get("dit_num_heads", 12)),
        dit_hidden_size=int(cfg.model.get("dit_hidden_size", 768)),
        ch_mult=tuple(cfg.model.get("ch_mult", (3, 4, 4))),
        dropout_rate=float(cfg.model.get("dropout_rate", 0.1)),
        use_attn_dropout=bool(cfg.model.get("use_attn_dropout", True)),
        mlp_type=str(cfg.model.get("mlp_type", "swiglu")),
        depth_scaled_init=bool(cfg.model.get("depth_scaled_init", False)),
        cond_type=str(cfg.model.get("cond_type", "adaln")),
        model_sharding=bool(cfg.model.get("model_sharding", False)),
        sequence_backbone=str(cfg.model.get("sequence_backbone", "auto")),
        **adm_backbone_kwargs(
            cfg.model,
            image_backbone_default="adm_unet5d",
            adm_num_res_blocks_default=4,
            adm_attention_resolutions_default=(2, 4),
            adm_num_heads_default=4,
            adm_num_head_channels_default=64,
        ),
        time_features=str(cfg.model.get("time_features", "t")),
        classes=int(cfg.model.get("classes", -1)),
        sampler=sampler_method(cfg, default="ddim"),
        sampling_grid=sampling_grid(cfg, default="uniform"),
        time_difference=float(
            sampler_cfg.get(
                "time_difference",
                cfg.model.get("time_difference", 0.0),
            )
        ),
        stochasticity=float(
            sampler_cfg.get("stochasticity", cfg.model.get("stochasticity", 0.0))
        ),
    )
