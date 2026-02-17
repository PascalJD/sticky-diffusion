# src/sticky/models/factory.py
from __future__ import annotations

from omegaconf import DictConfig

def build_model(
    cfg: DictConfig, *, data_shape: tuple[int, ...], vocab_size: int
):
    name = cfg.model.name

    if name == "md4":
        from sticky.models.md4.md4_model import MD4

        return MD4(
            data_shape=data_shape,
            cont_time=bool(cfg.model.cont_time),
            timesteps=int(cfg.model.timesteps),
            feature_dim=int(cfg.model.feature_dim),
            num_heads=int(cfg.model.num_heads),
            antithetic_time_sampling=bool(cfg.model.antithetic_time_sampling),
            n_layers=int(cfg.model.n_layers),
            n_dit_layers=int(cfg.model.n_dit_layers),
            dit_num_heads=int(cfg.model.dit_num_heads),
            dit_hidden_size=int(cfg.model.dit_hidden_size),
            ch_mult=tuple(cfg.model.ch_mult),
            vocab_size=vocab_size,
            noise_schedule_type=str(cfg.model.noise_schedule_type),
            dropout_rate=float(cfg.model.dropout_rate),
            use_attn_dropout=bool(cfg.model.use_attn_dropout),
            mlp_type=str(cfg.model.mlp_type),
            depth_scaled_init=bool(cfg.model.depth_scaled_init),
            cond_type=str(cfg.model.cond_type),
            outside_embed=bool(cfg.model.outside_embed),
            time_features=str(cfg.model.time_features),
            classes=int(cfg.model.classes),
            sampler=str(cfg.model.sampler),
            sampling_grid=str(cfg.model.sampling_grid),
            topp=float(cfg.model.topp),
            model_sharding=bool(cfg.model.model_sharding),
        )

    if name == "sjd":
        from sticky.models.sjd.sjd_model import SJD

        return SJD(
            vocab_size=vocab_size,
            anchor_dim=int(cfg.model.anchor_dim),
            learnable_anchors=bool(cfg.model.get("learnable_anchors", True)),
            anchors_init_std=float(cfg.model.get("anchors_init_std", 1.0)),
            feature_dim=int(cfg.model.feature_dim),
            n_layers=int(cfg.model.n_layers),
            n_dit_layers=int(cfg.model.n_dit_layers),
            dit_num_heads=int(cfg.model.dit_num_heads),
            dit_hidden_size=int(cfg.model.dit_hidden_size),
            ch_mult=tuple(cfg.model.ch_mult),
            num_heads=int(cfg.model.num_heads),
            dropout_rate=float(cfg.model.dropout_rate),
            use_attn_dropout=bool(cfg.model.get("use_attn_dropout", True)),
            mlp_type=str(cfg.model.get("mlp_type", "swiglu")),
            depth_scaled_init=bool(cfg.model.get("depth_scaled_init", False)),
            cond_type=str(cfg.model.get("cond_type", "adaln")),
            model_sharding=bool(cfg.model.get("model_sharding", False)),
            sequence_backbone=str(cfg.model.get("sequence_backbone", "auto")),
            image_backbone=str(cfg.model.get("image_backbone", "auto")),
        )

    raise ValueError(f"Unknown model.name={name}")
