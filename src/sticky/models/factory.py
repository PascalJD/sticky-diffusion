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
            sequence_backbone=str(cfg.model.get("sequence_backbone", "auto")),
            image_backbone=str(cfg.model.get("image_backbone", "auto")),
            adm_num_res_blocks=int(cfg.model.get("adm_num_res_blocks", 2)),
            adm_attention_resolutions=tuple(cfg.model.get("adm_attention_resolutions", (2, 4, 8))),
            adm_num_heads=int(cfg.model.get("adm_num_heads", 4)),
            adm_num_head_channels=int(cfg.model.get("adm_num_head_channels", -1)),
            adm_num_heads_upsample=int(cfg.model.get("adm_num_heads_upsample", -1)),
            adm_conv_resample=bool(cfg.model.get("adm_conv_resample", True)),
            adm_use_scale_shift_norm=bool(cfg.model.get("adm_use_scale_shift_norm", True)),
            adm_resblock_updown=bool(cfg.model.get("adm_resblock_updown", False)),
            adm_use_conv_skip=bool(cfg.model.get("adm_use_conv_skip", False)),
            adm_use_new_attention_order=bool(cfg.model.get("adm_use_new_attention_order", False)),
            time_features=str(cfg.model.time_features),
            classes=int(cfg.model.classes),
            sampler=str(cfg.model.sampler),
            sampling_grid=str(cfg.model.sampling_grid),
            topp=float(cfg.model.topp),
            model_sharding=bool(cfg.model.model_sharding),
        )

    if name == "sjd":
        from sticky.models.sjd.anchors import (
            anchor_learnable_from_mapping,
            anchor_table_config_from_mapping,
        )
        from sticky.models.sjd.sjd_model import SJD

        return SJD(
            vocab_size=vocab_size,
            anchor_config=anchor_table_config_from_mapping(
                cfg.model,
                vocab_size=vocab_size,
            ),
            learnable_anchors=anchor_learnable_from_mapping(cfg.model, default=True),
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
            adm_num_res_blocks=int(cfg.model.get("adm_num_res_blocks", 2)),
            adm_attention_resolutions=tuple(cfg.model.get("adm_attention_resolutions", (2, 4, 8))),
            adm_num_heads=int(cfg.model.get("adm_num_heads", 4)),
            adm_num_head_channels=int(cfg.model.get("adm_num_head_channels", -1)),
            adm_num_heads_upsample=int(cfg.model.get("adm_num_heads_upsample", -1)),
            adm_conv_resample=bool(cfg.model.get("adm_conv_resample", True)),
            adm_use_scale_shift_norm=bool(cfg.model.get("adm_use_scale_shift_norm", True)),
            adm_resblock_updown=bool(cfg.model.get("adm_resblock_updown", False)),
            adm_use_conv_skip=bool(cfg.model.get("adm_use_conv_skip", False)),
            adm_use_new_attention_order=bool(cfg.model.get("adm_use_new_attention_order", False)),
        )

    if name == "cadd":
        from sticky.models.cadd.cadd_model import CADD

        latent_cfg = cfg.model.get("cadd_latent", {})

        return CADD(
            data_shape=data_shape,
            vocab_size=vocab_size,
            cont_time=bool(cfg.model.get("cont_time", True)),
            timesteps=int(cfg.model.get("timesteps", 512)),
            antithetic_time_sampling=bool(cfg.model.get("antithetic_time_sampling", False)),
            discrete_schedule_type=str(cfg.model.get("discrete_schedule_type", "linear")),
            continuous_schedule_type=str(
                latent_cfg.get(
                    "continuous_schedule_type",
                    cfg.model.get("continuous_schedule_type", "linear"),
                )
            ),
            continuous_latent_type=str(
                latent_cfg.get(
                    "type",
                    cfg.model.get("continuous_latent_type", "gaussian"),
                )
            ),
            schedule_eps=float(cfg.model.get("schedule_eps", 1e-4)),
            feature_dim=int(cfg.model.get("feature_dim", 128)),
            num_heads=int(cfg.model.get("num_heads", 12)),
            n_layers=int(cfg.model.get("n_layers", 32)),
            n_dit_layers=int(cfg.model.get("n_dit_layers", 0)),
            dit_num_heads=int(cfg.model.get("dit_num_heads", 12)),
            dit_hidden_size=int(cfg.model.get("dit_hidden_size", 768)),
            ch_mult=tuple(cfg.model.get("ch_mult", (1, 2, 2, 2))),
            dropout_rate=float(cfg.model.get("dropout_rate", 0.0)),
            use_attn_dropout=bool(cfg.model.get("use_attn_dropout", True)),
            mlp_type=str(cfg.model.get("mlp_type", "swiglu")),
            depth_scaled_init=bool(cfg.model.get("depth_scaled_init", False)),
            cond_type=str(cfg.model.get("cond_type", "adaln")),
            model_sharding=bool(cfg.model.get("model_sharding", False)),
            sequence_backbone=str(cfg.model.get("sequence_backbone", "auto")),
            image_backbone=str(cfg.model.get("image_backbone", "auto")),
            adm_num_res_blocks=int(cfg.model.get("adm_num_res_blocks", 2)),
            adm_attention_resolutions=tuple(cfg.model.get("adm_attention_resolutions", (2, 4, 8))),
            adm_num_heads=int(cfg.model.get("adm_num_heads", 4)),
            adm_num_head_channels=int(cfg.model.get("adm_num_head_channels", -1)),
            adm_num_heads_upsample=int(cfg.model.get("adm_num_heads_upsample", -1)),
            adm_conv_resample=bool(cfg.model.get("adm_conv_resample", True)),
            adm_use_scale_shift_norm=bool(cfg.model.get("adm_use_scale_shift_norm", True)),
            adm_resblock_updown=bool(cfg.model.get("adm_resblock_updown", False)),
            adm_use_conv_skip=bool(cfg.model.get("adm_use_conv_skip", False)),
            adm_use_new_attention_order=bool(cfg.model.get("adm_use_new_attention_order", False)),
            time_features=str(cfg.model.get("time_features", "t")),
            classes=int(cfg.model.get("classes", -1)),
            sampling_grid=str(cfg.model.get("sampling_grid", "cosine")),
            temperature_schedule=str(cfg.model.get("temperature_schedule", "cosine_decay")),
            tau_max=float(cfg.model.get("tau_max", 2.5)),
            logit_temperature=float(cfg.model.get("logit_temperature", 1.0)),
            z0_estimator=str(cfg.model.get("z0_estimator", "hard")),
            K=int(cfg.model.get("K", 1)),
            force_decode_at_end=bool(cfg.model.get("force_decode_at_end", True)),

            corrector_enabled=bool(cfg.model.get("corrector_enabled", False)),
            corrector_steps=int(cfg.model.get("corrector_steps", 1)),
            corrector_remask_frac=float(cfg.model.get("corrector_remask_frac", 0.0)),
            corrector_metric=str(cfg.model.get("corrector_metric", "entropy")),
            corrector_sample_mode=str(cfg.model.get("corrector_sample_mode", "sample")),
        )

    raise ValueError(f"Unknown model.name={name}")
