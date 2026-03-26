# src/sticky/models/factory.py
from __future__ import annotations

from omegaconf import DictConfig


def build_model(
    cfg: DictConfig, *, data_shape: tuple[int, ...], vocab_size: int
):
    name = cfg.model.name
    sampler_cfg = cfg.get("sampler", {})

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
            sampler=str(sampler_cfg.get("method", sampler_cfg.get("sampler", cfg.model.get("sampler", "ancestral")))),
            sampling_grid=str(sampler_cfg.get("sampling_grid", cfg.model.get("sampling_grid", "cosine"))),
            topp=float(sampler_cfg.get("topp", cfg.model.get("topp", 0.98))),
            categorical_sampling_policy=str(
                sampler_cfg.get(
                    "categorical_sampling_policy",
                    cfg.model.get("categorical_sampling_policy", "legacy_low"),
                )
            ),
            model_sharding=bool(cfg.model.model_sharding),
        )

    if name == "mdlm":
        from sticky.models.mdlm.mdlm_model import MDLM

        return MDLM(
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
            sequence_mlp_hidden_dim=cfg.model.get("sequence_mlp_hidden_dim", None),
            sequence_max_length=cfg.model.get("sequence_max_length", None),
            sequence_causal=bool(cfg.model.get("sequence_causal", False)),
            image_backbone=str(cfg.model.get("image_backbone", "adm_unet5d")),
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
            sampler=str(sampler_cfg.get("method", sampler_cfg.get("sampler", cfg.model.get("sampler", "ancestral")))),
            sampling_grid=str(sampler_cfg.get("sampling_grid", cfg.model.get("sampling_grid", "cosine"))),
            topp=float(sampler_cfg.get("topp", cfg.model.get("topp", 0.98))),
            oracle_noise_type=str(
                sampler_cfg.get(
                    "oracle_noise_type",
                    cfg.model.get("oracle_noise_type", "none"),
                )
            ),
            oracle_noise_scale=float(
                sampler_cfg.get(
                    "oracle_noise_scale",
                    cfg.model.get("oracle_noise_scale", 0.0),
                )
            ),
            categorical_sampling_policy=str(
                sampler_cfg.get(
                    "categorical_sampling_policy",
                    cfg.model.get("categorical_sampling_policy", "legacy_low"),
                )
            ),
            revealed_token_sample_mode=str(
                sampler_cfg.get(
                    "revealed_token_sample_mode",
                    cfg.model.get("revealed_token_sample_mode", "sample"),
                )
            ),
            cache_predictions=bool(
                sampler_cfg.get(
                    "cache_predictions",
                    cfg.model.get("cache_predictions", False),
                )
            ),
            model_sharding=bool(cfg.model.model_sharding),
        )

    if name == "d3pm":
        from sticky.models.d3pm.d3pm_model import D3PM

        return D3PM(
            data_shape=data_shape,
            timesteps=int(cfg.model.get("timesteps", 256)),
            transition_type=str(cfg.model.get("transition_type", "gaussian")),
            transition_beta_schedule=str(
                cfg.model.get("transition_beta_schedule", "linear")
            ),
            beta_start=float(cfg.model.get("beta_start", 1e-4)),
            beta_end=float(cfg.model.get("beta_end", 2e-2)),
            cosine_s=float(cfg.model.get("cosine_s", 0.008)),
            max_beta=float(cfg.model.get("max_beta", 0.999)),
            auxiliary_loss_weight=float(cfg.model.get("auxiliary_loss_weight", 1e-3)),
            absorbing_state=int(cfg.model.get("absorbing_state", 128)),
            feature_dim=int(cfg.model.get("feature_dim", 96)),
            num_heads=int(cfg.model.get("num_heads", 12)),
            antithetic_time_sampling=bool(cfg.model.get("antithetic_time_sampling", True)),
            n_layers=int(cfg.model.get("n_layers", 32)),
            n_dit_layers=int(cfg.model.get("n_dit_layers", 0)),
            dit_num_heads=int(cfg.model.get("dit_num_heads", 12)),
            dit_hidden_size=int(cfg.model.get("dit_hidden_size", 768)),
            ch_mult=tuple(cfg.model.get("ch_mult", (3, 4, 4))),
            vocab_size=vocab_size,
            dropout_rate=float(cfg.model.get("dropout_rate", 0.1)),
            use_attn_dropout=bool(cfg.model.get("use_attn_dropout", True)),
            mlp_type=str(cfg.model.get("mlp_type", "swiglu")),
            depth_scaled_init=bool(cfg.model.get("depth_scaled_init", True)),
            cond_type=str(cfg.model.get("cond_type", "adaln_zero")),
            outside_embed=bool(cfg.model.get("outside_embed", False)),
            sequence_backbone=str(cfg.model.get("sequence_backbone", "auto")),
            image_backbone=str(cfg.model.get("image_backbone", "adm_unet5d")),
            adm_num_res_blocks=int(cfg.model.get("adm_num_res_blocks", 4)),
            adm_attention_resolutions=tuple(cfg.model.get("adm_attention_resolutions", (2, 4))),
            adm_num_heads=int(cfg.model.get("adm_num_heads", 4)),
            adm_num_head_channels=int(cfg.model.get("adm_num_head_channels", 64)),
            adm_num_heads_upsample=int(cfg.model.get("adm_num_heads_upsample", -1)),
            adm_conv_resample=bool(cfg.model.get("adm_conv_resample", True)),
            adm_use_scale_shift_norm=bool(cfg.model.get("adm_use_scale_shift_norm", True)),
            adm_resblock_updown=bool(cfg.model.get("adm_resblock_updown", False)),
            adm_use_conv_skip=bool(cfg.model.get("adm_use_conv_skip", False)),
            adm_use_new_attention_order=bool(cfg.model.get("adm_use_new_attention_order", False)),
            time_features=str(cfg.model.get("time_features", "t")),
            classes=int(cfg.model.get("classes", -1)),
            sampler=str(
                sampler_cfg.get(
                    "method",
                    sampler_cfg.get("sampler", cfg.model.get("sampler", "ancestral")),
                )
            ),
            sampling_grid=str(
                sampler_cfg.get("sampling_grid", cfg.model.get("sampling_grid", "uniform"))
            ),
            categorical_sampling_policy=str(
                sampler_cfg.get(
                    "categorical_sampling_policy",
                    cfg.model.get("categorical_sampling_policy", "legacy_low"),
                )
            ),
            model_sharding=bool(cfg.model.get("model_sharding", False)),
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
            sampling_grid=str(sampler_cfg.get("sampling_grid", cfg.model.get("sampling_grid", "cosine"))),
            temperature_schedule=str(
                sampler_cfg.get(
                    "temperature_schedule",
                    cfg.model.get("temperature_schedule", "cosine_decay"),
                )
            ),
            tau_max=float(sampler_cfg.get("tau_max", cfg.model.get("tau_max", 2.5))),
            logit_temperature=float(
                sampler_cfg.get("logit_temperature", cfg.model.get("logit_temperature", 1.0))
            ),
            z0_estimator=str(sampler_cfg.get("z0_estimator", cfg.model.get("z0_estimator", "hard"))),
            K=int(sampler_cfg.get("K", cfg.model.get("K", 1))),
            force_decode_at_end=bool(
                sampler_cfg.get("force_decode_at_end", cfg.model.get("force_decode_at_end", True))
            ),
            categorical_sampling_policy=str(
                sampler_cfg.get(
                    "categorical_sampling_policy",
                    cfg.model.get("categorical_sampling_policy", "legacy_low"),
                )
            ),

            corrector_enabled=bool(
                sampler_cfg.get("corrector_enabled", cfg.model.get("corrector_enabled", False))
            ),
            corrector_steps=int(
                sampler_cfg.get("corrector_steps", cfg.model.get("corrector_steps", 1))
            ),
            corrector_remask_frac=float(
                sampler_cfg.get(
                    "corrector_remask_frac",
                    cfg.model.get("corrector_remask_frac", 0.0),
                )
            ),
            corrector_metric=str(
                sampler_cfg.get("corrector_metric", cfg.model.get("corrector_metric", "entropy"))
            ),
            corrector_sample_mode=str(
                sampler_cfg.get(
                    "corrector_sample_mode",
                    cfg.model.get("corrector_sample_mode", "sample"),
                )
            ),
        )

    if name == "bitdiff":
        from sticky.models.bitdiff.bitdiff_model import BitDiffusion

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
            image_backbone=str(cfg.model.get("image_backbone", "adm_unet5d")),
            adm_num_res_blocks=int(cfg.model.get("adm_num_res_blocks", 4)),
            adm_attention_resolutions=tuple(cfg.model.get("adm_attention_resolutions", (2, 4))),
            adm_num_heads=int(cfg.model.get("adm_num_heads", 4)),
            adm_num_head_channels=int(cfg.model.get("adm_num_head_channels", 64)),
            adm_num_heads_upsample=int(cfg.model.get("adm_num_heads_upsample", -1)),
            adm_conv_resample=bool(cfg.model.get("adm_conv_resample", True)),
            adm_use_scale_shift_norm=bool(cfg.model.get("adm_use_scale_shift_norm", True)),
            adm_resblock_updown=bool(cfg.model.get("adm_resblock_updown", False)),
            adm_use_conv_skip=bool(cfg.model.get("adm_use_conv_skip", False)),
            adm_use_new_attention_order=bool(cfg.model.get("adm_use_new_attention_order", False)),
            time_features=str(cfg.model.get("time_features", "t")),
            classes=int(cfg.model.get("classes", -1)),
            sampler=str(sampler_cfg.get("method", cfg.model.get("sampler", "ddim"))),
            sampling_grid=str(sampler_cfg.get("sampling_grid", cfg.model.get("sampling_grid", "uniform"))),
            time_difference=float(
                sampler_cfg.get("time_difference", cfg.model.get("time_difference", 0.0))
            ),
            stochasticity=float(
                sampler_cfg.get("stochasticity", cfg.model.get("stochasticity", 0.0))
            ),
        )

    if name == "candi":
        from sticky.models.candi.candi_model import CANDI

        return CANDI(
            data_shape=data_shape,
            vocab_size=vocab_size,
            cont_time=bool(cfg.model.get("cont_time", True)),
            timesteps=int(cfg.model.get("timesteps", 256)),
            representation=str(cfg.model.get("representation", "embed")),
            experimental=bool(cfg.model.get("experimental", True)),
            alpha_schedule_type=str(cfg.model.get("alpha_schedule_type", "linear")),
            schedule_eps=float(cfg.model.get("schedule_eps", 0.0)),
            pure_continuous=bool(cfg.model.get("pure_continuous", False)),
            use_percentile_scheduling=bool(
                cfg.model.get("use_percentile_scheduling", True)
            ),
            min_percentile=float(cfg.model.get("min_percentile", 0.01)),
            max_percentile=float(cfg.model.get("max_percentile", 0.45)),
            sigma_min=float(cfg.model.get("sigma_min", 0.2)),
            sigma_max=float(cfg.model.get("sigma_max", 4.0)),
            ode_step_scale=float(cfg.model.get("ode_step_scale", 1.0)),
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
            image_backbone=str(cfg.model.get("image_backbone", "adm_unet5d")),
            adm_num_res_blocks=int(cfg.model.get("adm_num_res_blocks", 4)),
            adm_attention_resolutions=tuple(cfg.model.get("adm_attention_resolutions", (2, 4))),
            adm_num_heads=int(cfg.model.get("adm_num_heads", 4)),
            adm_num_head_channels=int(cfg.model.get("adm_num_head_channels", 64)),
            adm_num_heads_upsample=int(cfg.model.get("adm_num_heads_upsample", -1)),
            adm_conv_resample=bool(cfg.model.get("adm_conv_resample", True)),
            adm_use_scale_shift_norm=bool(cfg.model.get("adm_use_scale_shift_norm", True)),
            adm_resblock_updown=bool(cfg.model.get("adm_resblock_updown", False)),
            adm_use_conv_skip=bool(cfg.model.get("adm_use_conv_skip", False)),
            adm_use_new_attention_order=bool(cfg.model.get("adm_use_new_attention_order", False)),
            time_features=str(cfg.model.get("time_features", "t")),
            classes=int(cfg.model.get("classes", -1)),
            sampler=str(sampler_cfg.get("method", cfg.model.get("sampler", "hybrid_cache"))),
            sampling_grid=str(
                sampler_cfg.get("sampling_grid", cfg.model.get("sampling_grid", "cosine"))
            ),
            categorical_sampling_policy=str(
                sampler_cfg.get(
                    "categorical_sampling_policy",
                    cfg.model.get("categorical_sampling_policy", "legacy_low"),
                )
            ),
            guidance_scale=float(
                sampler_cfg.get("guidance_scale", cfg.model.get("guidance_scale", 0.0))
            ),
        )

    if name == "ddpm":
        from sticky.models.ddpm.ddpm_model import DDPM

        return DDPM(
            data_shape=data_shape,
            timesteps=int(cfg.model.get("timesteps", 1000)),
            beta_schedule=str(cfg.model.get("beta_schedule", "linear")),
            beta_start=float(cfg.model.get("beta_start", 1e-4)),
            beta_end=float(cfg.model.get("beta_end", 2e-2)),
            prediction_type=str(cfg.model.get("prediction_type", "eps")),
            variance_type=str(cfg.model.get("variance_type", "fixed_small")),
            clip_x0=bool(cfg.model.get("clip_x0", True)),
            feature_dim=int(cfg.model.get("feature_dim", 96)),
            ch_mult=tuple(cfg.model.get("ch_mult", (3, 4, 4))),
            dropout_rate=float(cfg.model.get("dropout_rate", 0.1)),
            image_backbone=str(cfg.model.get("image_backbone", "adm_unet2d")),
            adm_num_res_blocks=int(cfg.model.get("adm_num_res_blocks", 4)),
            adm_attention_resolutions=tuple(cfg.model.get("adm_attention_resolutions", (2, 4))),
            adm_num_heads=int(cfg.model.get("adm_num_heads", 4)),
            adm_num_head_channels=int(cfg.model.get("adm_num_head_channels", 64)),
            adm_num_heads_upsample=int(cfg.model.get("adm_num_heads_upsample", -1)),
            adm_conv_resample=bool(cfg.model.get("adm_conv_resample", True)),
            adm_use_scale_shift_norm=bool(cfg.model.get("adm_use_scale_shift_norm", True)),
            adm_resblock_updown=bool(cfg.model.get("adm_resblock_updown", False)),
            adm_use_conv_skip=bool(cfg.model.get("adm_use_conv_skip", False)),
            adm_use_new_attention_order=bool(cfg.model.get("adm_use_new_attention_order", False)),
            classes=int(cfg.model.get("classes", -1)),
        )

    raise ValueError(f"Unknown model.name={name}")
