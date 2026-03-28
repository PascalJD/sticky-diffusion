from __future__ import annotations

import sys
from types import SimpleNamespace

from omegaconf import OmegaConf

from sticky.models.factory import build_model


class _DummyCADD:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyMD4:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyMDLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyMDM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyD3PM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyBitDiff:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyCANDI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_build_model_reads_cadd_sampling_knobs_from_sampler_config(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.models.baselines.cadd.cadd_model",
        SimpleNamespace(CADD=_DummyCADD),
    )

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "cadd",
                "cont_time": True,
                "timesteps": 512,
                "discrete_schedule_type": "linear",
                "schedule_eps": 0.0,
                "feature_dim": 96,
                "num_heads": 12,
                "n_layers": 32,
                "n_dit_layers": 0,
                "dit_num_heads": 12,
                "dit_hidden_size": 768,
                "ch_mult": [3, 4, 4],
                "dropout_rate": 0.1,
                "use_attn_dropout": True,
                "mlp_type": "swiglu",
                "depth_scaled_init": False,
                "cond_type": "adaln",
                "model_sharding": False,
                "sequence_backbone": "auto",
                "image_backbone": "adm_unet5d",
                "adm_num_res_blocks": 4,
                "adm_attention_resolutions": [2, 4],
                "adm_num_heads": 4,
                "adm_num_head_channels": 64,
                "adm_num_heads_upsample": -1,
                "adm_conv_resample": True,
                "adm_use_scale_shift_norm": True,
                "adm_resblock_updown": False,
                "adm_use_conv_skip": False,
                "adm_use_new_attention_order": False,
                "time_features": "t",
                "classes": -1,
                "cadd_latent": {
                    "type": "gaussian",
                    "continuous_schedule_type": "linear",
                },
            },
            "sampler": {
                "sampling_grid": "uniform",
                "temperature_schedule": "constant",
                "tau_max": 3.5,
                "logit_temperature": 0.7,
                "z0_estimator": "soft",
                "K": 3,
                "force_decode_at_end": False,
                "categorical_sampling_policy": "exact",
                "corrector_enabled": True,
                "corrector_steps": 2,
                "corrector_remask_frac": 0.2,
                "corrector_metric": "neg_entropy",
                "corrector_sample_mode": "argmax",
            },
        }
    )

    model = build_model(cfg, data_shape=(32, 32, 3), vocab_size=256)

    assert isinstance(model, _DummyCADD)
    assert model.kwargs["sampling_grid"] == "uniform"
    assert model.kwargs["temperature_schedule"] == "constant"
    assert model.kwargs["tau_max"] == 3.5
    assert model.kwargs["logit_temperature"] == 0.7
    assert model.kwargs["z0_estimator"] == "soft"
    assert model.kwargs["K"] == 3
    assert model.kwargs["force_decode_at_end"] is False
    assert model.kwargs["categorical_sampling_policy"] == "exact"
    assert model.kwargs["corrector_enabled"] is True
    assert model.kwargs["corrector_steps"] == 2
    assert model.kwargs["corrector_remask_frac"] == 0.2
    assert model.kwargs["corrector_metric"] == "neg_entropy"
    assert model.kwargs["corrector_sample_mode"] == "argmax"


def test_build_model_reads_md4_sampling_knobs_from_sampler_config(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.models.baselines.md4.md4_model",
        SimpleNamespace(MD4=_DummyMD4),
    )

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "md4",
                "cont_time": True,
                "timesteps": 256,
                "feature_dim": 128,
                "num_heads": 12,
                "antithetic_time_sampling": True,
                "n_layers": 32,
                "n_dit_layers": 0,
                "dit_num_heads": 12,
                "dit_hidden_size": 768,
                "ch_mult": [1],
                "noise_schedule_type": "linear",
                "dropout_rate": 0.1,
                "use_attn_dropout": True,
                "mlp_type": "swiglu",
                "depth_scaled_init": True,
                "cond_type": "adaln_zero",
                "outside_embed": False,
                "sequence_backbone": "auto",
                "image_backbone": "unet5d",
                "adm_num_res_blocks": 2,
                "adm_attention_resolutions": [2, 4, 8],
                "adm_num_heads": 4,
                "adm_num_head_channels": -1,
                "adm_num_heads_upsample": -1,
                "adm_conv_resample": True,
                "adm_use_scale_shift_norm": True,
                "adm_resblock_updown": False,
                "adm_use_conv_skip": False,
                "adm_use_new_attention_order": False,
                "time_features": "t",
                "classes": -1,
                "model_sharding": False,
            },
            "sampler": {
                "method": "topp",
                "sampling_grid": "uniform",
                "topp": 0.91,
                "categorical_sampling_policy": "jax_high",
            },
        }
    )

    model = build_model(cfg, data_shape=(32, 32, 3), vocab_size=256)

    assert isinstance(model, _DummyMD4)
    assert model.kwargs["sampler"] == "topp"
    assert model.kwargs["sampling_grid"] == "uniform"
    assert model.kwargs["topp"] == 0.91
    assert model.kwargs["categorical_sampling_policy"] == "jax_high"


def test_build_model_reads_mdlm_sampling_knobs_from_sampler_config(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.models.baselines.mdlm.mdlm_model",
        SimpleNamespace(MDLM=_DummyMDLM),
    )

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "mdlm",
                "cont_time": True,
                "timesteps": 256,
                "feature_dim": 96,
                "num_heads": 12,
                "antithetic_time_sampling": True,
                "n_layers": 32,
                "n_dit_layers": 0,
                "dit_num_heads": 12,
                "dit_hidden_size": 768,
                "ch_mult": [3, 4, 4],
                "noise_schedule_type": "linear",
                "dropout_rate": 0.1,
                "use_attn_dropout": True,
                "mlp_type": "swiglu",
                "depth_scaled_init": True,
                "cond_type": "adaln_zero",
                "outside_embed": False,
                "sequence_backbone": "auto",
                "image_backbone": "adm_unet5d",
                "adm_num_res_blocks": 4,
                "adm_attention_resolutions": [2, 4],
                "adm_num_heads": 4,
                "adm_num_head_channels": 64,
                "adm_num_heads_upsample": -1,
                "adm_conv_resample": True,
                "adm_use_scale_shift_norm": True,
                "adm_resblock_updown": False,
                "adm_use_conv_skip": False,
                "adm_use_new_attention_order": False,
                "time_features": "none",
                "classes": -1,
                "cache_predictions": False,
                "model_sharding": False,
            },
            "sampler": {
                "method": "top_prob_margin",
                "sampling_grid": "loglinear",
                "topp": 0.95,
                "oracle_noise_type": "gumbel",
                "oracle_noise_scale": 0.5,
                "categorical_sampling_policy": "exact",
                "revealed_token_sample_mode": "argmax",
                "cache_predictions": True,
            },
        }
    )

    model = build_model(cfg, data_shape=(32, 32, 3), vocab_size=256)

    assert isinstance(model, _DummyMDLM)
    assert model.kwargs["image_backbone"] == "adm_unet5d"
    assert model.kwargs["sampler"] == "top_prob_margin"
    assert model.kwargs["sampling_grid"] == "loglinear"
    assert model.kwargs["topp"] == 0.95
    assert model.kwargs["oracle_noise_type"] == "gumbel"
    assert model.kwargs["oracle_noise_scale"] == 0.5
    assert model.kwargs["categorical_sampling_policy"] == "exact"
    assert model.kwargs["revealed_token_sample_mode"] == "argmax"
    assert model.kwargs["cache_predictions"] is True


def test_build_model_reads_mdm_reweighting_and_decoding_knobs(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.models.baselines.mdm.mdm_model",
        SimpleNamespace(MDM=_DummyMDM),
    )

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "mdm",
                "cont_time": False,
                "timesteps": 50,
                "feature_dim": 32,
                "num_heads": 12,
                "antithetic_time_sampling": False,
                "n_layers": 3,
                "n_dit_layers": 0,
                "dit_num_heads": 12,
                "dit_hidden_size": 384,
                "ch_mult": [1],
                "noise_schedule_type": "loglinear",
                "dropout_rate": 0.1,
                "use_attn_dropout": True,
                "mlp_type": "gelu",
                "depth_scaled_init": False,
                "cond_type": "adaln",
                "outside_embed": False,
                "sequence_backbone": "gpt2_like",
                "sequence_mlp_hidden_dim": 1536,
                "sequence_max_length": 245,
                "sequence_causal": False,
                "image_backbone": "auto",
                "adm_num_res_blocks": 2,
                "adm_attention_resolutions": [2, 4],
                "adm_num_heads": 4,
                "adm_num_head_channels": -1,
                "adm_num_heads_upsample": -1,
                "adm_conv_resample": True,
                "adm_use_scale_shift_norm": True,
                "adm_resblock_updown": False,
                "adm_use_conv_skip": False,
                "adm_use_new_attention_order": False,
                "time_features": "none",
                "classes": -1,
                "cache_predictions": False,
                "token_reweighting": True,
                "alpha": 0.25,
                "gamma": 1.0,
                "time_reweighting": "linear",
                "model_sharding": False,
            },
            "sampler": {
                "method": "top_prob_margin",
                "sampling_grid": "loglinear",
                "categorical_sampling_policy": "exact",
                "decoding_style": "topk_remask",
                "oracle_noise_type": "gumbel",
                "oracle_noise_scale": 0.5,
                "revealed_token_sample_mode": "sample",
                "cache_predictions": False,
            },
        }
    )

    model = build_model(cfg, data_shape=(245,), vocab_size=12)

    assert isinstance(model, _DummyMDM)
    assert model.kwargs["decoding_style"] == "topk_remask"
    assert model.kwargs["token_reweighting"] is True
    assert model.kwargs["alpha"] == 0.25
    assert model.kwargs["gamma"] == 1.0
    assert model.kwargs["time_reweighting"] == "linear"
    assert model.kwargs["oracle_noise_type"] == "gumbel"
    assert model.kwargs["oracle_noise_scale"] == 0.5


def test_build_model_reads_d3pm_config_and_sampler_knobs(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.models.baselines.d3pm.d3pm_model",
        SimpleNamespace(D3PM=_DummyD3PM),
    )

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "d3pm",
                "timesteps": 256,
                "transition_type": "gaussian",
                "transition_beta_schedule": "linear",
                "beta_start": 1.0e-4,
                "beta_end": 2.0e-2,
                "cosine_s": 0.008,
                "max_beta": 0.999,
                "auxiliary_loss_weight": 1.0e-3,
                "absorbing_state": 128,
                "feature_dim": 96,
                "num_heads": 12,
                "antithetic_time_sampling": True,
                "n_layers": 32,
                "n_dit_layers": 0,
                "dit_num_heads": 12,
                "dit_hidden_size": 768,
                "ch_mult": [3, 4, 4],
                "dropout_rate": 0.1,
                "use_attn_dropout": True,
                "mlp_type": "swiglu",
                "depth_scaled_init": True,
                "cond_type": "adaln_zero",
                "outside_embed": False,
                "sequence_backbone": "auto",
                "image_backbone": "adm_unet5d",
                "adm_num_res_blocks": 4,
                "adm_attention_resolutions": [2, 4],
                "adm_num_heads": 4,
                "adm_num_head_channels": 64,
                "adm_num_heads_upsample": -1,
                "adm_conv_resample": True,
                "adm_use_scale_shift_norm": True,
                "adm_resblock_updown": False,
                "adm_use_conv_skip": False,
                "adm_use_new_attention_order": False,
                "time_features": "t",
                "classes": -1,
                "model_sharding": False,
            },
            "sampler": {
                "method": "ancestral",
                "sampling_grid": "uniform",
                "categorical_sampling_policy": "exact",
            },
        }
    )

    model = build_model(cfg, data_shape=(32, 32, 3), vocab_size=256)

    assert isinstance(model, _DummyD3PM)
    assert model.kwargs["transition_type"] == "gaussian"
    assert model.kwargs["transition_beta_schedule"] == "linear"
    assert model.kwargs["image_backbone"] == "adm_unet5d"
    assert model.kwargs["sampler"] == "ancestral"
    assert model.kwargs["sampling_grid"] == "uniform"
    assert model.kwargs["categorical_sampling_policy"] == "exact"


def test_build_model_reads_bitdiff_config_and_sampler_knobs(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.models.baselines.bitdiff.bitdiff_model",
        SimpleNamespace(BitDiffusion=_DummyBitDiff),
    )

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "bitdiff",
                "cont_time": True,
                "timesteps": 256,
                "num_bits": 8,
                "encoding": "uint8",
                "predict_target": "x0",
                "loss_type": "mse",
                "self_conditioning": True,
                "self_conditioning_rate": 0.5,
                "analog_bit_scale": 1.0,
                "clip_x0": True,
                "signal_schedule_type": "linear",
                "schedule_eps": 0.0,
                "feature_dim": 96,
                "num_heads": 12,
                "n_layers": 32,
                "n_dit_layers": 0,
                "dit_num_heads": 12,
                "dit_hidden_size": 768,
                "ch_mult": [3, 4, 4],
                "dropout_rate": 0.1,
                "use_attn_dropout": True,
                "mlp_type": "swiglu",
                "depth_scaled_init": False,
                "cond_type": "adaln",
                "model_sharding": False,
                "sequence_backbone": "auto",
                "image_backbone": "adm_unet5d",
                "adm_num_res_blocks": 4,
                "adm_attention_resolutions": [2, 4],
                "adm_num_heads": 4,
                "adm_num_head_channels": 64,
                "adm_num_heads_upsample": -1,
                "adm_conv_resample": True,
                "adm_use_scale_shift_norm": True,
                "adm_resblock_updown": False,
                "adm_use_conv_skip": False,
                "adm_use_new_attention_order": False,
                "time_features": "t",
                "classes": -1,
            },
            "sampler": {
                "method": "ddpm",
                "sampling_grid": "cosine",
                "time_difference": 0.25,
                "stochasticity": 1.0,
            },
        }
    )

    model = build_model(cfg, data_shape=(32, 32, 3), vocab_size=256)

    assert isinstance(model, _DummyBitDiff)
    assert model.kwargs["encoding"] == "uint8"
    assert model.kwargs["predict_target"] == "x0"
    assert model.kwargs["self_conditioning"] is True
    assert model.kwargs["image_backbone"] == "adm_unet5d"
    assert model.kwargs["sampler"] == "ddpm"
    assert model.kwargs["sampling_grid"] == "cosine"
    assert model.kwargs["time_difference"] == 0.25
    assert model.kwargs["stochasticity"] == 1.0


def test_build_model_reads_candi_config_and_sampler_knobs(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.models.baselines.candi.candi_model",
        SimpleNamespace(CANDI=_DummyCANDI),
    )

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "candi",
                "cont_time": True,
                "timesteps": 256,
                "representation": "embed",
                "experimental": True,
                "alpha_schedule_type": "linear",
                "schedule_eps": 0.0,
                "pure_continuous": True,
                "use_percentile_scheduling": True,
                "min_percentile": 0.01,
                "max_percentile": 0.45,
                "sigma_min": 0.2,
                "sigma_max": 4.0,
                "ode_step_scale": 0.75,
                "feature_dim": 96,
                "num_heads": 12,
                "n_layers": 32,
                "n_dit_layers": 0,
                "dit_num_heads": 12,
                "dit_hidden_size": 768,
                "ch_mult": [3, 4, 4],
                "dropout_rate": 0.1,
                "use_attn_dropout": True,
                "mlp_type": "swiglu",
                "depth_scaled_init": False,
                "cond_type": "adaln",
                "model_sharding": False,
                "sequence_backbone": "auto",
                "image_backbone": "adm_unet5d",
                "adm_num_res_blocks": 4,
                "adm_attention_resolutions": [2, 4],
                "adm_num_heads": 4,
                "adm_num_head_channels": 64,
                "adm_num_heads_upsample": -1,
                "adm_conv_resample": True,
                "adm_use_scale_shift_norm": True,
                "adm_resblock_updown": False,
                "adm_use_conv_skip": False,
                "adm_use_new_attention_order": False,
                "time_features": "t",
                "classes": -1,
            },
            "sampler": {
                "method": "hybrid_expected",
                "sampling_grid": "uniform",
                "categorical_sampling_policy": "exact",
                "guidance_scale": 0.0,
            },
        }
    )

    model = build_model(cfg, data_shape=(32, 32, 3), vocab_size=256)

    assert isinstance(model, _DummyCANDI)
    assert model.kwargs["representation"] == "embed"
    assert model.kwargs["experimental"] is True
    assert model.kwargs["pure_continuous"] is True
    assert model.kwargs["image_backbone"] == "adm_unet5d"
    assert model.kwargs["sampler"] == "hybrid_expected"
    assert model.kwargs["sampling_grid"] == "uniform"
    assert model.kwargs["categorical_sampling_policy"] == "exact"
    assert model.kwargs["ode_step_scale"] == 0.75
