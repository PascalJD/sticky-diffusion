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


def test_build_model_reads_cadd_sampling_knobs_from_sampler_config(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.models.cadd.cadd_model",
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
    assert model.kwargs["corrector_enabled"] is True
    assert model.kwargs["corrector_steps"] == 2
    assert model.kwargs["corrector_remask_frac"] == 0.2
    assert model.kwargs["corrector_metric"] == "neg_entropy"
    assert model.kwargs["corrector_sample_mode"] == "argmax"


def test_build_model_reads_md4_sampling_knobs_from_sampler_config(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.models.md4.md4_model",
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
            },
        }
    )

    model = build_model(cfg, data_shape=(32, 32, 3), vocab_size=256)

    assert isinstance(model, _DummyMD4)
    assert model.kwargs["sampler"] == "topp"
    assert model.kwargs["sampling_grid"] == "uniform"
    assert model.kwargs["topp"] == 0.91
