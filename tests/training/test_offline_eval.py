from __future__ import annotations

from sticky.training.offline_eval import _extract_forward_config_metadata


def test_extract_forward_config_metadata_handles_missing_forward():
    cfg = {
        "task": {"name": "ddpm_cifar10"},
        "model": {"name": "ddpm"},
        "training": {"seed": 0},
        "sampler": {"n_steps": 1000},
    }

    meta = _extract_forward_config_metadata(cfg)

    assert meta == {
        "forward_beta_target": None,
        "forward_hazard_target": None,
        "forward_jump_target": None,
        "jump_eta": None,
    }


def test_extract_forward_config_metadata_reads_sjd_forward_fields():
    cfg = {
        "forward": {
            "beta": {"_target_": "sticky.beta.vp_linear"},
            "hazard": {"_target_": "sticky.hazard.poly_alpha"},
            "jump": {"_target_": "sticky.jump.vp_matched", "eta": 0.9},
        }
    }

    meta = _extract_forward_config_metadata(cfg)

    assert meta == {
        "forward_beta_target": "sticky.beta.vp_linear",
        "forward_hazard_target": "sticky.hazard.poly_alpha",
        "forward_jump_target": "sticky.jump.vp_matched",
        "jump_eta": 0.9,
    }
