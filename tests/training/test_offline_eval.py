from __future__ import annotations

from pathlib import Path

from sticky.training.eval import resolve_from_original_cwd
from sticky.training.offline_eval import (
    _extract_forward_config_metadata,
    _resolve_run_dir_from_offline_cfg,
)


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


def test_resolve_from_original_cwd_keeps_empty_string_as_relative(monkeypatch):
    monkeypatch.setattr(
        "hydra.utils.get_original_cwd",
        lambda: "/tmp/sticky-original-cwd",
    )

    assert resolve_from_original_cwd("") == "/tmp/sticky-original-cwd"


def test_offline_eval_treats_empty_run_dir_as_unset():
    assert _resolve_run_dir_from_offline_cfg({"run_dir": ""}) is None
    assert _resolve_run_dir_from_offline_cfg({"run_dir": "null"}) is None


def test_offline_eval_resolves_run_dir_against_original_cwd(monkeypatch):
    monkeypatch.setattr(
        "hydra.utils.get_original_cwd",
        lambda: "/tmp/sticky-original-cwd",
    )

    assert _resolve_run_dir_from_offline_cfg({"run_dir": "runs/demo"}) == Path(
        "/tmp/sticky-original-cwd/runs/demo"
    )
