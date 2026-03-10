from __future__ import annotations

from omegaconf import OmegaConf

from sticky.prescreen.anchors import _candidate_from_cfg


def test_candidate_from_cfg_accepts_missing_anchor_overrides():
    cfg = OmegaConf.create({"name": "normal_64", "preset": "normal_64", "learnable": False})
    candidate = _candidate_from_cfg(cfg)
    assert candidate.anchor_overrides == {}


def test_candidate_from_cfg_accepts_dict_anchor_overrides():
    cfg = OmegaConf.create(
        {
            "name": "normal_scaled",
            "preset": "normal_64",
            "learnable": False,
            "anchor_overrides": {"transform": {"center_columns": True}},
        }
    )
    candidate = _candidate_from_cfg(cfg)
    assert candidate.anchor_overrides == {"transform": {"center_columns": True}}
