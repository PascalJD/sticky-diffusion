from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_ROOT = _REPO_ROOT / "config"


def repo_root() -> Path:
    return _REPO_ROOT


def config_root() -> Path:
    return _CONFIG_ROOT


def hydra_config_path() -> str:
    return str(config_root())
