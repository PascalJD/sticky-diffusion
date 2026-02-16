from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import jax
from flax.training import checkpoints


def resolve_run_path(path_like: Optional[str], default_rel: str) -> Path:
    if path_like in (None, "", "null"):
        path = Path(default_rel)
    else:
        path = Path(str(path_like))
    if path.is_absolute():
        return path
    return Path.cwd() / path


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def metrics_to_floats(metrics: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


@dataclass
class MetricsWriter:
    root_dir: Path
    every_steps: int

    def __post_init__(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.root_dir / "metrics.jsonl"
        self.latest_path = self.root_dir / "latest_metrics.json"
        self.final_path = self.root_dir / "final_metrics.json"

    def should_write(self, step_i: int) -> bool:
        return (self.every_steps > 0) and ((step_i % self.every_steps) == 0)

    def write(self, *, step_i: int, metrics: Mapping[str, Any], tag: str = "periodic"):
        payload = {
            "timestamp_utc": now_utc_iso(),
            "step": int(step_i),
            "tag": str(tag),
            "metrics": metrics_to_floats(metrics),
        }
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
        self.latest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_final(self, *, step_i: int, metrics: Mapping[str, Any]):
        payload = {
            "timestamp_utc": now_utc_iso(),
            "step": int(step_i),
            "tag": "final",
            "metrics": metrics_to_floats(metrics),
        }
        self.final_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


@dataclass
class CheckpointWriter:
    root_dir: Path
    every_steps: int
    keep: int = 5
    save_final: bool = True
    best_metric_key: str = "eval/fid"
    best_mode: str = "min"

    def __post_init__(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir = self.root_dir / "best"
        self.final_dir = self.root_dir / "final"
        self.best_meta_path = self.root_dir / "best_metric.json"
        self._best_value: Optional[float] = None
        self._best_step: Optional[int] = None

    def _save(self, *, target, step_i: int, ckpt_dir: Path, keep: int, overwrite: bool):
        if jax.process_index() != 0:
            return
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        host_target = jax.device_get(target)
        checkpoints.save_checkpoint(
            ckpt_dir=str(ckpt_dir),
            target=host_target,
            step=int(step_i),
            keep=int(max(1, keep)),
            overwrite=bool(overwrite),
        )

    def maybe_save_periodic(self, *, target, step_i: int):
        if self.every_steps <= 0:
            return
        if (step_i % self.every_steps) != 0:
            return
        self._save(
            target=target,
            step_i=step_i,
            ckpt_dir=self.root_dir,
            keep=self.keep,
            overwrite=False,
        )

    def maybe_save_best(self, *, target, step_i: int, metrics: Mapping[str, Any]) -> bool:
        if self.best_metric_key not in metrics:
            return False
        try:
            value = float(metrics[self.best_metric_key])
        except Exception:
            return False

        improved = False
        if self._best_value is None:
            improved = True
        elif self.best_mode == "min":
            improved = value < self._best_value
        else:
            improved = value > self._best_value

        if not improved:
            return False

        self._best_value = value
        self._best_step = int(step_i)
        self._save(
            target=target,
            step_i=step_i,
            ckpt_dir=self.best_dir,
            keep=1,
            overwrite=True,
        )
        if jax.process_index() == 0:
            payload = {
                "timestamp_utc": now_utc_iso(),
                "metric": self.best_metric_key,
                "mode": self.best_mode,
                "value": float(self._best_value),
                "step": int(self._best_step),
            }
            self.best_meta_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return True

    def save_final_checkpoint(self, *, target, step_i: int):
        if not self.save_final:
            return
        self._save(
            target=target,
            step_i=step_i,
            ckpt_dir=self.final_dir,
            keep=1,
            overwrite=True,
        )
