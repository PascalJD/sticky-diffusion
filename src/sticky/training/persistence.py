from __future__ import annotations

import json
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import jax
from flax.training import checkpoints

from sticky.rng import legacy_prng_key_data


def get_hydra_output_dir() -> Path:
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            output_dir = HydraConfig.get().runtime.output_dir
            if output_dir:
                return Path(output_dir)
    except Exception:
        pass
    return Path.cwd()


def resolve_run_path(
    path_like: Optional[str],
    default_rel: str,
    *,
    base_dir: Optional[Path] = None,
) -> Path:
    if path_like in (None, "", "null"):
        path = Path(default_rel)
    else:
        path = Path(str(path_like))
    if path.is_absolute():
        return path
    base = base_dir if base_dir is not None else get_hydra_output_dir()
    return base / path


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


def count_parameters(params: Any) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(int(getattr(leaf, "size", 0)) for leaf in leaves))


def count_parameters_by_top_level(params: Any) -> Dict[str, int]:
    if not isinstance(params, AbcMapping):
        return {}
    out: Dict[str, int] = {}
    for key, subtree in params.items():
        leaves = jax.tree_util.tree_leaves(subtree)
        out[str(key)] = int(sum(int(getattr(leaf, "size", 0)) for leaf in leaves))
    return out


def _resolved_cfg_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            out = OmegaConf.to_container(cfg, resolve=True)
            return out if isinstance(out, dict) else {"value": out}
    except Exception:
        pass
    if isinstance(cfg, dict):
        return dict(cfg)
    return {"value": str(cfg)}


def write_run_context(
    *,
    run_dir: Path,
    experiment_cfg: Any,
    eval_cfg: Any,
    params: Any,
    metrics_dir: Path,
    checkpoint_dir: Path,
):
    if jax.process_index() != 0:
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    total_params = count_parameters(params)
    top_level = count_parameters_by_top_level(params)
    payload: Dict[str, Any] = {
        "timestamp_utc": now_utc_iso(),
        "run_dir": str(run_dir),
        "metrics_dir": str(metrics_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "parameter_count_total": total_params,
        "parameter_count_millions": round(total_params / 1_000_000.0, 3),
        "parameter_count_by_top_level": top_level,
        "config": {
            "experiment": _resolved_cfg_dict(experiment_cfg),
            "eval": _resolved_cfg_dict(eval_cfg),
        },
    }

    (run_dir / "run_context.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"timestamp_utc: {payload['timestamp_utc']}",
        f"run_dir: {payload['run_dir']}",
        f"metrics_dir: {payload['metrics_dir']}",
        f"checkpoint_dir: {payload['checkpoint_dir']}",
        f"parameter_count_total: {total_params:,}",
        f"parameter_count_millions: {payload['parameter_count_millions']}",
    ]
    if top_level:
        lines.append("parameter_count_by_top_level:")
        for name, count in sorted(top_level.items()):
            lines.append(f"  - {name}: {count:,}")
    (run_dir / "run_context.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        from omegaconf import OmegaConf

        cfg_yaml = OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "experiment": _resolved_cfg_dict(experiment_cfg),
                    "eval": _resolved_cfg_dict(eval_cfg),
                }
            ),
            resolve=True,
        )
        (run_dir / "resolved_config.yaml").write_text(cfg_yaml, encoding="utf-8")
    except Exception:
        (run_dir / "resolved_config.json").write_text(
            json.dumps(payload["config"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


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

    def _checkpoint_compatible_target(self, target: Any) -> Any:
        rng = getattr(target, "rng", None)
        if rng is None:
            return target
        try:
            return target.replace(rng=legacy_prng_key_data(rng))
        except Exception:
            return target

    def _save(self, *, target, step_i: int, ckpt_dir: Path, keep: int, overwrite: bool):
        if jax.process_index() != 0:
            return
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_target = self._checkpoint_compatible_target(target)
        host_target = jax.device_get(save_target)
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
