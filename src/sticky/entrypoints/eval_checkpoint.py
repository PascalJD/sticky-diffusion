from __future__ import annotations

import json
import os

import hydra
from omegaconf import DictConfig, OmegaConf

import tensorflow as tf


tf.config.set_visible_devices([], "GPU")


def _apply_runtime_env(cfg: DictConfig) -> None:
    runtime = cfg.experiment.get("runtime", {})

    xla_preallocate = runtime.get("xla_preallocate", None)
    if xla_preallocate is not None:
        os.environ.setdefault(
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "true" if bool(xla_preallocate) else "false",
        )

    xla_mem_fraction = runtime.get("xla_mem_fraction", None)
    if xla_mem_fraction not in (None, "", "null"):
        os.environ.setdefault(
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            str(float(xla_mem_fraction)),
        )

    xla_disable_command_buffer = runtime.get("xla_disable_command_buffer", False)
    if bool(xla_disable_command_buffer):
        token = "--xla_gpu_enable_command_buffer="
        xla_flags = os.environ.get("XLA_FLAGS", "")
        if token not in xla_flags:
            os.environ["XLA_FLAGS"] = f"{xla_flags} {token}".strip()


@hydra.main(
    version_base=None,
    config_path="../../../config",
    config_name="eval_checkpoint.yaml",
)
def main(cfg: DictConfig):
    _apply_runtime_env(cfg)
    from sticky.training import maybe_init_wandb
    from sticky.training.offline_eval import run_offline_checkpoint_eval

    print("Resolved config:\n", OmegaConf.to_yaml(cfg))

    wandb_mod = maybe_init_wandb(cfg)
    try:
        metrics = run_offline_checkpoint_eval(
            cfg=cfg.experiment,
            eval_cfg=cfg.eval,
            offline_cfg=cfg.offline_eval,
            wandb_mod=wandb_mod,
        )
        print(
            "[offline-eval] Metrics:",
            json.dumps(metrics, indent=2, sort_keys=True),
            flush=True,
        )
    finally:
        if wandb_mod is not None:
            wandb_mod.finish()


if __name__ == "__main__":
    main()

