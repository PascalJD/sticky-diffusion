from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf

import tensorflow as tf

from sticky.entrypoints.runtime import apply_runtime_env


tf.config.set_visible_devices([], "GPU")


@hydra.main(
    version_base=None,
    config_path="../../../config",
    config_name="eval_checkpoint.yaml",
)
def main(cfg: DictConfig):
    apply_runtime_env(cfg)
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
