from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

import tensorflow as tf

from sticky.entrypoints.runtime import apply_runtime_env


tf.config.set_visible_devices([], "GPU")


@hydra.main(version_base=None, config_path="../../../config", config_name="config.yaml")
def main(cfg: DictConfig):
    apply_runtime_env(cfg)
    from sticky.training import main_train_loop, maybe_init_wandb

    print("Resolved config:\n", OmegaConf.to_yaml(cfg))
    wandb_mod = maybe_init_wandb(cfg)
    try:
        main_train_loop(cfg.experiment, wandb_mod=wandb_mod, eval_cfg=cfg.get("eval", None))
    finally:
        if wandb_mod is not None:
            wandb_mod.finish()


if __name__ == "__main__":
    main()
