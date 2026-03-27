from __future__ import annotations

import hydra
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from sticky.cli.runtime import apply_runtime_env
from sticky.core.config_paths import hydra_config_path


tf.config.set_visible_devices([], "GPU")


@hydra.main(
    version_base=None,
    config_path=hydra_config_path(),
    config_name="config.yaml",
)
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
