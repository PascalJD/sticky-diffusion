from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig, OmegaConf


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
    config_name="prescreen_anchors.yaml",
)
def main(cfg: DictConfig):
    _apply_runtime_env(cfg)
    from sticky.prescreen import run_anchor_prescreen

    print("Resolved config:\n", OmegaConf.to_yaml(cfg))
    artifacts = run_anchor_prescreen(cfg)
    print("Prescreen artifacts:\n", OmegaConf.to_yaml(OmegaConf.create(artifacts)))


if __name__ == "__main__":
    main()
