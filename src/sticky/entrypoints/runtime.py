from __future__ import annotations

import os

from omegaconf import DictConfig


def apply_runtime_env(cfg: DictConfig) -> None:
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
