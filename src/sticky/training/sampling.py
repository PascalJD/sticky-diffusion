from __future__ import annotations

from typing import Any, Optional, Tuple

import hydra
import jax
from omegaconf import DictConfig

def build_sampling_fns(
    *,
    cfg: DictConfig,
    task: Any,
    model: Any,
    num_log_images: int,
    sample_timesteps: int,
    fid_every: int,
    fid_batch_size: int,
) -> Tuple[Optional[Any], Optional[Any]]:
    sample_images_jit = None
    sample_images_fid_jit = None

    if str(cfg.model.name) == "md4":
        from sticky.models.md4 import sampling as md4_sampling

        def _sample_images_md4(params, rng, batch_size: int):
            sample_state = {"params": params, "ema_params": None}
            return md4_sampling.simple_generate(
                rng,
                sample_state,
                model=model,
                batch_size=batch_size,
                timesteps=sample_timesteps,
                conditioning=None,
                use_ema=False,
            )

        sample_images_jit = jax.jit(lambda p, r: _sample_images_md4(p, r, num_log_images))

        if fid_every > 0:
            if fid_batch_size == num_log_images:
                sample_images_fid_jit = sample_images_jit
            else:
                sample_images_fid_jit = jax.jit(
                    lambda p, r: _sample_images_md4(p, r, fid_batch_size)
                )

    elif str(cfg.model.name) == "sjd":
        from sticky.models.sjd import sampling as sjd_sampling
        from sticky.models.sjd.anchors import AnchorTable
        from sticky.models.sjd.sampler import SamplerConfig

        beta = getattr(task, "beta", None)
        if beta is None:
            beta = hydra.utils.instantiate(cfg.forward.beta)
        hazard = hydra.utils.instantiate(cfg.forward.hazard, beta=beta)
        jump = hydra.utils.instantiate(cfg.forward.jump, beta=beta)

        sampler_cfg = SamplerConfig(
            T=float(cfg.sampler.get("T", 1.0)),
            n_steps=int(cfg.sampler.get("n_steps", 250)),
            score_scale=float(cfg.sampler.get("score_scale", 1.0)),
            hazard_mode=str(cfg.sampler.get("hazard_mode", "plugin")),
            alloc_mode=str(cfg.sampler.get("alloc_mode", "argmax")),
            log_ratio_clip=float(cfg.sampler.get("log_ratio_clip", 10.0)),
            intensity_chunk_size=int(cfg.sampler.get("intensity_chunk_size", 256)),
            init_std=float(cfg.sampler.get("init_std", 1.0)),
            force_classify_at_end=bool(cfg.sampler.get("force_classify_at_end", True)),
        )

        def _sample_images_sjd(params, rng, batch_size: int):
            a_table = model.apply({"params": params}, method=model.anchor_table)
            anchors = AnchorTable(table_float=a_table)
            return sjd_sampling.simple_generate(
                rng=rng,
                params=params,
                model=model,
                anchors=anchors,
                beta=beta,
                hazard=hazard,
                jump=jump,
                cfg=sampler_cfg,
                batch_size=batch_size,
                shape=tuple(task.spec.data_shape),
            )

        sample_images_jit = jax.jit(lambda p, r: _sample_images_sjd(p, r, num_log_images))

        if fid_every > 0:
            if fid_batch_size == num_log_images:
                sample_images_fid_jit = sample_images_jit
            else:
                sample_images_fid_jit = jax.jit(
                    lambda p, r: _sample_images_sjd(p, r, fid_batch_size)
                )

    return sample_images_jit, sample_images_fid_jit
