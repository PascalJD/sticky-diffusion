from __future__ import annotations

from typing import Any, Optional, Tuple

import hydra
import jax
from omegaconf import DictConfig


def _build_non_sjd_sampling_fns(
    simple_generate,
    *,
    model: Any,
    sample_timesteps: int,
    num_log_images: int,
    fid_every: int,
    fid_batch_size: int,
    validate=None,
) -> Tuple[Optional[Any], Optional[Any]]:
    if validate is not None:
        validate(model=model, timesteps=sample_timesteps)

    def _sample_images(params, rng, batch_size: int):
        sample_state = {"params": params, "ema_params": None}
        return simple_generate(
            rng,
            sample_state,
            model=model,
            batch_size=batch_size,
            timesteps=sample_timesteps,
            conditioning=None,
            use_ema=False,
        )

    sample_images_jit = jax.jit(lambda p, r: _sample_images(p, r, num_log_images))
    sample_images_fid_jit = None
    if fid_every > 0:
        if fid_batch_size == num_log_images:
            sample_images_fid_jit = sample_images_jit
        else:
            sample_images_fid_jit = jax.jit(lambda p, r: _sample_images(p, r, fid_batch_size))
    return sample_images_jit, sample_images_fid_jit


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

        sample_images_jit, sample_images_fid_jit = _build_non_sjd_sampling_fns(
            md4_sampling.simple_generate,
            model=model,
            sample_timesteps=sample_timesteps,
            num_log_images=num_log_images,
            fid_every=fid_every,
            fid_batch_size=fid_batch_size,
        )

    elif str(cfg.model.name) == "mdlm":
        from sticky.models.mdlm import sampling as mdlm_sampling

        sample_images_jit, sample_images_fid_jit = _build_non_sjd_sampling_fns(
            mdlm_sampling.simple_generate,
            model=model,
            sample_timesteps=sample_timesteps,
            num_log_images=num_log_images,
            fid_every=fid_every,
            fid_batch_size=fid_batch_size,
        )

    elif str(cfg.model.name) == "d3pm":
        from sticky.models.d3pm import sampling as d3pm_sampling

        def _validate_d3pm(*, model: Any, timesteps: int) -> None:
            d3pm_sampling.validate_timesteps(model=model, timesteps=timesteps)
            d3pm_sampling.validate_sampling_grid(model=model)

        sample_images_jit, sample_images_fid_jit = _build_non_sjd_sampling_fns(
            d3pm_sampling.simple_generate,
            model=model,
            sample_timesteps=sample_timesteps,
            num_log_images=num_log_images,
            fid_every=fid_every,
            fid_batch_size=fid_batch_size,
            validate=_validate_d3pm,
        )

    elif str(cfg.model.name) == "cadd":
        from sticky.models.cadd import sampling as cadd_sampling

        sample_images_jit, sample_images_fid_jit = _build_non_sjd_sampling_fns(
            cadd_sampling.simple_generate,
            model=model,
            sample_timesteps=sample_timesteps,
            num_log_images=num_log_images,
            fid_every=fid_every,
            fid_batch_size=fid_batch_size,
        )

    elif str(cfg.model.name) == "candi":
        from sticky.models.candi import sampling as candi_sampling

        sample_images_jit, sample_images_fid_jit = _build_non_sjd_sampling_fns(
            candi_sampling.simple_generate,
            model=model,
            sample_timesteps=sample_timesteps,
            num_log_images=num_log_images,
            fid_every=fid_every,
            fid_batch_size=fid_batch_size,
            validate=candi_sampling.validate_timesteps,
        )

    elif str(cfg.model.name) == "ddpm":
        from sticky.models.ddpm import sampling as ddpm_sampling

        sample_images_jit, sample_images_fid_jit = _build_non_sjd_sampling_fns(
            ddpm_sampling.simple_generate,
            model=model,
            sample_timesteps=sample_timesteps,
            num_log_images=num_log_images,
            fid_every=fid_every,
            fid_batch_size=fid_batch_size,
            validate=ddpm_sampling.validate_timesteps,
        )

    elif str(cfg.model.name) == "bitdiff":
        from sticky.models.bitdiff import sampling as bitdiff_sampling

        sample_images_jit, sample_images_fid_jit = _build_non_sjd_sampling_fns(
            bitdiff_sampling.simple_generate,
            model=model,
            sample_timesteps=sample_timesteps,
            num_log_images=num_log_images,
            fid_every=fid_every,
            fid_batch_size=fid_batch_size,
            validate=bitdiff_sampling.validate_timesteps,
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
            n_steps=int(sample_timesteps),
            sampling_grid=str(cfg.sampler.get("sampling_grid", "uniform")),
            score_scale=float(cfg.sampler.get("score_scale", 1.0)),
            logit_temperature=float(
                cfg.sampler.get(
                    "logit_temperature",
                    cfg.sampler.get("temperature", 1.0),
                )
            ),
            categorical_sampling_policy=str(
                cfg.sampler.get("categorical_sampling_policy", "legacy_low")
            ),
            hazard_mode=str(cfg.sampler.get("hazard_mode", "plugin")),
            alloc_mode=str(cfg.sampler.get("alloc_mode", "sample")),
            intensity_mode=str(cfg.sampler.get("intensity_mode", "full")),
            log_ratio_clip=float(cfg.sampler.get("log_ratio_clip", 10.0)),
            intensity_chunk_size=int(cfg.sampler.get("intensity_chunk_size", 256)),
            init_std=float(cfg.sampler.get("init_std", 1.0)),
            force_classify_at_end=bool(cfg.sampler.get("force_classify_at_end", True)),
            refresh_logits_after_em_step=bool(
                cfg.sampler.get("refresh_logits_after_em_step", False)
            ),
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
