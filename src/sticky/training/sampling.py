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
        from sticky.models.baselines.md4 import sampling as md4_sampling

        sample_images_jit, sample_images_fid_jit = _build_non_sjd_sampling_fns(
            md4_sampling.simple_generate,
            model=model,
            sample_timesteps=sample_timesteps,
            num_log_images=num_log_images,
            fid_every=fid_every,
            fid_batch_size=fid_batch_size,
        )

    elif str(cfg.model.name) == "mdlm":
        from sticky.models.baselines.mdlm import sampling as mdlm_sampling

        sample_images_jit, sample_images_fid_jit = _build_non_sjd_sampling_fns(
            mdlm_sampling.simple_generate,
            model=model,
            sample_timesteps=sample_timesteps,
            num_log_images=num_log_images,
            fid_every=fid_every,
            fid_batch_size=fid_batch_size,
        )

    elif str(cfg.model.name) == "d3pm":
        from sticky.models.baselines.d3pm import sampling as d3pm_sampling

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
        from sticky.models.baselines.cadd import sampling as cadd_sampling

        sample_images_jit, sample_images_fid_jit = _build_non_sjd_sampling_fns(
            cadd_sampling.simple_generate,
            model=model,
            sample_timesteps=sample_timesteps,
            num_log_images=num_log_images,
            fid_every=fid_every,
            fid_batch_size=fid_batch_size,
        )

    elif str(cfg.model.name) == "candi":
        from sticky.models.baselines.candi import sampling as candi_sampling

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
        from sticky.models.baselines.ddpm import sampling as ddpm_sampling

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
        from sticky.models.baselines.bitdiff import sampling as bitdiff_sampling

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

        beta = getattr(task, "beta", None)
        if beta is None:
            beta = hydra.utils.instantiate(cfg.forward.beta)
        hazard = hydra.utils.instantiate(cfg.forward.hazard, beta=beta)
        jump = hydra.utils.instantiate(cfg.forward.jump, beta=beta)
        anchor_log_w = getattr(task, "anchor_log_w", None)

        sampler_cfg = _sjd_sampler_cfg_from_dict(
            _base_sampler_spec_from_cfg(cfg.sampler, sample_timesteps=sample_timesteps)
        )

        def _sample_images_sjd(params, rng, batch_size: int):
            a_table = model.apply({"params": params}, method=model.anchor_table)
            if anchor_log_w is not None:
                anchors = AnchorTable(table_float=a_table, log_w=anchor_log_w)
            else:
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


def _base_sampler_spec_from_cfg(cfg_sampler, *, sample_timesteps: int) -> dict:
    """Build a flat spec dict from a Hydra sampler config group.

    This is the single source of truth for reading ``cfg.sampler`` into the flat
    dict consumed by :func:`_sjd_sampler_cfg_from_dict`.  Both the primary
    ``build_sampling_fns`` SJD path and ``build_multi_fid_sampling_fns`` call
    this helper so that every ``SamplerConfig`` field is threaded through.
    """
    return {
        "T": float(cfg_sampler.get("T", 1.0)),
        "n_steps": int(sample_timesteps),
        "sampling_grid": str(cfg_sampler.get("sampling_grid", "uniform")),
        "score_from_classifier": bool(cfg_sampler.get("score_from_classifier", True)),
        "score_scale": float(cfg_sampler.get("score_scale", 1.0)),
        "logit_temperature": float(
            cfg_sampler.get("logit_temperature", cfg_sampler.get("temperature", 1.0))
        ),
        "categorical_sampling_policy": str(
            cfg_sampler.get("categorical_sampling_policy", "legacy_low")
        ),
        "hazard_mode": str(cfg_sampler.get("hazard_mode", "plugin")),
        "alloc_mode": str(cfg_sampler.get("alloc_mode", "sample")),
        "log_ratio_clip": float(cfg_sampler.get("log_ratio_clip", 10.0)),
        "init_std": float(cfg_sampler.get("init_std", 1.0)),
        "eps_denom": float(cfg_sampler.get("eps_denom", 1e-12)),
        "force_classify_at_end": bool(cfg_sampler.get("force_classify_at_end", True)),
        "refresh_logits_after_em_step": bool(
            cfg_sampler.get("refresh_logits_after_em_step", False)
        ),
        "metrics_count_nfe": bool(cfg_sampler.get("metrics_count_nfe", True)),
        # tau-grid resolution for the SJD plug-in / classifier_induced_score.
        "tau_grid_size": int(cfg_sampler.get("tau_grid_size", 32)),
    }


def _sjd_sampler_cfg_from_dict(spec: dict) -> "SamplerConfig":
    """Build a SamplerConfig from a flat dict of sampler fields."""
    from sticky.models.sjd.sampler import SamplerConfig

    return SamplerConfig(
        T=float(spec.get("T", 1.0)),
        n_steps=int(spec["n_steps"]),
        sampling_grid=str(spec.get("sampling_grid", "uniform")),
        score_from_classifier=bool(spec.get("score_from_classifier", True)),
        score_scale=float(spec.get("score_scale", 1.0)),
        logit_temperature=float(spec.get("logit_temperature", 1.0)),
        categorical_sampling_policy=str(spec.get("categorical_sampling_policy", "legacy_low")),
        hazard_mode=str(spec.get("hazard_mode", "plugin")),
        alloc_mode=str(spec.get("alloc_mode", "sample")),
        log_ratio_clip=float(spec.get("log_ratio_clip", 10.0)),
        init_std=float(spec.get("init_std", 1.0)),
        eps_denom=float(spec.get("eps_denom", 1e-12)),
        force_classify_at_end=bool(spec.get("force_classify_at_end", True)),
        refresh_logits_after_em_step=bool(spec.get("refresh_logits_after_em_step", False)),
        metrics_count_nfe=bool(spec.get("metrics_count_nfe", True)),
        tau_grid_size=int(spec.get("tau_grid_size", 32)),
    )


def build_multi_fid_sampling_fns(
    *,
    cfg: DictConfig,
    task: Any,
    model: Any,
    fid_batch_size: int,
    sample_timesteps: int,
    eval_sjd_runs: dict,
) -> dict:
    """Build a dict of labelled JIT'd SJD FID sampling functions.

    Each entry in *eval_sjd_runs* is ``{label: str, ...sampler_overrides}``.
    Overrides are merged on top of the base sampler config from *cfg.sampler*.

    Returns ``{label: jitted_sample_fn}`` where each ``sample_fn(params, rng)``
    returns a ``ReverseSampleResult``.
    """
    if str(cfg.model.name) != "sjd":
        raise ValueError(
            "build_multi_fid_sampling_fns only supports SJD models, "
            f"got model.name={cfg.model.name!r}."
        )

    from sticky.models.sjd import sampling as sjd_sampling
    from sticky.models.sjd.anchors import AnchorTable

    beta = getattr(task, "beta", None)
    if beta is None:
        beta = hydra.utils.instantiate(cfg.forward.beta)
    hazard = hydra.utils.instantiate(cfg.forward.hazard, beta=beta)
    jump = hydra.utils.instantiate(cfg.forward.jump, beta=beta)
    anchor_log_w = getattr(task, "anchor_log_w", None)

    # Base sampler spec from the experiment config.
    base_spec: dict = _base_sampler_spec_from_cfg(
        cfg.sampler, sample_timesteps=sample_timesteps
    )

    data_shape = tuple(task.spec.data_shape)
    result: dict = {}

    for run_key, run_entry in eval_sjd_runs.items():
        if not isinstance(run_entry, dict):
            run_entry = dict(run_entry)
        label = str(run_entry.get("label", run_key)).strip()
        if not label:
            label = str(run_key)

        # Merge overrides on top of base.
        spec = dict(base_spec)
        for k, v in run_entry.items():
            if k == "label":
                continue
            if k in spec and v is not None:
                spec[k] = v

        sampler_cfg = _sjd_sampler_cfg_from_dict(spec)

        def _make_sample_fn(sc):
            def _sample(params, rng):
                a_table = model.apply({"params": params}, method=model.anchor_table)
                if anchor_log_w is not None:
                    anchors = AnchorTable(table_float=a_table, log_w=anchor_log_w)
                else:
                    anchors = AnchorTable(table_float=a_table)
                return sjd_sampling.simple_generate(
                    rng=rng,
                    params=params,
                    model=model,
                    anchors=anchors,
                    beta=beta,
                    hazard=hazard,
                    jump=jump,
                    cfg=sc,
                    batch_size=fid_batch_size,
                    shape=data_shape,
                )
            return jax.jit(_sample)

        result[label] = _make_sample_fn(sampler_cfg)

    return result
