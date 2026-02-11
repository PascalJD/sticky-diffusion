# src/sticky/entrypoints/train.py
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import hydra
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.jax_utils import replicate, unreplicate
from omegaconf import DictConfig, OmegaConf
import jax.profiler

from sticky.models.factory import build_model
from sticky.tasks.factory import build_task
from sticky.models.md4 import sampling as md4_sampling

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

try:
    import numpy as np
except Exception:
    np = None 

Array = jnp.ndarray


def _shard(batch: Dict[str, Array]) -> Dict[str, Array]:
    """(B, ...) -> (n_devices, B//n_devices, ...)"""
    n = jax.device_count()
    out = {}
    for k, v in batch.items():
        assert v.shape[0] % n == 0, f"Batch dim for {k} must be divisible by device_count."
        out[k] = v.reshape((n, v.shape[0] // n) + v.shape[1:])
    return out


def _to_numpy(x: Any):
    """Best-effort conversion of JAX arrays to NumPy."""
    if np is None:
        raise RuntimeError("NumPy not available but required for W&B image logging.")
    try:
        x = jax.device_get(x)
    except Exception:
        pass
    return np.asarray(x)


def _make_image_grid_np(images: "np.ndarray", max_images: int = 64) -> "np.ndarray":
    """
    Build a single HxWxC uint8 grid from a batch (N,H,W,C).
    Pure NumPy to avoid JAX/GPU work during logging.
    """
    assert images.ndim == 4, f"Expected (N,H,W,C), got {images.shape}"
    imgs = images[:max_images]
    n = imgs.shape[0]
    g = int(np.floor(np.sqrt(n)))
    g = max(1, g)
    imgs = imgs[: g * g]

    # (g*g, H, W, C) -> (g, g, H, W, C)
    imgs = imgs.reshape(g, g, imgs.shape[1], imgs.shape[2], imgs.shape[3])

    # stitch rows
    rows = [np.concatenate(list(imgs[r]), axis=1) for r in range(g)]
    grid = np.concatenate(rows, axis=0)
    return grid.astype(np.uint8)


def _is_scalar_array(x: Any) -> bool:
    """True if x is a scalar array (0-d) or size-1 array."""
    if hasattr(x, "shape"):
        try:
            return (x.shape == ()) or (getattr(x, "size", 0) == 1)
        except Exception:
            return False
    return False



def _to_py_scalar(x: Any) -> Optional[float]:
    """Convert JAX/NumPy scalar arrays to Python floats; return None if not scalar."""
    try:
        x = jax.device_get(x)
    except Exception:
        pass

    # Python scalars
    if isinstance(x, (int, float)):
        return float(x)

    # JAX/NumPy scalars or size-1 arrays
    if _is_scalar_array(x):
        try:
            # works for DeviceArray / numpy scalar
            return float(x)
        except Exception:
            pass
        try:
            # fallback
            if np is not None:
                return float(np.asarray(x).reshape(()))
        except Exception:
            return None

    return None


def _sanitize_metrics(metrics: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten nested dicts and keep only scalar entries (wandb likes scalars)."""
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(_sanitize_metrics(v, prefix=key))
        else:
            s = _to_py_scalar(v)
            if s is not None:
                out[key] = s
    return out


def _maybe_init_wandb(full_cfg: DictConfig):
    """Init W&B on host 0 only. Returns wandb module (or None)."""
    wb = full_cfg.get("wandb", None)
    if wb is None or not bool(wb.get("enabled", False)):
        return None

    # If multi-host, this prevents duplicate runs:
    if jax.process_index() != 0:
        return None

    mode = str(wb.get("mode", "online"))
    if mode == "disabled":
        return None

    try:
        import wandb  # local import so repo works without wandb installed
    except ImportError as e:
        raise ImportError(
            "Weights & Biases is enabled in config, but wandb isn't installed. "
            "Install with `pip install wandb` (or add it to environment.yml)."
        ) from e

    entity = wb.get("entity", None)
    if entity in (None, "null"):
        entity = None

    run_name = wb.get("run_name", None)
    # Fallback if config omitted it
    if run_name in (None, "", "null"):
        run_name = (
            f"{full_cfg.experiment.task.name}_{full_cfg.experiment.model.name}"
        )

    tags = wb.get("tags", None)
    tags = list(tags) if tags is not None else None

    # Log the resolved config
    resolved = OmegaConf.to_container(full_cfg, resolve=True)

    wandb.init(
        project=str(wb.project),
        entity=entity,
        name=str(run_name),
        tags=tags,
        mode=mode,
        config=resolved,
    )

    # Runtime info
    wandb.config.update(
        {
            "jax_device_count": jax.device_count(),
            "jax_platform": jax.default_backend(),
        },
        allow_val_change=True,
    )

    return wandb


@struct.dataclass
class TrainState:
    step: Array
    rng: Array
    params: Any
    ema_params: Any
    opt_state: optax.OptState


def _make_lr_schedule(cfg: DictConfig):
    # Warmup + cosine, matching MD4 config
    warmup_steps = int(cfg.optim.warmup_steps)
    total_steps = int(cfg.training.num_train_steps)
    base_lr = float(cfg.optim.learning_rate)

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(1, total_steps - warmup_steps),
        end_value=0.0,
    )


def _make_optimizer(cfg: DictConfig):
    lr_schedule = _make_lr_schedule(cfg)
    return optax.adamw(
        learning_rate=lr_schedule,
        b1=0.9,
        b2=float(cfg.optim.b2),
        weight_decay=float(cfg.optim.weight_decay),
    )


def _init_state(cfg: DictConfig, model, rng: jax.random.PRNGKey):
    name = str(cfg.model.name)

    # dummy input uses dataset data_shape
    B = int(cfg.dataset.per_device_batch_size) if int(cfg.dataset.per_device_batch_size) > 0 else 1

    if name == "md4":
        # MD4 init needs both 'params' and 'sample' rng streams.
        rng, rng_params, rng_sample = jax.random.split(rng, 3)
        dummy_x = jnp.zeros((B,) + tuple(cfg.dataset.data_shape), dtype=jnp.int32)
        dummy_cond = jnp.zeros((B,), dtype=jnp.int32) if int(cfg.model.classes) > 0 else None
        variables = model.init(
            {"params": rng_params, "sample": rng_sample},
            dummy_x,
            cond=dummy_cond,
            train=False,
        )

    elif name == "sjd":
        # SJD classifier is continuous: inputs are anchor vectors in R^d.
        rng, rng_params = jax.random.split(rng, 2)

        anchor_dim = int(cfg.model.anchor_dim)
        dummy_z = jnp.zeros((B,) + tuple(cfg.dataset.data_shape) + (anchor_dim,), dtype=jnp.float32)
        dummy_t = jnp.zeros((B,), dtype=jnp.float32)
        variables = model.init({"params": rng_params}, dummy_z, dummy_t, train=False)
        dummy_ids = jnp.zeros(tuple(dummy_z.shape[:-1]), dtype=jnp.int32)  # (B,H,W,C)

        rng, rng_anchors = jax.random.split(rng, 2)
        _, mutated = model.apply(
            variables,
            dummy_ids,
            method=model.embed,
            mutable=["params"],
            rngs={"params": rng_anchors},
        )
        variables = {**variables, "params": mutated["params"]}

    else:
        raise ValueError(f"Unknown model.name={name!r} for init")
    params = variables["params"]

    tx = _make_optimizer(cfg)
    opt_state = tx.init(params)

    ema_rate = float(cfg.training.ema_rate)
    if ema_rate > 0.0:
        ema_params = jax.tree_util.tree_map(lambda x: x, params)
    else:
        ema_params = None

    return TrainState(
        step=jnp.array(0, dtype=jnp.int32),
        rng=rng,
        params=params,
        ema_params=ema_params,
        opt_state=opt_state,
    ), tx


def main_train_loop(cfg: DictConfig, wandb_mod=None):
    task = build_task(cfg)
    model = build_model(
        cfg, 
        data_shape=task.spec.data_shape, 
        vocab_size=task.spec.vocab_size
    )
    num_log_images = int(cfg.training.num_log_images)
    sample_timesteps = int(cfg.training.sample_timesteps)
    num_train_steps = int(cfg.training.num_train_steps)
    log_images_every_steps = int(cfg.training.log_images_every_steps)
    log_every_steps = int(cfg.training.log_every_steps)

    rng = jax.random.PRNGKey(int(cfg.training.seed))
    state, tx = _init_state(cfg, model, rng)
    lr_schedule = _make_lr_schedule(cfg)

    train_iter, eval_iter = task.make_dataloaders(seed=int(cfg.training.seed))

    if cfg.runtime.platform == "auto":
        use_pmap = (jax.device_count() > 1)
    elif cfg.runtime.platform == "pmap":
        use_pmap = True
    else:
        use_pmap = False

    ema_rate = float(cfg.training.ema_rate)

    # Image sampling is model-specific.
    sample_images_jit = None

    if str(cfg.model.name) == "md4":
        def _sample_images_md4(params, rng):
            sample_state = {"params": params, "ema_params": None}
            return md4_sampling.simple_generate(
                rng,
                sample_state,
                model=model,
                batch_size=num_log_images,
                timesteps=sample_timesteps,
                conditioning=None,
                use_ema=False,
            )

        sample_images_jit = jax.jit(_sample_images_md4)

    elif str(cfg.model.name) == "sjd":
        # lazy imports
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
            init_std=float(cfg.sampler.get("init_std", 1.0)),
            force_classify_at_end=bool(cfg.sampler.get("force_classify_at_end", True)),
        )

        def _sample_images_sjd(params, rng):
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
                batch_size=num_log_images,
                shape=tuple(task.spec.data_shape),
            )

        sample_images_jit = jax.jit(_sample_images_sjd)

    def _log_images_to_wandb(step_i: int, gt_images, samples=None):
        if wandb_mod is None:
            return

        gt_np = np.clip(_to_numpy(gt_images), 0, 255).astype(np.uint8)
        gt_grid = _make_image_grid_np(gt_np, max_images=num_log_images)
        log_payload = {
            "images/ground_truth": wandb_mod.Image(gt_grid, caption="GT"),
        }

        if samples is not None:
            samp_np = np.clip(_to_numpy(samples), 0, 255).astype(np.uint8)
            samp_grid = _make_image_grid_np(samp_np, max_images=num_log_images)
            log_payload["images/samples"] = wandb_mod.Image(samp_grid, caption="Samples")

        wandb_mod.log(log_payload, step=step_i)

    def loss_and_metrics(params, rng, batch, train: bool):
        loss, metrics = task.loss_fn(
            rng=rng, model=model, params=params, batch=batch, train=train
        )
        return loss, metrics

    def train_step_fn(
        state: TrainState, batch: Dict[str, Array], axis_name: str | None
    ):
        rng, step_rng = jax.random.split(state.rng)
        (loss, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(
            state.params, step_rng, batch, True
        )

        if axis_name is not None:
            grads = jax.lax.pmean(grads, axis_name=axis_name)
            metrics = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name=axis_name), metrics)

        updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        if ema_rate > 0.0:
            new_ema_params = jax.tree_util.tree_map(
                lambda e, p: e + (1.0 - ema_rate) * (p - e),
                state.ema_params,
                new_params,
            )
        else:
            new_ema_params = None

        new_state = state.replace(
            step=state.step + 1,
            rng=rng,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
        )
        metrics = dict(metrics)
        metrics["train/loss"] = loss
        return new_state, metrics

    if use_pmap:
        p_train_step = jax.pmap(
            lambda st, b: train_step_fn(st, b, axis_name="batch"),
            axis_name="batch",
        )

        state = replicate(state)

        for step in range(num_train_steps):

            if step in (4400, 4550, 4650):
                jax.profiler.save_device_memory_profile(f"/tmp/mem_{step}.prof")

            batch = next(train_iter)
            gt_images = batch["image"][:num_log_images]
            batch = _shard(batch)
        
            state, metrics = p_train_step(state, batch)
            _ = jax.block_until_ready(metrics["train/loss"])

            if (step + 1) % log_every_steps == 0:
                metrics = unreplicate(metrics)
                print(f"[step {step+1}] loss={float(metrics['train/loss']):.4f}")

                if wandb_mod is not None:
                    log_dict = _sanitize_metrics(metrics)
                    log_dict["lr"] = float(_to_py_scalar(lr_schedule(step)) or lr_schedule(step))
                    log_dict["step"] = step + 1
                    wandb_mod.log(log_dict, step=step + 1)

            if (wandb_mod is not None) and (log_images_every_steps > 0) and (step % log_images_every_steps == 0):
                state_s = unreplicate(state)
                params_for_sampling = state_s.ema_params if state_s.ema_params is not None else state_s.params
                sjd_sample_metrics = None
                if sample_images_jit is not None:
                    sample_rng = jax.random.fold_in(
                        jax.random.PRNGKey(int(cfg.training.seed) + 999), step
                    )
                    out = sample_images_jit(params_for_sampling, sample_rng)
                    if str(cfg.model.name) == "sjd":
                        # SJD sampler returns a structured result.
                        sjd_sample_metrics = out.metrics
                        samples = out.k_filled
                        print("about to sample", step)
                        samples = jax.block_until_ready(samples)
                        print("done sampling", step)
                    else:
                        # MD4 sampler returns the image tokens directly.
                        samples = jax.block_until_ready(out)
                else:
                    samples = None

                _log_images_to_wandb(step, gt_images, samples)

                # Also log sampling diagnostics (jump statistics, etc.) for SJD.
                if (wandb_mod is not None) and (sjd_sample_metrics is not None):
                    wandb_mod.log(_sanitize_metrics(sjd_sample_metrics), step=step)

    else:
        train_step_jit = jax.jit(
            lambda st, b: train_step_fn(st, b, axis_name=None),
        )

        for step in range(num_train_steps):
            batch = next(train_iter)
            gt_images = batch["image"][:num_log_images]
            state, metrics = train_step_jit(state, batch)

            if (step + 1) % log_every_steps == 0:
                print(f"[step {step+1}] loss={float(metrics['train/loss']):.4f}")

                if wandb_mod is not None:
                    log_dict = _sanitize_metrics(metrics)
                    log_dict["lr"] = float(_to_py_scalar(lr_schedule(step)) or lr_schedule(step))
                    log_dict["step"] = step + 1
                    wandb_mod.log(log_dict, step=step + 1)
            
            if (wandb_mod is not None) and (log_images_every_steps > 0) and (step % log_images_every_steps == 0):
                params_for_sampling = state.ema_params if state.ema_params is not None else state.params
                sjd_sample_metrics = None
                if sample_images_jit is not None:
                    sample_rng = jax.random.fold_in(
                        jax.random.PRNGKey(int(cfg.training.seed) + 999), step
                    )
                    out = sample_images_jit(params_for_sampling, sample_rng)
                    if str(cfg.model.name) == "sjd":
                        # SJD sampler returns a structured result.
                        sjd_sample_metrics = out.metrics
                        samples = out.k_filled
                        samples = jax.block_until_ready(samples)
                    else:
                        # MD4 sampler returns the image tokens directly.
                        samples = jax.block_until_ready(out)
                else:
                    samples = None

                _log_images_to_wandb(step, gt_images, samples)

                # Also log sampling diagnostics (jump statistics, etc.) for SJD.
                if (wandb_mod is not None) and (sjd_sample_metrics is not None):
                    wandb_mod.log(_sanitize_metrics(sjd_sample_metrics), step=step)


@hydra.main(version_base=None, config_path="../../../config", config_name="config.yaml")
def main(cfg: DictConfig):
    print("Resolved config:\n", OmegaConf.to_yaml(cfg))
    wandb_mod = _maybe_init_wandb(cfg)
    try:
        main_train_loop(cfg.experiment, wandb_mod=wandb_mod)
    finally:
        if wandb_mod is not None:
            wandb_mod.finish()


if __name__ == "__main__":
    main()
