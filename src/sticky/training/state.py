from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax
from flax import struct
from omegaconf import DictConfig

from sticky.core.paths import is_nullish
from sticky.models import factories  # noqa: F401  triggers @register_init
from sticky.models._registry import get_init
from sticky.models.sjd.anchors import anchor_learnable_from_mapping
from sticky.rng import PRNGKey, ensure_prng_key


Array = jnp.ndarray
_LOGGED_PARAM_COUNT_KEYS: set[tuple[str, str, tuple[int, ...]]] = set()


@struct.dataclass
class TrainState:
    step: Array
    rng: Array
    params: Any
    ema_params: Any
    opt_state: optax.OptState


def shard_batch(batch: Dict[str, Array]) -> Dict[str, Array]:
    """(B, ...) -> (n_local_devices, B//n_local_devices, ...)"""
    n = jax.local_device_count()
    out: Dict[str, Array] = {}
    for k, v in batch.items():
        assert v.shape[0] % n == 0, (
            f"Batch dim for {k} must be divisible by local_device_count."
        )
        out[k] = v.reshape((n, v.shape[0] // n) + v.shape[1:])
    return out


def make_lr_schedule(cfg: DictConfig):
    warmup_steps = int(cfg.optim.warmup_steps)
    total_steps = int(cfg.training.num_train_steps)
    base_lr = float(cfg.optim.learning_rate)
    schedule_name = str(cfg.optim.get("lr_schedule", "warmup_cosine_decay"))

    # Optax expects decay_steps to be the overall schedule horizon and requires
    # decay_steps > warmup_steps. Short smoke runs can otherwise fail.
    total_steps = max(1, total_steps)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    decay_steps = max(total_steps, warmup_steps + 1)

    if schedule_name == "warmup_constant":
        if warmup_steps <= 0:
            return optax.constant_schedule(base_lr)
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=base_lr,
                    transition_steps=warmup_steps,
                ),
                optax.constant_schedule(base_lr),
            ],
            boundaries=[warmup_steps],
        )

    if schedule_name == "warmup_cosine_decay":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )

    raise ValueError(f"Unknown optim.lr_schedule={schedule_name!r}")


def _path_keys(path: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(
        getattr(entry, "key", getattr(entry, "name", getattr(entry, "idx", entry)))
        for entry in path
    )


def _sjd_optimizer_labels(params: Any, *, anchors_learnable: bool):
    def _label(path, _leaf):
        keys = _path_keys(path)
        if keys == ("anchors", "table"):
            return "trainable" if anchors_learnable else "frozen"
        if keys == ("anchors", "log_w"):
            return "log_w"
        return "trainable"

    return jax.tree_util.tree_map_with_path(_label, params)


def _params_have_log_w(params: Any) -> bool:
    anchors = params.get("anchors", None) if hasattr(params, "get") else None
    if anchors is None:
        return False
    return "log_w" in anchors


def make_optimizer(cfg: DictConfig, params: Any):
    lr_schedule = make_lr_schedule(cfg)
    adamw_tx = optax.adamw(
        learning_rate=lr_schedule,
        b1=0.9,
        b2=float(cfg.optim.b2),
        weight_decay=float(cfg.optim.weight_decay),
    )
    base_tx: optax.GradientTransformation = adamw_tx
    if str(cfg.model.name) == "sjd":
        anchors_learnable = anchor_learnable_from_mapping(cfg.model, default=True)
        learn_log_w = _params_have_log_w(params)
        if (not anchors_learnable) or learn_log_w:
            log_w_multiplier = float(
                cfg.training.get("log_w_lr_multiplier", 100.0)
            )
            # log_w needs a much larger LR than the classifier (the gradient
            # through the SJD ELBO prefactor is ~O(1e-3) per coordinate).
            # Plain Adam, no weight decay — we don't want to shrink log_w.
            log_w_lr_schedule = lambda step: log_w_multiplier * lr_schedule(step)
            log_w_tx = optax.adam(
                learning_rate=log_w_lr_schedule,
                b1=0.9,
                b2=float(cfg.optim.b2),
            )
            base_tx = optax.multi_transform(
                {
                    "trainable": adamw_tx,
                    "frozen": optax.set_to_zero(),
                    "log_w": log_w_tx,
                },
                _sjd_optimizer_labels(
                    params, anchors_learnable=anchors_learnable
                ),
            )

    grad_clip_norm = float(cfg.optim.get("grad_clip_norm", 0.0))
    if grad_clip_norm > 0.0:
        return optax.chain(
            optax.clip_by_global_norm(grad_clip_norm),
            base_tx,
        )
    return base_tx


def init_state(cfg: DictConfig, model, rng: PRNGKey):
    rng = ensure_prng_key(rng)
    name = str(cfg.model.name)
    # data_shape is kept here for the param-count logging block below.
    data_shape = tuple(getattr(model, "data_shape", tuple(cfg.dataset.data_shape)))

    # Registry dispatch: each family's init fn is registered under cfg.model.name.
    # rng sequencing note: the outer split here differs from the pre-Phase-5
    # per-family splits (discrete baselines used 3-way; SJD used 2-way). This
    # means rng_params will differ from pre-Phase-5 code. This only affects
    # re-init from a pre-Phase-5 checkpoint (step 0 parameters); after step 1
    # the rng divergence washes out via step_rng updates.
    init_fn = get_init(name)
    rng, init_rng = jax.random.split(rng, 2)
    variables = init_fn(model, cfg, init_rng)

    params = variables["params"]
    model_name = str(cfg.model.get("name", "unknown"))
    sequence_backbone = str(cfg.model.get("sequence_backbone", ""))
    if (model_name in {"mdlm", "mdm"}) and (sequence_backbone == "gpt2_like"):
        key = (model_name, sequence_backbone, data_shape)
        if key not in _LOGGED_PARAM_COUNT_KEYS:
            _LOGGED_PARAM_COUNT_KEYS.add(key)
            param_count = int(
                sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params))
            )
            print(
                f"[model] parameter_count model={model_name} "
                f"sequence_backbone={sequence_backbone} total={param_count}",
                flush=True,
            )
    tx = make_optimizer(cfg, params)
    opt_state = tx.init(params)

    ema_rate = float(cfg.training.ema_rate)
    if ema_rate > 0.0:
        # Keep EMA numerically identical to params at init while ensuring the
        # buffers are distinct, so TrainState donation can safely donate the
        # whole state tuple.
        ema_params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), params)
    else:
        ema_params = None

    state = TrainState(
        step=jnp.array(0, dtype=jnp.int32),
        rng=rng,
        params=params,
        ema_params=ema_params,
        opt_state=opt_state,
    )

    ckpt_path_cfg = cfg.training.get("init_from_checkpoint", None)
    if not is_nullish(ckpt_path_cfg, nullish=(None, "", "null")):
        from pathlib import Path

        from sticky.training.checkpoint_io import restore_state_from_checkpoint

        ckpt_dir = Path(str(ckpt_path_cfg))
        ckpt_step_cfg = cfg.training.get("init_checkpoint_step", None)
        ckpt_step = None if is_nullish(ckpt_step_cfg, nullish=(None, "", "null")) else int(ckpt_step_cfg)
        load_ema = bool(cfg.training.get("init_load_ema", True))
        reset_optimizer = bool(cfg.training.get("init_reset_optimizer", True))
        reset_step = bool(cfg.training.get("init_reset_step", True))

        restored, restored_step, selected_path = restore_state_from_checkpoint(
            state, ckpt_dir, ckpt_step,
        )
        new_params = restored.params
        new_ema = restored.ema_params if (load_ema and restored.ema_params is not None) else ema_params
        new_opt_state = opt_state if reset_optimizer else restored.opt_state
        new_step = jnp.array(0, dtype=jnp.int32) if reset_step else restored.step

        state = TrainState(
            step=new_step,
            rng=rng,
            params=new_params,
            ema_params=new_ema,
            opt_state=new_opt_state,
        )
        print(
            f"[init_from_checkpoint] loaded {selected_path} step={restored_step} "
            f"load_ema={load_ema} reset_optimizer={reset_optimizer} reset_step={reset_step}",
            flush=True,
        )

    return state, tx
