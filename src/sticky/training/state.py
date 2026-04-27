from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax
from flax import struct
from omegaconf import DictConfig

from sticky.core.paths import is_nullish
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


def _sjd_optimizer_labels(params: Any):
    def _label(path, _leaf):
        return "frozen" if _path_keys(path) == ("anchors", "table") else "trainable"

    return jax.tree_util.tree_map_with_path(_label, params)


def make_optimizer(cfg: DictConfig, params: Any):
    lr_schedule = make_lr_schedule(cfg)
    adamw_tx = optax.adamw(
        learning_rate=lr_schedule,
        b1=0.9,
        b2=float(cfg.optim.b2),
        weight_decay=float(cfg.optim.weight_decay),
    )
    base_tx: optax.GradientTransformation = adamw_tx
    if (
        str(cfg.model.name) == "sjd"
        and not anchor_learnable_from_mapping(cfg.model, default=True)
    ):
        base_tx = optax.multi_transform(
            {
                "trainable": adamw_tx,
                "frozen": optax.set_to_zero(),
            },
            _sjd_optimizer_labels(params),
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
    batch_size = (
        int(cfg.dataset.per_device_batch_size)
        if int(cfg.dataset.per_device_batch_size) > 0
        else 1
    )
    data_shape = tuple(getattr(model, "data_shape", tuple(cfg.dataset.data_shape)))

    if name in ("md4", "mdlm", "mdm", "d3pm", "candi", "cadd", "bitdiff", "ddpm"):
        rng, rng_params, rng_sample = jax.random.split(rng, 3)
        dummy_x = jnp.zeros((batch_size,) + data_shape, dtype=jnp.int32)
        dummy_cond = (
            jnp.zeros((batch_size,), dtype=jnp.int32)
            if int(cfg.model.classes) > 0
            else None
        )
        if name == "mdm":
            dummy_t = jnp.zeros((batch_size,), dtype=jnp.float32)
            variables = model.init(
                {"params": rng_params, "sample": rng_sample},
                dummy_x,
                dummy_t,
                cond=dummy_cond,
                train=False,
            )
        else:
            variables = model.init(
                {"params": rng_params, "sample": rng_sample},
                dummy_x,
                cond=dummy_cond,
                train=False,
            )

    elif name == "sjd":
        rng, rng_params = jax.random.split(rng, 2)

        anchor_dim = int(model.anchor_config.anchor_dim)
        dummy_z = jnp.zeros(
            (batch_size,) + tuple(cfg.dataset.data_shape) + (anchor_dim,),
            dtype=jnp.float32,
        )
        dummy_t = jnp.zeros((batch_size,), dtype=jnp.float32)
        dummy_ids = jnp.zeros(tuple(dummy_z.shape[:-1]), dtype=jnp.int32)
        if bool(getattr(model, "enable_joint_input", False)):
            # Sudoku slack-augmented SJD: init the joint-input projection too.
            dummy_slack = jnp.zeros((batch_size, 27, anchor_dim), dtype=jnp.float32)
            variables = model.init(
                {"params": rng_params},
                dummy_z,
                dummy_t,
                anchor_token_ids=dummy_ids,
                slack_y_t=dummy_slack,
                train=False,
            )
        else:
            variables = model.init(
                {"params": rng_params},
                dummy_z,
                dummy_t,
                anchor_token_ids=dummy_ids,
                train=False,
            )

    else:
        raise ValueError(f"Unknown model.name={name!r} for init")

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
