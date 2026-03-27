from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

import sticky.training.loop as loop_mod


def _make_train_cfg(
    *,
    num_train_steps: int,
    platform: str = "single",
    log_every_steps: int = 0,
    log_images_every_steps: int = 0,
    metrics_every_steps: int = 0,
    checkpoint_every_steps: int = 0,
    save_final_metrics: bool = True,
    save_final_checkpoint: bool = True,
):
    return OmegaConf.create(
        {
            "model": {"name": "ddpm"},
            "dataset": {"batch_size": 2, "drop_remainder": True, "data_dir": None},
            "training": {
                "seed": 0,
                "num_train_steps": num_train_steps,
                "num_log_images": 2,
                "sample_timesteps": 5,
                "log_images_every_steps": log_images_every_steps,
                "log_every_steps": log_every_steps,
                "eval_every_steps": 0,
                "timing_warn_seconds": 999.0,
                "ema_rate": 0.0,
                "metrics_every_steps": metrics_every_steps,
                "save_final_metrics": save_final_metrics,
                "checkpoint_every_steps": checkpoint_every_steps,
                "checkpoint_keep": 5,
                "save_final_checkpoint": save_final_checkpoint,
                "best_checkpoint_metric": "eval/fid",
                "best_checkpoint_mode": "min",
                "best_update_on_equal": False,
                "likelihood_eval_every_steps": 0,
                "likelihood_eval_max_batches": -1,
            },
            "runtime": {
                "platform": platform,
                "sync_train_step": False,
                "pmap_prefetch_buffer_size": 2,
            },
        }
    )


def _install_main_loop_fakes(
    monkeypatch,
    tmp_path,
    *,
    num_train_steps: int,
    platform: str = "single",
):
    records = {
        "run_context": [],
        "sample_for_logging": [],
        "log_images": [],
        "eval_calls": [],
        "metrics_events": [],
        "checkpoint_events": [],
        "wandb_logs": [],
        "replicate": [],
        "unreplicate": [],
        "prefetch": [],
    }

    local_devices = jax.local_device_count()
    batch_size = local_devices * 2 if platform == "pmap" else 2
    batch = {
        "image": jnp.zeros((batch_size, 4, 4, 3), dtype=jnp.float32),
        "label": jnp.zeros((batch_size,), dtype=jnp.int32),
    }
    train_batches = [batch for _ in range(num_train_steps)]

    task = SimpleNamespace(
        spec=SimpleNamespace(task_type="image", data_shape=(4, 4, 3), vocab_size=8),
        make_dataloaders=lambda seed: (iter(train_batches), None),
    )
    model = SimpleNamespace()
    state0 = SimpleNamespace(
        step=jnp.asarray(0, dtype=jnp.int32),
        params={"live": 0},
        ema_params={"ema": 0},
    )

    monkeypatch.setattr(loop_mod, "build_task", lambda cfg: task)
    monkeypatch.setattr(
        loop_mod,
        "build_model",
        lambda cfg, data_shape, vocab_size: model,
    )
    monkeypatch.setattr(loop_mod, "init_state", lambda cfg, model, rng: (state0, "tx"))
    monkeypatch.setattr(loop_mod, "make_lr_schedule", lambda cfg: (lambda step: 0.01 * (step + 1)))
    monkeypatch.setattr(
        loop_mod,
        "get_hydra_output_dir",
        lambda: tmp_path / "run",
    )
    monkeypatch.setattr(
        loop_mod,
        "resolve_run_path",
        lambda value, default_name, *, base_dir: Path(base_dir) / str(value or default_name),
    )
    monkeypatch.setattr(
        loop_mod,
        "resolve_from_original_cwd",
        lambda value: None
        if value in (None, "", "null", "None")
        else str(tmp_path / str(value)),
    )
    monkeypatch.setattr(
        loop_mod,
        "write_run_context",
        lambda **kwargs: records["run_context"].append(kwargs),
    )
    monkeypatch.setattr(
        loop_mod,
        "build_sampling_fns",
        lambda **kwargs: (
            lambda params, rng: jnp.ones((2, 4, 4, 3), dtype=jnp.float32),
            lambda params, rng: jnp.ones((2, 4, 4, 3), dtype=jnp.float32),
        ),
    )

    def fake_sample_for_logging(*, cfg, sample_images_jit, params_for_sampling, step):
        del cfg, sample_images_jit
        records["sample_for_logging"].append({"step": step, "params": params_for_sampling})
        return (
            jnp.full((2, 4, 4, 3), float(step), dtype=jnp.float32),
            {"sample/score": jnp.asarray(float(step), dtype=jnp.float32)},
        )

    monkeypatch.setattr(loop_mod, "sample_for_logging", fake_sample_for_logging)
    monkeypatch.setattr(
        loop_mod,
        "log_images_to_wandb",
        lambda **kwargs: records["log_images"].append(kwargs),
    )

    def fake_build_eval_logger(**kwargs):
        del kwargs

        def maybe_log_eval(step_i, params, force_fid: bool = False, force_is: bool = False):
            records["eval_calls"].append(
                {
                    "step": step_i,
                    "params": params,
                    "force_fid": force_fid,
                    "force_is": force_is,
                }
            )
            return {"eval/fid": float(step_i)}

        return maybe_log_eval

    monkeypatch.setattr(loop_mod, "build_eval_logger", fake_build_eval_logger)
    monkeypatch.setattr(loop_mod, "make_train_step_fn", lambda **kwargs: "train_step_fn")
    monkeypatch.setattr(loop_mod, "make_wrapped_eval_step", lambda *args, **kwargs: "eval_step")

    def fake_make_wrapped_train_step(train_step_fn, *, use_pmap, axis_name=None):
        del train_step_fn, axis_name

        def run(state, batch):
            del batch
            step = int(jax.device_get(state.step)) + 1
            next_state = SimpleNamespace(
                step=jnp.asarray(step, dtype=jnp.int32),
                params={"live": step},
                ema_params={"ema": step},
            )
            return next_state, {"train/loss": jnp.asarray(float(step), dtype=jnp.float32)}

        return run

    monkeypatch.setattr(loop_mod, "make_wrapped_train_step", fake_make_wrapped_train_step)

    class RecordingMetricsWriter:
        def __init__(self, root_dir, every_steps):
            self.root_dir = root_dir
            self.every_steps = every_steps

        def should_write(self, step_i):
            return self.every_steps > 0 and (step_i % self.every_steps) == 0

        def write(self, *, step_i, metrics, tag):
            records["metrics_events"].append(
                ("write", int(step_i), str(tag), dict(metrics))
            )

        def write_final(self, *, step_i, metrics):
            records["metrics_events"].append(
                ("final", int(step_i), dict(metrics))
            )

    class RecordingCheckpointWriter:
        def __init__(
            self,
            *,
            root_dir,
            every_steps,
            keep,
            save_final,
            best_metric_key,
            best_mode,
            best_update_on_equal,
        ):
            del root_dir, keep, best_metric_key, best_mode, best_update_on_equal
            self.every_steps = every_steps
            self.save_final = save_final

        def maybe_save_best(self, *, target, step_i, metrics):
            records["checkpoint_events"].append(
                ("best", int(step_i), dict(metrics), target)
            )
            return True

        def maybe_save_periodic(self, *, target, step_i):
            if self.every_steps > 0 and (step_i % self.every_steps) == 0:
                records["checkpoint_events"].append(("periodic", int(step_i), target))

        def save_final_checkpoint(self, *, target, step_i):
            if self.save_final:
                records["checkpoint_events"].append(("final", int(step_i), target))

    monkeypatch.setattr(loop_mod, "MetricsWriter", RecordingMetricsWriter)
    monkeypatch.setattr(loop_mod, "CheckpointWriter", RecordingCheckpointWriter)

    def fake_replicate(value):
        records["replicate"].append(value)
        return value

    def fake_unreplicate(value):
        records["unreplicate"].append(value)
        return value

    def fake_prefetch_to_device(iterator, buffer_size):
        records["prefetch"].append(int(buffer_size))
        return iterator

    monkeypatch.setattr(loop_mod, "replicate", fake_replicate)
    monkeypatch.setattr(loop_mod, "unreplicate", fake_unreplicate)
    monkeypatch.setattr(loop_mod, "prefetch_to_device", fake_prefetch_to_device)

    wandb_mod = SimpleNamespace(
        log=lambda metrics, step=None: records["wandb_logs"].append((step, dict(metrics)))
    )

    return records, task, model, wandb_mod


def test_make_pmap_batch_iterator_prefetch_preserves_shapes():
    per_device_batch = 2
    batch = {
        "image": jnp.zeros(
            (jax.local_device_count() * per_device_batch, 4, 4, 3),
            dtype=jnp.float32,
        ),
        "label": jnp.zeros(
            (jax.local_device_count() * per_device_batch,),
            dtype=jnp.int32,
        ),
    }

    iterator = loop_mod._make_pmap_batch_iterator(
        iter([batch]),
        prefetch_buffer_size=2,
    )
    prefetched = next(iterator)

    assert prefetched["image"].shape == (
        jax.local_device_count(),
        per_device_batch,
        4,
        4,
        3,
    )
    assert prefetched["label"].shape == (
        jax.local_device_count(),
        per_device_batch,
    )


def test_sync_flag_toggles_blocking_behavior(monkeypatch):
    calls = []

    def fake_block_until_ready(x):
        calls.append(x)
        return x

    monkeypatch.setattr(loop_mod.jax, "block_until_ready", fake_block_until_ready)

    metric = jnp.asarray(1.0, dtype=jnp.float32)
    loop_mod._maybe_sync_training_metric(metric, sync=False)
    assert calls == []

    loop_mod._maybe_sync_training_metric(metric, sync=True)
    assert len(calls) == 1


def test_resolve_num_train_steps_prefers_epoch_derived_budget():
    cfg = OmegaConf.create(
        {
            "dataset": {"batch_size": 128, "drop_remainder": True},
            "training": {"num_train_steps": 123, "num_train_epochs": 300},
        }
    )

    class _Task:
        def train_num_examples(self):
            return 1800

    num_steps = loop_mod._resolve_num_train_steps(cfg, _Task())

    assert num_steps == 4200


def test_resolve_num_train_steps_uses_explicit_steps_without_epochs():
    cfg = OmegaConf.create(
        {
            "dataset": {"batch_size": 128, "drop_remainder": True},
            "training": {"num_train_steps": 777},
        }
    )

    num_steps = loop_mod._resolve_num_train_steps(cfg, object())

    assert num_steps == 777


def test_params_for_sudoku_eval_honors_requested_param_source():
    state = SimpleNamespace(params={"w": 1}, ema_params={"w": 2})

    assert loop_mod._params_for_sudoku_eval(
        state,
        eval_cfg=OmegaConf.create({"param_source": "live"}),
    ) is state.params
    assert loop_mod._params_for_sudoku_eval(
        state,
        eval_cfg=OmegaConf.create({"param_source": "ema"}),
    ) is state.ema_params

    no_ema_state = SimpleNamespace(params={"w": 3}, ema_params=None)
    assert loop_mod._params_for_sudoku_eval(
        no_ema_state,
        eval_cfg=OmegaConf.create({"param_source": "ema"}),
    ) is no_ema_state.params


def test_main_train_loop_non_pmap_preserves_logging_eval_and_checkpoint_cadence(
    monkeypatch,
    tmp_path,
):
    records, _, _, wandb_mod = _install_main_loop_fakes(
        monkeypatch,
        tmp_path,
        num_train_steps=3,
        platform="single",
    )
    cfg = _make_train_cfg(
        num_train_steps=3,
        platform="single",
        log_every_steps=2,
        log_images_every_steps=2,
        metrics_every_steps=2,
        checkpoint_every_steps=2,
    )
    eval_cfg = OmegaConf.create(
        {
            "enabled": True,
            "mode": "fid_is",
            "fid_every": 2,
            "is_every": 2,
            "run_at_end": True,
            "fid_num_samples": 8,
            "fid_batch_size": 4,
        }
    )

    loop_mod.main_train_loop(cfg, wandb_mod=wandb_mod, eval_cfg=eval_cfg)

    assert [event[:3] for event in records["metrics_events"][:2]] == [
        ("write", 2, "train"),
        ("write", 2, "eval"),
    ]
    assert records["metrics_events"][-1][0] == "final"
    assert records["metrics_events"][-1][1] == 3
    assert records["metrics_events"][-1][2]["train/loss"] == 3.0
    assert records["metrics_events"][-1][2]["eval/fid"] == 3.0

    assert [call["step"] for call in records["sample_for_logging"]] == [2]
    assert [call["step_i"] for call in records["log_images"]] == [2]
    assert records["eval_calls"] == [
        {"step": 2, "params": {"ema": 2}, "force_fid": False, "force_is": False},
        {"step": 3, "params": {"ema": 3}, "force_fid": True, "force_is": True},
    ]

    assert [event[0:2] for event in records["checkpoint_events"]] == [
        ("best", 2),
        ("periodic", 2),
        ("best", 3),
        ("final", 3),
    ]
    assert [step for step, metrics in records["wandb_logs"] if "train/loss" in metrics] == [2]
    assert [step for step, metrics in records["wandb_logs"] if "sample/score" in metrics] == [2]
    assert len(records["run_context"]) == 1


def test_main_train_loop_final_eval_only_forces_when_last_periodic_eval_missed(
    monkeypatch,
    tmp_path,
):
    records, _, _, _ = _install_main_loop_fakes(
        monkeypatch,
        tmp_path,
        num_train_steps=4,
        platform="single",
    )
    cfg = _make_train_cfg(
        num_train_steps=4,
        platform="single",
        save_final_metrics=False,
        save_final_checkpoint=False,
    )
    eval_cfg = OmegaConf.create(
        {
            "enabled": True,
            "mode": "fid_is",
            "fid_every": 2,
            "is_every": 2,
            "run_at_end": True,
        }
    )

    loop_mod.main_train_loop(cfg, wandb_mod=None, eval_cfg=eval_cfg)

    assert records["eval_calls"] == [
        {"step": 2, "params": {"ema": 2}, "force_fid": False, "force_is": False},
        {"step": 4, "params": {"ema": 4}, "force_fid": False, "force_is": False},
    ]


def test_main_train_loop_pmap_path_uses_replicate_prefetch_and_unreplicate(
    monkeypatch,
    tmp_path,
):
    records, _, _, wandb_mod = _install_main_loop_fakes(
        monkeypatch,
        tmp_path,
        num_train_steps=1,
        platform="pmap",
    )
    cfg = _make_train_cfg(
        num_train_steps=1,
        platform="pmap",
        log_every_steps=1,
        log_images_every_steps=1,
        metrics_every_steps=1,
        checkpoint_every_steps=1,
    )
    eval_cfg = OmegaConf.create({"enabled": False})
    original_make_pmap_batch_iterator = loop_mod._make_pmap_batch_iterator

    def recording_make_pmap_batch_iterator(train_iter, *, prefetch_buffer_size):
        records["prefetch"].append(int(prefetch_buffer_size))
        return original_make_pmap_batch_iterator(
            train_iter,
            prefetch_buffer_size=0,
        )

    monkeypatch.setattr(
        loop_mod,
        "_make_pmap_batch_iterator",
        recording_make_pmap_batch_iterator,
    )

    loop_mod.main_train_loop(cfg, wandb_mod=wandb_mod, eval_cfg=eval_cfg)

    assert len(records["replicate"]) == 1
    assert records["prefetch"] == [2]
    assert len(records["unreplicate"]) >= 3
    assert [event[0:2] for event in records["checkpoint_events"]] == [
        ("periodic", 1),
        ("final", 1),
    ]
    assert [call["step_i"] for call in records["log_images"]] == [1]
