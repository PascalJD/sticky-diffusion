"""Text8SJDTask composes correctly and decodes via the char map."""
import os

import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from sticky.tasks.factory import build_task
from sticky.tasks.text8_sjd import Text8SJDTask

_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))


def _build_text8_task():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["experiment=text8/sjd_text8_fixed"],
        )
    task = build_task(cfg.experiment)
    GlobalHydra.instance().clear()
    return task


def test_build_task_returns_text8_sjd_task():
    task = _build_text8_task()
    assert isinstance(task, Text8SJDTask)


def test_task_spec_is_char_level():
    task = _build_text8_task()
    assert task.spec.task_type == "text"
    assert task.spec.data_shape == (128,)
    assert task.spec.vocab_size == 27


def test_task_is_fixed_baseline():
    # CANDI-equivalent FIXED: ce objective, fixed hazard (not learned).
    task = _build_text8_task()
    assert str(task.objective) == "ce"
    assert bool(task.learn_log_w) is False
    assert abs(float(task.forward.jump.eta) - 1.0) < 1e-9
    assert getattr(task.forward.jump, "blur_kernel", None) is None


def test_extract_x0_idx_is_int32():
    task = _build_text8_task()
    batch = {"image": np.zeros((2, 128), dtype=np.int64)}
    x0 = task._extract_x0_idx(batch)
    assert x0.dtype == jnp.int32


def test_format_samples_decodes_chars():
    task = _build_text8_task()
    # ids 0,1,2 -> "abc"; 26 -> space.
    rendered = task.format_samples_for_logging(jnp.asarray([[0, 1, 2, 26, 0]]))
    assert rendered == ["abc a"]
