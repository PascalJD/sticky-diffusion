from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

import sticky.models.sjd.anchors as anchor_mod
import sticky.models.bitdiff.sampling as bitdiff_sampling_mod
import sticky.models.candi.sampling as candi_sampling_mod
import sticky.models.d3pm.sampling as d3pm_sampling_mod
import sticky.models.ddpm.sampling as ddpm_sampling_mod
import sticky.models.mdlm.sampling as mdlm_sampling_mod
import sticky.models.sjd.sampler as sampler_mod
import sticky.models.sjd.sampling as sjd_sampling_mod
from sticky.training.sampling import build_sampling_fns


def test_sjd_sampling_uses_sample_timesteps(monkeypatch):
    recorded: dict[str, int | str] = {}

    class DummySamplerConfig:
        def __init__(self, **kwargs):
            recorded["n_steps"] = int(kwargs["n_steps"])
            self.n_steps = int(kwargs["n_steps"])
            recorded["sampling_grid"] = str(kwargs["sampling_grid"])
            self.sampling_grid = str(kwargs["sampling_grid"])
            recorded["categorical_sampling_policy"] = str(kwargs["categorical_sampling_policy"])
            self.categorical_sampling_policy = str(kwargs["categorical_sampling_policy"])

    class DummyAnchorTable:
        def __init__(self, table_float):
            self.table_float = table_float

    class DummyModel:
        def apply(self, variables, method=None):
            del variables, method
            return jnp.zeros((256, 4), dtype=jnp.float32)

        def anchor_table(self):
            raise NotImplementedError

    class DummyBeta:
        T = 1.0

        def __call__(self, t):
            return jnp.ones_like(t)

    def fake_instantiate(cfg, **kwargs):
        del kwargs
        target = str(cfg.get("_target_", ""))
        if "beta" in target:
            return DummyBeta()
        return SimpleNamespace()

    def fake_simple_generate(*, rng, params, model, anchors, beta, hazard, jump, cfg, batch_size, shape):
        del rng, params, model, anchors, beta, hazard, jump, shape
        recorded["generate_n_steps"] = int(cfg.n_steps)
        return jnp.zeros((batch_size, 32, 32, 3), dtype=jnp.float32)

    monkeypatch.setattr(sampler_mod, "SamplerConfig", DummySamplerConfig)
    monkeypatch.setattr(anchor_mod, "AnchorTable", DummyAnchorTable)
    monkeypatch.setattr(sjd_sampling_mod, "simple_generate", fake_simple_generate)
    monkeypatch.setattr("hydra.utils.instantiate", fake_instantiate)

    cfg = OmegaConf.create(
        {
            "model": {"name": "sjd"},
            "forward": {
                "beta": {"_target_": "tests.beta"},
                "hazard": {"_target_": "tests.hazard"},
                "jump": {"_target_": "tests.jump"},
            },
            "sampler": {
                "n_steps": 999,
                "T": 1.0,
                "sampling_grid": "cosine",
                "score_scale": 1.0,
                "logit_temperature": 1.0,
                "categorical_sampling_policy": "exact",
            },
        }
    )
    task = SimpleNamespace(
        spec=SimpleNamespace(
            data_shape=(32, 32, 3),
            vocab_size=256,
        )
    )
    model = DummyModel()

    _, sample_images_fid_jit = build_sampling_fns(
        cfg=cfg,
        task=task,
        model=model,
        num_log_images=4,
        sample_timesteps=37,
        fid_every=1,
        fid_batch_size=4,
    )

    assert sample_images_fid_jit is not None
    out = sample_images_fid_jit({}, jax.random.PRNGKey(0))
    jax.block_until_ready(out)

    assert recorded["n_steps"] == 37
    assert recorded["generate_n_steps"] == 37
    assert recorded["sampling_grid"] == "cosine"
    assert recorded["categorical_sampling_policy"] == "exact"


def test_sjd_sampling_time_grid_supports_cosine():
    uniform = sampler_mod.make_sampling_time_grid(
        T=1.0,
        n_steps=4,
        sampling_grid="uniform",
    )
    cosine = sampler_mod.make_sampling_time_grid(
        T=1.0,
        n_steps=4,
        sampling_grid="cosine",
    )

    assert uniform.shape == (5,)
    assert cosine.shape == (5,)

    assert jnp.allclose(uniform, jnp.asarray([1.0, 0.75, 0.5, 0.25, 0.0], dtype=jnp.float32))
    assert jnp.allclose(
        cosine[jnp.asarray([0, -1])],
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
    )
    assert bool(jnp.all(cosine[:-1] >= cosine[1:]))
    assert float(cosine[1]) > float(uniform[1])


def test_ddpm_sampling_uses_sample_timesteps(monkeypatch):
    recorded: dict[str, int] = {}

    def fake_simple_generate(
        rng,
        train_state,
        *,
        model,
        batch_size,
        conditioning=None,
        timesteps=None,
        use_ema=True,
    ):
        del rng, train_state, model, conditioning, use_ema
        recorded["batch_size"] = int(batch_size)
        recorded["timesteps"] = int(timesteps)
        return jnp.zeros((batch_size, 32, 32, 3), dtype=jnp.float32)

    monkeypatch.setattr(ddpm_sampling_mod, "simple_generate", fake_simple_generate)

    cfg = OmegaConf.create({"model": {"name": "ddpm"}})
    task = SimpleNamespace(
        spec=SimpleNamespace(
            data_shape=(32, 32, 3),
            vocab_size=256,
        )
    )
    model = SimpleNamespace(timesteps=17)

    _, sample_images_fid_jit = build_sampling_fns(
        cfg=cfg,
        task=task,
        model=model,
        num_log_images=4,
        sample_timesteps=17,
        fid_every=1,
        fid_batch_size=4,
    )

    assert sample_images_fid_jit is not None
    out = sample_images_fid_jit({}, jax.random.PRNGKey(0))
    jax.block_until_ready(out)

    assert recorded["batch_size"] == 4
    assert recorded["timesteps"] == 17


def test_mdlm_sampling_uses_sample_timesteps(monkeypatch):
    recorded: dict[str, int] = {}

    def fake_simple_generate(
        rng,
        train_state,
        *,
        model,
        batch_size,
        conditioning=None,
        timesteps=None,
        use_ema=True,
    ):
        del rng, train_state, model, conditioning, use_ema
        recorded["batch_size"] = int(batch_size)
        recorded["timesteps"] = int(timesteps)
        return jnp.zeros((batch_size, 32, 32, 3), dtype=jnp.int32)

    monkeypatch.setattr(mdlm_sampling_mod, "simple_generate", fake_simple_generate)

    cfg = OmegaConf.create({"model": {"name": "mdlm"}})
    task = SimpleNamespace(
        spec=SimpleNamespace(
            data_shape=(32, 32, 3),
            vocab_size=256,
        )
    )
    model = SimpleNamespace(timesteps=23)

    _, sample_images_fid_jit = build_sampling_fns(
        cfg=cfg,
        task=task,
        model=model,
        num_log_images=4,
        sample_timesteps=23,
        fid_every=1,
        fid_batch_size=4,
    )

    assert sample_images_fid_jit is not None
    out = sample_images_fid_jit({}, jax.random.PRNGKey(0))
    jax.block_until_ready(out)

    assert recorded["batch_size"] == 4
    assert recorded["timesteps"] == 23


def test_d3pm_sampling_uses_sample_timesteps(monkeypatch):
    recorded: dict[str, int] = {}

    def fake_simple_generate(
        rng,
        train_state,
        *,
        model,
        batch_size,
        conditioning=None,
        timesteps=None,
        use_ema=True,
    ):
        del rng, train_state, model, conditioning, use_ema
        recorded["batch_size"] = int(batch_size)
        recorded["timesteps"] = int(timesteps)
        return jnp.zeros((batch_size, 32, 32, 3), dtype=jnp.int32)

    monkeypatch.setattr(d3pm_sampling_mod, "simple_generate", fake_simple_generate)

    cfg = OmegaConf.create({"model": {"name": "d3pm"}})
    task = SimpleNamespace(
        spec=SimpleNamespace(
            data_shape=(32, 32, 3),
            vocab_size=256,
        )
    )
    model = SimpleNamespace(timesteps=19)

    _, sample_images_fid_jit = build_sampling_fns(
        cfg=cfg,
        task=task,
        model=model,
        num_log_images=4,
        sample_timesteps=19,
        fid_every=1,
        fid_batch_size=4,
    )

    assert sample_images_fid_jit is not None
    out = sample_images_fid_jit({}, jax.random.PRNGKey(0))
    jax.block_until_ready(out)

    assert recorded["batch_size"] == 4
    assert recorded["timesteps"] == 19


def test_bitdiff_sampling_uses_sample_timesteps(monkeypatch):
    recorded: dict[str, int] = {}

    def fake_simple_generate(
        rng,
        train_state,
        *,
        model,
        batch_size,
        conditioning=None,
        timesteps=None,
        use_ema=True,
    ):
        del rng, train_state, model, conditioning, use_ema
        recorded["batch_size"] = int(batch_size)
        recorded["timesteps"] = int(timesteps)
        return jnp.zeros((batch_size, 32, 32, 3), dtype=jnp.int32)

    monkeypatch.setattr(bitdiff_sampling_mod, "simple_generate", fake_simple_generate)

    cfg = OmegaConf.create({"model": {"name": "bitdiff"}})
    task = SimpleNamespace(
        spec=SimpleNamespace(
            data_shape=(32, 32, 3),
            vocab_size=256,
        )
    )
    model = SimpleNamespace(timesteps=21)

    _, sample_images_fid_jit = build_sampling_fns(
        cfg=cfg,
        task=task,
        model=model,
        num_log_images=4,
        sample_timesteps=21,
        fid_every=1,
        fid_batch_size=4,
    )

    assert sample_images_fid_jit is not None
    out = sample_images_fid_jit({}, jax.random.PRNGKey(0))
    jax.block_until_ready(out)

    assert recorded["batch_size"] == 4
    assert recorded["timesteps"] == 21


def test_candi_sampling_uses_sample_timesteps(monkeypatch):
    recorded: dict[str, int] = {}

    def fake_simple_generate(
        rng,
        train_state,
        *,
        model,
        batch_size,
        conditioning=None,
        timesteps=None,
        use_ema=True,
    ):
        del rng, train_state, model, conditioning, use_ema
        recorded["batch_size"] = int(batch_size)
        recorded["timesteps"] = int(timesteps)
        return jnp.zeros((batch_size, 32, 32, 3), dtype=jnp.int32)

    monkeypatch.setattr(candi_sampling_mod, "simple_generate", fake_simple_generate)

    cfg = OmegaConf.create({"model": {"name": "candi"}})
    task = SimpleNamespace(
        spec=SimpleNamespace(
            data_shape=(32, 32, 3),
            vocab_size=256,
        )
    )
    model = SimpleNamespace(timesteps=21)

    _, sample_images_fid_jit = build_sampling_fns(
        cfg=cfg,
        task=task,
        model=model,
        num_log_images=4,
        sample_timesteps=21,
        fid_every=1,
        fid_batch_size=4,
    )

    assert sample_images_fid_jit is not None
    out = sample_images_fid_jit({}, jax.random.PRNGKey(0))
    jax.block_until_ready(out)

    assert recorded["batch_size"] == 4
    assert recorded["timesteps"] == 21


def test_ddpm_sampling_rejects_mismatched_timesteps():
    cfg = OmegaConf.create({"model": {"name": "ddpm"}})
    task = SimpleNamespace(
        spec=SimpleNamespace(
            data_shape=(32, 32, 3),
            vocab_size=256,
        )
    )
    model = SimpleNamespace(timesteps=32)

    with pytest.raises(ValueError, match="sample_timesteps == model.timesteps"):
        build_sampling_fns(
            cfg=cfg,
            task=task,
            model=model,
            num_log_images=4,
            sample_timesteps=16,
            fid_every=1,
            fid_batch_size=4,
        )


def test_d3pm_sampling_rejects_mismatched_timesteps():
    cfg = OmegaConf.create({"model": {"name": "d3pm"}})
    task = SimpleNamespace(
        spec=SimpleNamespace(
            data_shape=(32, 32, 3),
            vocab_size=256,
        )
    )
    model = SimpleNamespace(timesteps=24)

    with pytest.raises(ValueError, match="sample_timesteps == model.timesteps"):
        build_sampling_fns(
            cfg=cfg,
            task=task,
            model=model,
            num_log_images=4,
            sample_timesteps=12,
            fid_every=1,
            fid_batch_size=4,
        )


def test_d3pm_sampling_rejects_non_uniform_grid():
    cfg = OmegaConf.create({"model": {"name": "d3pm"}})
    task = SimpleNamespace(
        spec=SimpleNamespace(
            data_shape=(32, 32, 3),
            vocab_size=256,
        )
    )
    model = SimpleNamespace(timesteps=24, sampling_grid="cosine")

    with pytest.raises(ValueError, match="sampling_grid='uniform'"):
        build_sampling_fns(
            cfg=cfg,
            task=task,
            model=model,
            num_log_images=4,
            sample_timesteps=24,
            fid_every=1,
            fid_batch_size=4,
        )
