from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

import sticky.models.sjd.anchors as anchor_mod
import sticky.models.sjd.sampler as sampler_mod
import sticky.models.sjd.sampling as sjd_sampling_mod
from sticky.training.sampling import build_sampling_fns


def test_sjd_sampling_uses_sample_timesteps(monkeypatch):
    recorded: dict[str, int] = {}

    class DummySamplerConfig:
        def __init__(self, **kwargs):
            recorded["n_steps"] = int(kwargs["n_steps"])
            self.n_steps = int(kwargs["n_steps"])

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
                "score_scale": 1.0,
                "logit_temperature": 1.0,
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
