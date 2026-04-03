from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

import sticky.models.sjd.anchors as anchor_mod
import sticky.models.baselines.bitdiff.sampling as bitdiff_sampling_mod
import sticky.models.baselines.cadd.sampling as cadd_sampling_mod
import sticky.models.baselines.candi.sampling as candi_sampling_mod
import sticky.models.baselines.d3pm.sampling as d3pm_sampling_mod
import sticky.models.baselines.ddpm.sampling as ddpm_sampling_mod
import sticky.models.baselines.md4.sampling as md4_sampling_mod
import sticky.models.baselines.mdlm.sampling as mdlm_sampling_mod
import sticky.models.sjd.sampler as sampler_mod
import sticky.models.sjd.sampling as sjd_sampling_mod
from sticky.training.sampling import build_sampling_fns


def test_sjd_sampling_wrapper_preserves_generation_contract(monkeypatch):
    recorded: dict[str, object] = {"calls": []}

    class DummySamplerConfig:
        def __init__(self, **kwargs):
            recorded["n_steps"] = int(kwargs["n_steps"])
            self.n_steps = int(kwargs["n_steps"])
            recorded["sampling_grid"] = str(kwargs["sampling_grid"])
            self.sampling_grid = str(kwargs["sampling_grid"])
            recorded["categorical_sampling_policy"] = str(kwargs["categorical_sampling_policy"])
            self.categorical_sampling_policy = str(kwargs["categorical_sampling_policy"])
            recorded["pc_enabled"] = bool(kwargs["pc_enabled"])
            recorded["corrector_substeps"] = int(kwargs["corrector_substeps"])
            recorded["corrector_step_scale"] = float(kwargs["corrector_step_scale"])

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
        del rng, params, model, anchors, beta, hazard, jump
        recorded["calls"].append(
            {
                "batch_size": int(batch_size),
                "shape": tuple(shape),
                "n_steps": int(cfg.n_steps),
            }
        )
        return sampler_mod.ReverseSampleResult(
            k=-jnp.ones((batch_size,) + tuple(shape), dtype=jnp.int32),
            k_filled=jnp.full((batch_size,) + tuple(shape), 7, dtype=jnp.int32),
            committed=jnp.ones((batch_size,) + tuple(shape), dtype=jnp.bool_),
            metrics={"jump_count": jnp.asarray(float(batch_size), dtype=jnp.float32)},
        )

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
                "pc_enabled": True,
                "corrector_substeps": 2,
                "corrector_step_scale": 0.1,
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

    sample_images_jit, sample_images_fid_jit = build_sampling_fns(
        cfg=cfg,
        task=task,
        model=model,
        num_log_images=4,
        sample_timesteps=37,
        fid_every=1,
        fid_batch_size=6,
    )

    assert sample_images_jit is not None
    assert sample_images_fid_jit is not None
    out = sample_images_jit({}, jax.random.PRNGKey(0))
    out_fid = sample_images_fid_jit({}, jax.random.PRNGKey(1))
    jax.block_until_ready(out.k_filled)
    jax.block_until_ready(out_fid.k_filled)

    assert recorded["n_steps"] == 37
    assert recorded["sampling_grid"] == "cosine"
    assert recorded["categorical_sampling_policy"] == "exact"
    assert recorded["pc_enabled"] is True
    assert recorded["corrector_substeps"] == 2
    assert recorded["corrector_step_scale"] == 0.1
    assert recorded["calls"] == [
        {"batch_size": 4, "shape": (32, 32, 3), "n_steps": 37},
        {"batch_size": 6, "shape": (32, 32, 3), "n_steps": 37},
    ]
    assert out.k_filled.shape == (4, 32, 32, 3)
    assert out_fid.k_filled.shape == (6, 32, 32, 3)
    assert float(out.metrics["jump_count"]) == 4.0
    assert float(out_fid.metrics["jump_count"]) == 6.0


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


@pytest.mark.parametrize(
    ("model_name", "sampling_module", "timesteps"),
    [
        pytest.param("md4", md4_sampling_mod, 13, id="md4"),
        pytest.param("mdlm", mdlm_sampling_mod, 23, id="mdlm"),
        pytest.param("d3pm", d3pm_sampling_mod, 19, id="d3pm"),
        pytest.param("cadd", cadd_sampling_mod, 15, id="cadd"),
        pytest.param("candi", candi_sampling_mod, 21, id="candi"),
        pytest.param("ddpm", ddpm_sampling_mod, 17, id="ddpm"),
        pytest.param("bitdiff", bitdiff_sampling_mod, 25, id="bitdiff"),
    ],
)
def test_non_sjd_sampling_wrappers_preserve_generation_contract(
    monkeypatch,
    model_name,
    sampling_module,
    timesteps,
):
    calls: list[dict[str, object]] = []

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
        del rng, model
        calls.append(
            {
                "batch_size": int(batch_size),
                "timesteps": int(timesteps),
                "conditioning_is_none": conditioning is None,
                "use_ema": bool(use_ema),
                "train_state_keys": tuple(sorted(train_state.keys())),
                "ema_params_is_none": train_state["ema_params"] is None,
            }
        )
        return jnp.full((batch_size, 2, 3), int(batch_size), dtype=jnp.int32)

    monkeypatch.setattr(sampling_module, "simple_generate", fake_simple_generate)

    cfg = OmegaConf.create({"model": {"name": model_name}})
    task = SimpleNamespace(
        spec=SimpleNamespace(
            data_shape=(32, 32, 3),
            vocab_size=256,
        )
    )
    model = SimpleNamespace(timesteps=timesteps)
    params = {"weights": jnp.asarray(1.0, dtype=jnp.float32)}

    sample_images_jit, sample_images_fid_jit = build_sampling_fns(
        cfg=cfg,
        task=task,
        model=model,
        num_log_images=4,
        sample_timesteps=timesteps,
        fid_every=1,
        fid_batch_size=6,
    )

    assert sample_images_jit is not None
    assert sample_images_fid_jit is not None

    out = sample_images_jit(params, jax.random.PRNGKey(0))
    out_fid = sample_images_fid_jit(params, jax.random.PRNGKey(1))
    jax.block_until_ready(out)
    jax.block_until_ready(out_fid)

    assert out.shape == (4, 2, 3)
    assert out_fid.shape == (6, 2, 3)
    assert calls == [
        {
            "batch_size": 4,
            "timesteps": timesteps,
            "conditioning_is_none": True,
            "use_ema": False,
            "train_state_keys": ("ema_params", "params"),
            "ema_params_is_none": True,
        },
        {
            "batch_size": 6,
            "timesteps": timesteps,
            "conditioning_is_none": True,
            "use_ema": False,
            "train_state_keys": ("ema_params", "params"),
            "ema_params_is_none": True,
        },
    ]


def test_bitdiff_sampling_rejects_ddpm_without_unit_eta():
    cfg = OmegaConf.create({"model": {"name": "bitdiff"}})
    task = SimpleNamespace(
        spec=SimpleNamespace(
            data_shape=(32, 32, 3),
            vocab_size=256,
        )
    )
    model = SimpleNamespace(timesteps=21, sampler="ddpm", stochasticity=0.4)

    with pytest.raises(ValueError, match="fixed eta=1.0"):
        build_sampling_fns(
            cfg=cfg,
            task=task,
            model=model,
            num_log_images=4,
            sample_timesteps=21,
            fid_every=1,
            fid_batch_size=4,
        )


def test_candi_sampling_rejects_hybrid_exact_for_embed_mode():
    cfg = OmegaConf.create({"model": {"name": "candi"}})
    task = SimpleNamespace(
        spec=SimpleNamespace(
            data_shape=(32, 32, 3),
            vocab_size=256,
        )
    )
    model = SimpleNamespace(
        timesteps=21,
        sampler="hybrid_exact",
        representation="embed",
    )

    with pytest.raises(ValueError, match="reserved for representation='onehot'"):
        build_sampling_fns(
            cfg=cfg,
            task=task,
            model=model,
            num_log_images=4,
            sample_timesteps=21,
            fid_every=1,
            fid_batch_size=4,
        )


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
