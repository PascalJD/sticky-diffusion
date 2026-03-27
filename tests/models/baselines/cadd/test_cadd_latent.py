from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from sticky.models.baselines.cadd.cadd_model import CADD


class _ProbeCADD(CADD):
    def predict_logits(self, z_tilde, t, *, cond=None, train: bool = False):
        del t, cond, train
        return z_tilde[..., : self.vocab_size]


def _init_tiny_cadd(*, K: int = 1, params_seed: int = 0, sample_seed: int = 1):
    model = CADD(
        data_shape=(2,),
        vocab_size=8,
        feature_dim=4,
        num_heads=1,
        n_layers=1,
        ch_mult=(1,),
        dropout_rate=0.0,
        use_attn_dropout=False,
        sequence_backbone="transformer",
        image_backbone="auto",
        discrete_schedule_type="linear",
        continuous_schedule_type="linear",
        continuous_latent_type="gaussian",
        schedule_eps=0.0,
        K=K,
    )
    x = jnp.asarray([[0, 1]], dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(params_seed), "sample": jax.random.PRNGKey(sample_seed)},
        x,
        cond=None,
        train=False,
    )
    return model, variables


def _init_probe_cadd(*, K: int = 1, params_seed: int = 0, sample_seed: int = 1):
    model = _ProbeCADD(
        data_shape=(2,),
        vocab_size=4,
        feature_dim=4,
        num_heads=1,
        n_layers=1,
        ch_mult=(1,),
        dropout_rate=0.0,
        use_attn_dropout=False,
        sequence_backbone="transformer",
        image_backbone="auto",
        discrete_schedule_type="linear",
        continuous_schedule_type="linear",
        continuous_latent_type="gaussian",
        schedule_eps=0.0,
        K=K,
    )
    x = jnp.asarray([[0, 1]], dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(params_seed), "sample": jax.random.PRNGKey(sample_seed)},
        x,
        cond=None,
        train=False,
    )
    return model, variables


@pytest.mark.parametrize("latent_type", ["gaussian", "flow_matching"])
def test_masked_latent_marginal_uses_selected_process(latent_type: str):
    model = CADD(
        data_shape=(2,),
        vocab_size=8,
        feature_dim=4,
        num_heads=1,
        n_layers=1,
        ch_mult=(1,),
        dropout_rate=0.0,
        use_attn_dropout=False,
        sequence_backbone="transformer",
        image_backbone="auto",
        discrete_schedule_type="linear",
        continuous_schedule_type="linear",
        continuous_latent_type=latent_type,
        schedule_eps=0.0,
    )

    x = jnp.asarray([[0, 1]], dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "sample": jax.random.PRNGKey(1)},
        x,
        cond=None,
        train=False,
    )

    z0 = model.apply(
        variables,
        x,
        method=lambda mdl, ids: mdl.token_embed(ids),
    )
    t = jnp.asarray([0.75], dtype=jnp.float32)
    sample_rng = jax.random.PRNGKey(7)
    latent = model.apply(
        variables,
        rng=sample_rng,
        z0=z0,
        t=t,
        method=model._sample_masked_continuous_latent,
    )

    gbar = 1.0 - t
    noise = jax.random.normal(sample_rng, z0.shape)
    if latent_type == "gaussian":
        expected = jnp.sqrt(gbar)[:, None, None] * z0 + jnp.sqrt(1.0 - gbar)[:, None, None] * noise
    else:
        expected = gbar[:, None, None] * z0 + (1.0 - gbar)[:, None, None] * noise

    np.testing.assert_allclose(
        np.asarray(latent),
        np.asarray(expected),
        atol=1e-6,
        rtol=1e-6,
    )


def test_gaussian_reverse_reaches_z0_without_noise_at_final_step():
    model = CADD(
        data_shape=(2,),
        vocab_size=8,
        feature_dim=4,
        num_heads=1,
        n_layers=1,
        ch_mult=(1,),
        dropout_rate=0.0,
        use_attn_dropout=False,
        sequence_backbone="transformer",
        image_backbone="auto",
        discrete_schedule_type="linear",
        continuous_schedule_type="linear",
        continuous_latent_type="gaussian",
        schedule_eps=0.0,
    )

    x = jnp.asarray([[0, 1]], dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(2), "sample": jax.random.PRNGKey(3)},
        x,
        cond=None,
        train=False,
    )

    z_t = jnp.arange(8, dtype=jnp.float32).reshape(1, 2, 4)
    z0_hat = -jnp.ones_like(z_t)
    z_s = model.apply(
        variables,
        rng=jax.random.PRNGKey(5),
        z_t=z_t,
        z0_hat=z0_hat,
        s=jnp.asarray(0.0, dtype=jnp.float32),
        t=jnp.asarray(0.5, dtype=jnp.float32),
        method=model._reverse_masked_continuous_latent,
    )

    np.testing.assert_allclose(
        np.asarray(z_s),
        np.asarray(z0_hat),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sampling_probs_is_permutation_invariant_across_k_axis():
    model, variables = _init_probe_cadd(K=3, params_seed=11, sample_seed=12)
    x_t = jnp.full((1, 2), model.mask_token_id, dtype=jnp.int32)
    z_samples = jnp.stack(
        [
            jnp.zeros((1, 2, 4), dtype=jnp.float32),
            5.0 * jnp.ones((1, 2, 4), dtype=jnp.float32),
            -3.0 * jnp.ones((1, 2, 4), dtype=jnp.float32),
        ],
        axis=0,
    )
    t = jnp.asarray(0.5, dtype=jnp.float32)

    probs = model.apply(
        variables,
        x_t=x_t,
        z_t=z_samples,
        t=t,
        cond_emb=None,
        method=model._sampling_probs,
    )
    permuted_probs = model.apply(
        variables,
        x_t=x_t,
        z_t=z_samples[jnp.asarray([2, 0, 1])],
        t=t,
        cond_emb=None,
        method=model._sampling_probs,
    )

    np.testing.assert_allclose(
        np.asarray(probs),
        np.asarray(permuted_probs),
        atol=1e-6,
        rtol=1e-6,
    )


def test_repeating_same_latent_k_times_matches_k1_sampling_probs():
    model_single, variables_single = _init_probe_cadd(K=1, params_seed=13, sample_seed=14)
    model_multi, variables_multi = _init_probe_cadd(K=3, params_seed=13, sample_seed=14)
    x_t = jnp.full((1, 2), model_single.mask_token_id, dtype=jnp.int32)
    z_single = 2.0 * jnp.ones((1, 2, 4), dtype=jnp.float32)
    z_repeated = jnp.repeat(z_single[None, ...], repeats=3, axis=0)
    t = jnp.asarray(0.5, dtype=jnp.float32)

    probs_single = model_single.apply(
        variables_single,
        x_t=x_t,
        z_t=z_single,
        t=t,
        cond_emb=None,
        method=model_single._sampling_probs,
    )
    probs_multi = model_multi.apply(
        variables_multi,
        x_t=x_t,
        z_t=z_repeated,
        t=t,
        cond_emb=None,
        method=model_multi._sampling_probs,
    )

    np.testing.assert_allclose(
        np.asarray(probs_single),
        np.asarray(probs_multi),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sampling_probs_depends_on_all_k_latents():
    model, variables = _init_probe_cadd(K=3, params_seed=15, sample_seed=16)
    x_t = jnp.full((1, 2), model.mask_token_id, dtype=jnp.int32)
    z_first = jnp.zeros((1, 2, 4), dtype=jnp.float32)
    repeated = jnp.repeat(z_first[None, ...], repeats=3, axis=0)
    mixed = jnp.stack(
        [
            z_first,
            jnp.asarray(
                [[[4.0, -4.0, 0.0, 0.0], [0.0, 4.0, -4.0, 0.0]]],
                dtype=jnp.float32,
            ),
            jnp.asarray(
                [[[-2.0, 1.0, 0.5, 0.0], [1.5, -1.5, 0.25, 0.0]]],
                dtype=jnp.float32,
            ),
        ],
        axis=0,
    )
    t = jnp.asarray(0.5, dtype=jnp.float32)

    probs_repeated = model.apply(
        variables,
        x_t=x_t,
        z_t=repeated,
        t=t,
        cond_emb=None,
        method=model._sampling_probs,
    )
    probs_mixed = model.apply(
        variables,
        x_t=x_t,
        z_t=mixed,
        t=t,
        cond_emb=None,
        method=model._sampling_probs,
    )

    assert not np.allclose(np.asarray(probs_repeated), np.asarray(probs_mixed))


def test_sample_step_supports_multi_sample_estimation():
    model, variables = _init_tiny_cadd(K=3, params_seed=21, sample_seed=22)

    state = model.apply(
        variables,
        1,
        method=model.prior_sample,
        rngs={"sample": jax.random.PRNGKey(23)},
    )
    x_t, z_t = state
    assert x_t.shape == (1, 2)
    assert z_t.shape == (3, 1, 2, 4)

    x_s, z_s = model.apply(
        variables,
        jax.random.PRNGKey(24),
        0,
        4,
        state,
        conditioning=None,
        method=model.sample_step,
    )

    assert x_s.shape == (1, 2)
    assert x_s.dtype == jnp.int32
    assert z_s.shape == (3, 1, 2, 4)
    assert z_s.dtype == jnp.float32
