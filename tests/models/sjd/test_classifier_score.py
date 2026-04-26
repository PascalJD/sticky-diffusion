from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sdes import alpha_sigma, make_beta
from sticky.models.sjd.corruption import classifier_induced_score


def _setup(*, eta=0.5, kappa=2.0, L=8, d=4, B=3, S=5, seed=0):
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=kappa)
    jump = VPMatchedGaussianJump(beta=beta, eta=eta, std_floor=1e-3)

    key = jax.random.PRNGKey(seed)
    k_a, k_y, k_logits = jax.random.split(key, 3)
    a_table = jax.random.normal(k_a, (L, d), dtype=jnp.float32)
    anchors = SimpleNamespace(table_float=a_table)

    y = jax.random.normal(k_y, (B, S, d), dtype=jnp.float32)
    anchor_logits = jax.random.normal(k_logits, (B, S, L), dtype=jnp.float32)
    t = jnp.linspace(0.3, 0.7, B, dtype=jnp.float32)
    return beta, hazard, jump, anchors, y, anchor_logits, t


def test_score_shape():
    """Output shape equals y.shape for both 2D (sequence) and 4D (image)
    batches. Image-style x_0 anchors live in (B, H, W, C, d)."""
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=2.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.5, std_floor=1e-3)
    L, d = 4, 3
    a_table = jax.random.normal(
        jax.random.PRNGKey(0), (L, d), dtype=jnp.float32
    )
    anchors = SimpleNamespace(table_float=a_table)

    # Sequence: (B, S, d).
    key = jax.random.PRNGKey(1)
    k_y, k_l = jax.random.split(key)
    y_seq = jax.random.normal(k_y, (3, 7, d), dtype=jnp.float32)
    logits_seq = jax.random.normal(k_l, (3, 7, L), dtype=jnp.float32)
    t_seq = jnp.full((3,), 0.5, dtype=jnp.float32)

    score = classifier_induced_score(
        y_seq, t_seq, anchor_logits=logits_seq, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump,
    )
    assert score.shape == y_seq.shape, (
        f"seq: {score.shape} != {y_seq.shape}"
    )

    # Image: (B, H, W, C, d) — CIFAR-10 shape style.
    key2 = jax.random.PRNGKey(2)
    k_y2, k_l2 = jax.random.split(key2)
    y_img = jax.random.normal(k_y2, (2, 4, 4, 3, d), dtype=jnp.float32)
    logits_img = jax.random.normal(k_l2, (2, 4, 4, 3, L), dtype=jnp.float32)
    t_img = jnp.full((2,), 0.5, dtype=jnp.float32)

    score = classifier_induced_score(
        y_img, t_img, anchor_logits=logits_img, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump,
    )
    assert score.shape == y_img.shape, (
        f"img: {score.shape} != {y_img.shape}"
    )


def test_score_finite():
    """Score is finite for `y` near anchors (sharp posterior) and `y` far
    from anchors (extreme dist^2 / variance ratios)."""
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=2.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.5, std_floor=1e-3)
    L, d = 4, 3
    a_table = jax.random.normal(
        jax.random.PRNGKey(0), (L, d), dtype=jnp.float32
    )
    anchors = SimpleNamespace(table_float=a_table)

    keys = jax.random.split(jax.random.PRNGKey(7), 4)
    idx = jax.random.randint(keys[0], (3, 5), 0, L)
    alpha_t, _ = alpha_sigma(beta, jnp.asarray(0.5, dtype=jnp.float32))
    # y near anchors: y[b, s] = alpha(t) * a[idx[b, s]] + tiny noise.
    y_near = float(alpha_t) * a_table[idx] + 0.01 * jax.random.normal(
        keys[1], (3, 5, d), dtype=jnp.float32
    )
    # y far: large magnitude.
    y_far = 100.0 * jax.random.normal(keys[2], (3, 5, d), dtype=jnp.float32)

    logits = jax.random.normal(keys[3], (3, 5, L), dtype=jnp.float32)
    t = jnp.array([0.3, 0.5, 0.7], dtype=jnp.float32)

    for name, y in (("near", y_near), ("far", y_far)):
        score = classifier_induced_score(
            y, t, anchor_logits=logits, anchors=anchors,
            beta=beta, hazard=hazard, jump=jump,
        )
        assert bool(jnp.all(jnp.isfinite(score))), (
            f"non-finite score for regime={name}"
        )
