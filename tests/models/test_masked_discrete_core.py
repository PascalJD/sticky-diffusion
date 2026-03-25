from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sticky.models import masked_discrete_core as masked_core


def _reference_loglinear_move_chance(t: jnp.ndarray, *, eps: float) -> jnp.ndarray:
    return eps + (1.0 - 2.0 * eps) * t


def _reference_loglinear_sigma(t: jnp.ndarray, *, eps: float) -> jnp.ndarray:
    move_chance = _reference_loglinear_move_chance(t, eps=eps)
    return -jnp.log1p(-move_chance)


def _reference_loglinear_gamma(t: jnp.ndarray, *, eps: float) -> jnp.ndarray:
    sigma = _reference_loglinear_sigma(t, eps=eps)
    return -jnp.log(jnp.expm1(sigma))


def _reference_loglinear_dgamma_times_alpha(t: jnp.ndarray, *, eps: float) -> jnp.ndarray:
    move_chance = _reference_loglinear_move_chance(t, eps=eps)
    return -(1.0 - 2.0 * eps) / move_chance


def test_loglinear_schedule_matches_reference_formulas():
    eps = 1.0e-4
    t = jnp.asarray([0.0, 0.2, 0.5, 0.8, 1.0], dtype=jnp.float32)

    np.testing.assert_allclose(
        np.asarray(
            masked_core.clipped_schedule_move_chance(
                t, schedule_fn_type="loglinear", eps=eps
            )
        ),
        np.asarray(_reference_loglinear_move_chance(t, eps=eps)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(masked_core.masked_sigma_schedule(t, schedule_fn_type="loglinear", eps=eps)),
        np.asarray(_reference_loglinear_sigma(t, eps=eps)),
        rtol=5.0e-5,
        atol=5.0e-4,
    )
    np.testing.assert_allclose(
        np.asarray(masked_core.masked_logit_schedule(t, schedule_fn_type="loglinear", eps=eps)),
        np.asarray(_reference_loglinear_gamma(t, eps=eps)),
        rtol=5.0e-5,
        atol=5.0e-4,
    )
    np.testing.assert_allclose(
        np.asarray(
            masked_core.masked_dgamma_times_alpha(
                t, schedule_fn_type="loglinear", eps=eps
            )
        ),
        np.asarray(_reference_loglinear_dgamma_times_alpha(t, eps=eps)),
        rtol=5.0e-4,
        atol=5.0,
    )


def test_loglinear_sampling_grid_matches_uniform_reference_discretization():
    for i in range(6):
        uniform_s, uniform_t = masked_core.make_sampling_time_pair(
            i, 8, sampling_grid="uniform"
        )
        loglinear_s, loglinear_t = masked_core.make_sampling_time_pair(
            i, 8, sampling_grid="loglinear"
        )
        np.testing.assert_allclose(np.asarray(loglinear_s), np.asarray(uniform_s))
        np.testing.assert_allclose(np.asarray(loglinear_t), np.asarray(uniform_t))
