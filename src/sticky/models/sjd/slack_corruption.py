"""Forward law for Sudoku constraint-slack sites.

Slack sites have no anchor set (they live entirely in X_A) and follow a pure
VP-SDE forward kernel. This file is intentionally separate from corruption.py
so that the cell SJD path (sample_pair, mixture_logpdf, ...) is byte-identical
to today and other tasks (CIFAR / ImageNet64 / OpenWebText) cannot regress.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from sticky.rng import PRNGKey

from .sdes import vp_perturb

Array = jnp.ndarray


def sample_slack_pair(
    key: PRNGKey,
    slack_x0: Array,
    t: Array,
    beta: Callable[[Array], Array],
) -> Array:
    """Slack forward law: pure VP perturbation, no anchor stickiness.

    Parameters
    ----------
    slack_x0 : (B, 27, 9) clean slack vectors. For valid Sudoku solutions every
        entry equals 1.
    t : (B,) per-example forward time, shared with the cell sampler.
    beta : VP schedule callable.

    Returns
    -------
    slack_x_t : (B, 27, 9) noisy slacks.
    """
    slack_x0 = jnp.asarray(slack_x0, dtype=jnp.float32)
    slack_x_t, _target_score = vp_perturb(key, slack_x0, t, beta)
    return slack_x_t
