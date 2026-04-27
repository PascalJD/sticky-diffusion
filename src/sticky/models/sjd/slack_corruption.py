"""Forward law for Sudoku constraint-slack sites.

Slack sites have no anchor set (they live entirely in X_A) and follow a pure
VP-SDE forward kernel. This file is intentionally separate from corruption.py
so that the cell SJD path (sample_pair, mixture_logpdf, ...) is byte-identical
to today and other tasks (CIFAR / ImageNet64 / OpenWebText) cannot regress.

TODO (Phase D — joint un-sticking kernel): the current implementation runs
INDEPENDENT cell-SJD and slack-VP processes given t_img. Under that design,
p^ac(y_cells, y_slacks | a_p) factors and the per-cell DHM spatial factor
r_a(y_p) / p^ac(y_p | a_p) does NOT literally depend on the slack — the
constraint signal flows through P_theta only. The genuinely-joint variant
draws (y_p, dS_R, dS_C, dS_B) jointly when cell p unsticks at tau and commits
to anchor e_v, with slack increments correlated to y_p - alpha(tau) * e_v.
The cleanest version is a Gaussian whose mean reflects the data-manifold
constraint that, conditional on cell p = e_v, the rest of p's row sums to
(1,...,1) - e_v, and whose covariance bundles y_p with dS_G. Only under that
design does p^ac(y_cells, y_slacks | a_v) not factor and r_a/p^ac literally
depend on slack. See docs/plans/task-augment-the-agile-popcorn.md.
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
