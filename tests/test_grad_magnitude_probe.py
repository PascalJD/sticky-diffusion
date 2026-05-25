"""Gradient-magnitude probe (plan validation #3).

Measures ||grad_log_w L_total|| and ||grad_theta L_total|| under both:
  (a) the existing `hazard_deriv` path (ce_allocation_loss), and
  (b) the new `elbo_eta1` path (elbo_eta1_loss),
on the same minibatch at log_w = 0, with a tiny synthetic model whose
"params" are a single weight vector. Prints the norms and the ratio
||grad_log_w|| / ||grad_theta|| for each path.

The new path's gradient on log_w is *unbiased* (no Bernoulli stop-grad), so
its magnitude can differ substantially from the legacy biased gradient. The
ratio reported here informs `training.log_w_lr_multiplier` (see plan Risk #1);
do not guess a multiplier without running this probe.

Asserts only finiteness, that the new gradient is non-zero, and that the
two paths produce *different* log_w gradients (the bias-removal predicts
they will differ — if they don't, the diagnosis was wrong).
"""
import jax
import jax.numpy as jnp

from sticky.models.sjd.sdes import make_beta
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.losses import ce_allocation_loss
from sticky.models.sjd.sjd_elbo_loss import elbo_eta1_loss


B, L_seq, K, d = 16, 4, 8, 4


def _make_schedules():
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    jump = VPMatchedGaussianJump(beta=beta)
    return beta, hazard, jump


def _make_inputs(rng):
    rng_idx, rng_anc = jax.random.split(rng)
    x0_idx = jax.random.randint(rng_idx, shape=(B, L_seq), minval=0, maxval=K)
    anchor_table = jax.random.normal(rng_anc, shape=(K, d))
    x0_anchor = jnp.take(anchor_table, x0_idx, axis=0)
    return x0_idx, x0_anchor, anchor_table


def _apply_fn_with_params(params, x_in, t_img):
    """Tiny model: logits = (x_in . theta) + t_img offset.

    `params` is shape (d, K). Gradient w.r.t. params is non-trivial so we
    can compare ||grad_theta|| to ||grad_log_w||.
    """
    W = params["W"]                   # (d, K)
    logits = jnp.einsum("bld,dk->blk", x_in, W) + t_img[:, None, None]
    return logits, {}


def _norm(g):
    leaves = jax.tree_util.tree_leaves(g)
    return float(jnp.sqrt(sum(jnp.sum(l * l) for l in leaves)))


def test_gradient_magnitude_probe():
    beta, hazard, jump = _make_schedules()
    rng = jax.random.PRNGKey(123)
    x0_idx, x0_anchor, _ = _make_inputs(rng)
    rng_params, _ = jax.random.split(rng)
    params0 = {"W": jax.random.normal(rng_params, shape=(d, K)) * 0.1}
    log_w0 = jnp.zeros((K,), dtype=jnp.float32)
    key = jax.random.PRNGKey(7)

    # ---- (a) Legacy hazard_deriv path ----
    def hd_loss(params, log_w):
        loss, _ = ce_allocation_loss(
            key=key,
            params=params,
            apply_fn=_apply_fn_with_params,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=beta,
            hazard=hazard,
            jump=jump,
            T=1.0,
            loss_weighting="hazard_deriv",
            anchor_log_w=log_w,
            log_anchor_log_w_stats=False,
        )
        return loss

    g_params_hd, g_logw_hd = jax.grad(hd_loss, argnums=(0, 1))(params0, log_w0)
    nrm_th_hd = _norm(g_params_hd)
    nrm_lw_hd = _norm(g_logw_hd)

    # ---- (b) New elbo_eta1 path ----
    def e2e_loss(params, log_w):
        loss, _ = elbo_eta1_loss(
            key=key,
            params=params,
            apply_fn=_apply_fn_with_params,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=beta,
            hazard=hazard,
            jump=jump,
            T=1.0,
            anchor_log_w=log_w,
            prior_strength=0.0,
        )
        return loss

    g_params_e2e, g_logw_e2e = jax.grad(e2e_loss, argnums=(0, 1))(params0, log_w0)
    nrm_th_e2e = _norm(g_params_e2e)
    nrm_lw_e2e = _norm(g_logw_e2e)

    print(
        f"\n[grad-magnitude-probe] at log_w=0, B={B}, L_seq={L_seq}, K={K}, d={d}:\n"
        f"  hazard_deriv: ||grad_theta||={nrm_th_hd:.5g}, ||grad_log_w||={nrm_lw_hd:.5g}, "
        f"ratio log_w/theta = {nrm_lw_hd / max(nrm_th_hd, 1e-12):.5g}\n"
        f"  elbo_eta1   : ||grad_theta||={nrm_th_e2e:.5g}, ||grad_log_w||={nrm_lw_e2e:.5g}, "
        f"ratio log_w/theta = {nrm_lw_e2e / max(nrm_th_e2e, 1e-12):.5g}\n"
        f"  Default log_w_lr_multiplier was tuned for hazard_deriv (100x).\n"
        f"  For elbo_eta1, a reasonable starting multiplier is:\n"
        f"     hd_ratio / e2e_ratio * 100 = "
        f"{(nrm_lw_hd / max(nrm_th_hd, 1e-12)) / max(nrm_lw_e2e / max(nrm_th_e2e, 1e-12), 1e-12) * 100:.5g}\n"
        f"  (Empirical re-tuning on a real model is still advised — this is a synthetic probe.)"
    )

    # Sanity assertions
    assert all(
        jnp.isfinite(jnp.asarray(v)).item()
        for v in (nrm_th_hd, nrm_lw_hd, nrm_th_e2e, nrm_lw_e2e)
    ), "non-finite gradient norm"
    assert nrm_lw_e2e > 1e-6, "elbo_eta1 grad on log_w is ~ 0 (silent stop_grad?)"
    # Bias-removal sanity: the two paths should give different log_w gradients.
    # If they're suspiciously close, the diagnosis (bias from non-differentiable
    # Bernoulli mask) might be wrong.
    rel_diff = float(
        jnp.linalg.norm(g_logw_e2e - g_logw_hd) / max(_norm(g_logw_hd), 1e-12)
    )
    assert rel_diff > 1e-3, (
        f"hazard_deriv and elbo_eta1 produced ~identical log_w gradients (rel "
        f"diff = {rel_diff:.3e}). The bias-removal diagnosis predicts they differ. "
        f"Investigate before relying on the new mode."
    )
