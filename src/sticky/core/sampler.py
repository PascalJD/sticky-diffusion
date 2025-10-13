# src/sticky/core/sampler.py
from __future__ import annotations
import jax, jax.numpy as jnp
import jax.random as jr
from typing import Callable, Optional, Tuple

Array = jnp.ndarray

def _reverse_vp_drift(
    score: Callable[[Array, Array, bool], Array],
    x: Array, 
    t: Array, beta_fn
) -> Array:
    beta_t = beta_fn(t)[..., None]
    s = score(x, t, False)
    return -0.5 * beta_t * (x + s)

def _heun_step(score, x, ti, ti1, beta_fn):
    h = (ti1 - ti)  # negative
    d1 = _reverse_vp_drift(score, x, jnp.full((x.shape[0],), ti), beta_fn)
    xe = x + h * d1
    d2 = _reverse_vp_drift(score, xe, jnp.full((x.shape[0],), ti1), beta_fn)
    return x + 0.5 * h * (d1 + d2)

def _flatten_marks(logits: Array, stuck_mask: Array) -> Tuple[Array, Array]:
    """Flatten (B,d,L) logits to (B, d*L) while masking already-stuck dims with -inf."""
    B, d, L = logits.shape
    # very negative where stuck
    bigneg = -1e9
    mask = jnp.where(stuck_mask[..., None], bigneg, 0.0)
    flat = (logits + mask).reshape(B, d * L)
    return flat, jnp.array([d, L])

def _apply_stick(x: Array, i: int, ell: int, bins: Array) -> Array:
    return x.at[:, i].set(bins[ell])

def sticky_sample_coord(
    key: jr.PRNGKey,
    n_samples: int,
    d: int,
    T: float,
    beta_fn,
    score_apply: Callable,
    cls_apply: Callable,
    int_apply: Callable,
    params_score,
    params_cls,
    params_int,
    anchors,
    steps: int = 50,
    eps: float = 1e-4,
    track_index: Optional[int] = None,
) -> Tuple[Array, Array, Array]:
    """Deterministic ODE sampling with sticky jumps (reverse time).
    Returns:
      x_T0: (n_samples, d) final samples in the *current* space
      t_hist: (steps,) times for trajectory plotting
      v_hist: (steps,) values of a tracked coordinate (first sample), if track_index is not None
    """
    key_s, key_e, key_m = jr.split(key, 3)

    # time grid, reverse direction
    tgrid = jnp.linspace(T, eps, steps)
    dt = jnp.abs(tgrid[:-1] - tgrid[1:])  # size (steps-1,)

    # initialize from standard normal
    x = jr.normal(key_s, (n_samples, d))
    stuck = jnp.zeros((n_samples, d), dtype=bool)

    v_hist = []
    t_hist = []

    def score(x, t, train: bool):
        # pass-through wrapper to your Flax module
        return score_apply({'params': params_score}, x, t, False)

    for k in range(steps - 1):
        ti, ti1 = tgrid[k], tgrid[k + 1]

        # ODE step
        x = _heun_step(score, x, ti, ti1, beta_fn)

        # total reverse stickiness
        lam = int_apply({'params': params_int}, x, jnp.full((x.shape[0],), ti), False).reshape(-1)
        lam = jnp.maximum(0.0, lam)  # safety

        # Poisson thinning: P(event in [ti1, ti]) ~ 1 - exp(-lambda delta t)
        p_evt = 1.0 - jnp.exp(-lam * (ti - ti1))
        key_e, kval, ksel = jr.split(key_e, 3)
        u = jr.uniform(kval, (n_samples,))
        happened = (u < p_evt)

        if happened.any():
            # logits over anchors for each sample
            logits = cls_apply({'params': params_cls}, x, jnp.full((x.shape[0],), ti), False)  # (B,d,L)
            flat_logits, shape = _flatten_marks(logits, stuck)  # (B, d*L)
            # sample a single anchor (i,ell) per event sample
            sel = jr.categorical(ksel, flat_logits)
            di, L = int(shape[0]), int(shape[1])
            idx_i = (sel // L).astype(jnp.int32)
            idx_l = (sel %  L).astype(jnp.int32)

            # apply sticks only to rows where happened=True
            # broadcast bins
            def stick_row(xrow, do, i, ell):
                return jnp.where(do, xrow.at[i].set(anchors.bins[ell]), xrow)

            x = jax.vmap(stick_row)(x, happened, idx_i, idx_l)
            # freeze those coordinates
            stuck = stuck.at[jnp.arange(n_samples), idx_i].set(stuck[jnp.arange(n_samples), idx_i] | happened)

        if track_index is not None and n_samples > 0:
            v_hist.append(float(x[0, track_index]))
            t_hist.append(float(ti1))

    t_hist = jnp.array(t_hist) if track_index is not None else jnp.array([])
    v_hist = jnp.array(v_hist) if track_index is not None else jnp.array([])

    return x, t_hist, v_hist