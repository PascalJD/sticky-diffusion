"""G2 analytic gate for the W-aware classifier-induced score (eta=1).

Tiny enumerable SJD: L=2 anchors, N=4 sites (a 2x2 grid flattened row-major),
d=2, a known non-uniform joint p0 over all L**N = 16 clean sequences, and a
fixed ASYMMETRIC row-stochastic 4x4 W (asymmetry is load-bearing: a symmetric
W would let a W-transpose bug pass every check here). On the all-unstuck slice
the marginal density of y is the Gaussian mixture

    p_t(y) = sum_k p0(x0_k) prod_i N(y_i; alpha_t (W E(x0_k))_i, sigma_t^2 I_d)

(with uniform anchor weights the unstick-probability factors are
X0-independent and cancel from grad log p; for Lebesgue-a.e. y the mixed
measure's density is exactly this all-unstuck term). The gate asserts, in
float64,

    jax.grad(log p_t)(y)  ==  -(y - alpha_t * (W m(y))) / sigma_t^2

with m the exact per-site Bayes MARGINAL posterior means (delta on the
committed anchor at committed sites) — rtol <= 1e-5. Mutation checks assert
the identity FAILS under kernel.T and under kernel=I. Production float32
cross-checks drive `classifier_induced_score` itself with exact-posterior
logits on both the (B, N, d) and (B, H, W, C, d) layouts.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64
from jax.scipy.special import logsumexp

from sticky.models.sjd.corruption import classifier_induced_score
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sdes import alpha_sigma, make_beta

# ---------------------------------------------------------------------------
# Fixture constants (module-level, deterministic).
# ---------------------------------------------------------------------------

L = 2  # anchors
N = 4  # sites (2x2 grid, row-major)
D = 2  # embedding dim

# Generic anchors: not symmetric about 0, distinct coordinates.
E_TABLE = np.asarray([[1.7, -0.3], [-0.9, 1.1]], dtype=np.float64)

# Strongly asymmetric row-stochastic W (seeded random rows).
_W_RNG = np.random.default_rng(0)
_W_RAW = _W_RNG.uniform(0.05, 1.0, size=(N, N))
W_KERNEL = _W_RAW / _W_RAW.sum(axis=1, keepdims=True)  # rows sum to 1

# Known non-uniform, correlated joint p0 over all 16 clean sequences.
_P0_RNG = np.random.default_rng(1)
_P0_RAW = _P0_RNG.uniform(0.2, 1.0, size=L**N)
P0 = _P0_RAW / _P0_RAW.sum()

CONFIGS = np.asarray(list(itertools.product(range(L), repeat=N)), dtype=np.int64)
assert CONFIGS.shape == (L**N, N)

# Evaluation point away from score ~ 0 (seeded, mildly off-center).
_Y_RNG = np.random.default_rng(2)
Y_EVAL = _Y_RNG.normal(0.4, 0.9, size=(N, D)).astype(np.float64)

# Mid-trajectory (alpha, sigma) pairs; values match alpha_sigma(make_beta) at
# t = 0.3 / 0.7 approximately, but the identity holds for ANY (alpha, sigma).
ALPHA_SIGMA_CASES = [(0.6295, 0.7770), (0.8, 0.55)]


def _mixture_means(kernel: np.ndarray, alpha: float) -> np.ndarray:
    """mu_k = alpha * (W @ E[x0_k]) for every enumerated clean sequence."""
    e_fields = E_TABLE[CONFIGS]  # (K, N, D)
    return alpha * np.einsum("ij,kjd->kid", kernel, e_fields)


def _log_joint_components(
    y: jnp.ndarray, kernel: np.ndarray, alpha: float, sigma: float
) -> jnp.ndarray:
    """log[p0_k * prod_i N(y_i; mu_k_i, sigma^2 I_d)] for every component k."""
    mu = jnp.asarray(_mixture_means(kernel, alpha))  # (K, N, D)
    log_p0 = jnp.log(jnp.asarray(P0))
    sq = jnp.sum((y[None, :, :] - mu) ** 2, axis=(1, 2))  # (K,)
    log_norm = -0.5 * N * D * jnp.log(2.0 * jnp.pi * sigma * sigma)
    return log_p0 - sq / (2.0 * sigma * sigma) + log_norm


def _marginal_posterior_means(log_comp: jnp.ndarray) -> jnp.ndarray:
    """m_j = sum_a P_j(a | y) E(a) from joint component posteriors."""
    w = jnp.exp(log_comp - logsumexp(log_comp))  # (K,)
    onehot = jnp.asarray(
        (CONFIGS[:, :, None] == np.arange(L)[None, None, :]).astype(np.float64)
    )  # (K, N, L)
    post = jnp.einsum("k,knl->nl", w, onehot)  # per-site marginal P_j(a|y)
    return post @ jnp.asarray(E_TABLE)  # (N, D)


def _formula_score(
    y: jnp.ndarray, m: jnp.ndarray, kernel: np.ndarray, alpha: float, sigma: float
) -> jnp.ndarray:
    return -(y - alpha * (jnp.asarray(kernel) @ m)) / (sigma * sigma)


def _autograd_score(
    y: jnp.ndarray, kernel: np.ndarray, alpha: float, sigma: float
) -> jnp.ndarray:
    def log_p(y_in):
        return logsumexp(_log_joint_components(y_in, kernel, alpha, sigma))

    return jax.grad(log_p)(y)


# ---------------------------------------------------------------------------
# G2 core: autograd vs formula (float64, rtol <= 1e-5).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha,sigma", ALPHA_SIGMA_CASES)
def test_g2_autograd_matches_w_formula(alpha, sigma):
    with enable_x64():
        y = jnp.asarray(Y_EVAL)
        auto = _autograd_score(y, W_KERNEL, alpha, sigma)
        log_comp = _log_joint_components(y, W_KERNEL, alpha, sigma)
        m = _marginal_posterior_means(log_comp)
        formula = _formula_score(y, m, W_KERNEL, alpha, sigma)
        np.testing.assert_allclose(
            np.asarray(auto), np.asarray(formula), rtol=1e-5, atol=1e-8
        )


@pytest.mark.parametrize(
    "mutated_kernel_fn",
    [lambda w: w.T, lambda w: np.eye(N)],
    ids=["transpose", "identity"],
)
def test_g2_mutated_kernel_fails(mutated_kernel_fn):
    """The gate must be discriminating: evaluating the FORMULA with W.T or
    W=I against the true (W) autograd score must NOT match. Guards against a
    transposed einsum or a silently dropped kernel passing the gate."""
    alpha, sigma = ALPHA_SIGMA_CASES[0]
    mutated = mutated_kernel_fn(W_KERNEL)
    with enable_x64():
        y = jnp.asarray(Y_EVAL)
        auto = _autograd_score(y, W_KERNEL, alpha, sigma)
        log_comp = _log_joint_components(y, W_KERNEL, alpha, sigma)
        m = _marginal_posterior_means(log_comp)
        wrong = _formula_score(y, m, mutated, alpha, sigma)
        rel = np.max(
            np.abs(np.asarray(wrong) - np.asarray(auto))
            / (np.abs(np.asarray(auto)) + 1e-12)
        )
        assert rel > 1e-3, (
            f"mutated kernel unexpectedly matched the true score (max rel {rel});"
            " the fixture is degenerate — make W more asymmetric."
        )


# ---------------------------------------------------------------------------
# G2 committed variant: one site stuck; delta override on that site.
# ---------------------------------------------------------------------------

J_COMMITTED = 1
A_STAR = 0


def test_g2_committed_site_delta_override():
    alpha, sigma = ALPHA_SIGMA_CASES[0]
    subset = CONFIGS[:, J_COMMITTED] == A_STAR  # consistent clean sequences
    uncommitted = np.asarray([i for i in range(N) if i != J_COMMITTED])

    with enable_x64():
        y_full = jnp.asarray(Y_EVAL).at[J_COMMITTED].set(jnp.asarray(E_TABLE[A_STAR]))

        mu_all = jnp.asarray(_mixture_means(W_KERNEL, alpha))[jnp.asarray(subset)]
        log_p0_sub = jnp.log(jnp.asarray(P0[subset]))

        def log_p_unc(y_unc):
            # Conditional density over uncommitted coords given the stuck
            # pattern: the stuck coordinate contributes an X0-independent
            # delta within the consistent subset and drops out.
            sq = jnp.sum(
                (y_unc[None, :, :] - mu_all[:, jnp.asarray(uncommitted), :]) ** 2,
                axis=(1, 2),
            )
            return logsumexp(log_p0_sub - sq / (2.0 * sigma * sigma))

        y_unc = y_full[jnp.asarray(uncommitted)]
        auto = jax.grad(log_p_unc)(y_unc)  # (3, D)

        # Formula: marginal posterior means over the subset; delta at j0.
        sq = jnp.sum(
            (y_unc[None, :, :] - mu_all[:, jnp.asarray(uncommitted), :]) ** 2,
            axis=(1, 2),
        )
        log_comp = log_p0_sub - sq / (2.0 * sigma * sigma)
        w = jnp.exp(log_comp - logsumexp(log_comp))
        onehot = jnp.asarray(
            (CONFIGS[subset][:, :, None] == np.arange(L)[None, None, :]).astype(
                np.float64
            )
        )
        post = jnp.einsum("k,knl->nl", w, onehot)
        m = post @ jnp.asarray(E_TABLE)  # (N, D); row J_COMMITTED is exactly E(a*)
        np.testing.assert_allclose(
            np.asarray(m[J_COMMITTED]), E_TABLE[A_STAR], rtol=0, atol=1e-12
        )
        full_formula = _formula_score(y_full, m, W_KERNEL, alpha, sigma)
        np.testing.assert_allclose(
            np.asarray(auto),
            np.asarray(full_formula[jnp.asarray(uncommitted)]),
            rtol=1e-5,
            atol=1e-8,
        )


# ---------------------------------------------------------------------------
# Production float32 cross-checks: drive classifier_induced_score itself.
# ---------------------------------------------------------------------------


def _make_forward(kernel_f32):
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    jump = VPMatchedGaussianJump(
        beta=beta, eta=1.0, std_floor=1e-3, blur_kernel=kernel_f32
    )
    return beta, hazard, jump


def _exact_posterior_logits(alpha: float, sigma: float) -> np.ndarray:
    """log P_j(a | Y_EVAL) from the float64 enumeration (all-uncommitted)."""
    with enable_x64():
        log_comp = _log_joint_components(
            jnp.asarray(Y_EVAL), W_KERNEL, alpha, sigma
        )
        w = jnp.exp(log_comp - logsumexp(log_comp))
        onehot = jnp.asarray(
            (CONFIGS[:, :, None] == np.arange(L)[None, None, :]).astype(np.float64)
        )
        post = jnp.einsum("k,knl->nl", w, onehot)
        return np.log(np.maximum(np.asarray(post), 1e-300))


def test_production_score_matches_formula_3d():
    t = 0.3
    kernel32 = jnp.asarray(W_KERNEL, dtype=jnp.float32)
    beta, hazard, jump = _make_forward(kernel32)
    t_arr = jnp.full((1,), t, dtype=jnp.float32)
    alpha_t, sigma_t = alpha_sigma(beta, t_arr)
    alpha, sigma = float(alpha_t[0]), float(sigma_t[0])

    logits = jnp.asarray(
        _exact_posterior_logits(alpha, sigma), dtype=jnp.float32
    )[None]  # (1, N, L)
    y32 = jnp.asarray(Y_EVAL, dtype=jnp.float32)[None]  # (1, N, D)
    anchors = SimpleNamespace(
        table_float=jnp.asarray(E_TABLE, dtype=jnp.float32)
    )

    got = classifier_induced_score(
        y=y32,
        t=t_arr,
        anchor_logits=logits,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
    )

    with enable_x64():
        log_comp = _log_joint_components(jnp.asarray(Y_EVAL), W_KERNEL, alpha, sigma)
        m = _marginal_posterior_means(log_comp)
        want = _formula_score(jnp.asarray(Y_EVAL), m, W_KERNEL, alpha, sigma)
    np.testing.assert_allclose(
        np.asarray(got)[0], np.asarray(want), rtol=1e-4, atol=1e-5
    )


def test_production_score_5d_layout_matches_3d():
    """(B, 2, 2, 1, d) image layout must agree with the flat (B, 4, d) layout
    under the same 4x4 kernel — exercises blur_means' 5-D reshape path (the
    layout every CIFAR arm uses)."""
    t = 0.3
    kernel32 = jnp.asarray(W_KERNEL, dtype=jnp.float32)
    beta, hazard, jump = _make_forward(kernel32)
    t_arr = jnp.full((1,), t, dtype=jnp.float32)
    alpha_t, sigma_t = alpha_sigma(beta, t_arr)
    alpha, sigma = float(alpha_t[0]), float(sigma_t[0])

    logits = jnp.asarray(_exact_posterior_logits(alpha, sigma), dtype=jnp.float32)
    y32 = jnp.asarray(Y_EVAL, dtype=jnp.float32)
    anchors = SimpleNamespace(table_float=jnp.asarray(E_TABLE, dtype=jnp.float32))

    common = dict(
        t=t_arr, anchors=anchors, beta=beta, hazard=hazard, jump=jump
    )
    got_3d = classifier_induced_score(
        y=y32[None], anchor_logits=logits[None], **common
    )
    got_5d = classifier_induced_score(
        y=y32.reshape(1, 2, 2, 1, D),
        anchor_logits=logits.reshape(1, 2, 2, 1, L),
        **common,
    )
    np.testing.assert_allclose(
        np.asarray(got_5d).reshape(1, N, D),
        np.asarray(got_3d),
        rtol=1e-6,
        atol=1e-7,
    )


def test_production_committed_override_beats_contaminated_logits():
    """With a committed site whose logits deliberately argmax the WRONG
    anchor, the score must follow the committed delta (k_idx), proving the
    override path — and must differ from the contaminated no-override run."""
    t = 0.3
    kernel32 = jnp.asarray(W_KERNEL, dtype=jnp.float32)
    beta, hazard, jump = _make_forward(kernel32)
    t_arr = jnp.full((1,), t, dtype=jnp.float32)
    alpha_t, sigma_t = alpha_sigma(beta, t_arr)
    alpha, sigma = float(alpha_t[0]), float(sigma_t[0])

    subset = CONFIGS[:, J_COMMITTED] == A_STAR
    uncommitted = np.asarray([i for i in range(N) if i != J_COMMITTED])

    with enable_x64():
        y_full = jnp.asarray(Y_EVAL).at[J_COMMITTED].set(jnp.asarray(E_TABLE[A_STAR]))
        mu_all = jnp.asarray(_mixture_means(W_KERNEL, alpha))[jnp.asarray(subset)]
        log_p0_sub = jnp.log(jnp.asarray(P0[subset]))
        y_unc = y_full[jnp.asarray(uncommitted)]
        sq = jnp.sum(
            (y_unc[None, :, :] - mu_all[:, jnp.asarray(uncommitted), :]) ** 2,
            axis=(1, 2),
        )
        log_comp = log_p0_sub - sq / (2.0 * sigma * sigma)
        w = jnp.exp(log_comp - logsumexp(log_comp))
        onehot = jnp.asarray(
            (CONFIGS[subset][:, :, None] == np.arange(L)[None, None, :]).astype(
                np.float64
            )
        )
        post = jnp.einsum("k,knl->nl", w, onehot)
        m64 = post @ jnp.asarray(E_TABLE)
        want = _formula_score(y_full, m64, W_KERNEL, alpha, sigma)
        y_full64 = np.asarray(y_full)
        post_np = np.asarray(post)

    # Contaminate the committed site's logits: strongly favor the OTHER anchor.
    logits_np = np.log(np.maximum(post_np, 1e-30))
    logits_np[J_COMMITTED, :] = -30.0
    logits_np[J_COMMITTED, 1 - A_STAR] = 0.0
    logits = jnp.asarray(logits_np, dtype=jnp.float32)[None]

    y32 = jnp.asarray(y_full64, dtype=jnp.float32)[None]
    anchors = SimpleNamespace(table_float=jnp.asarray(E_TABLE, dtype=jnp.float32))
    committed = jnp.zeros((1, N), dtype=bool).at[0, J_COMMITTED].set(True)
    committed_idx = jnp.full((1, N), -1, dtype=jnp.int32).at[0, J_COMMITTED].set(
        A_STAR
    )

    common = dict(
        y=y32, t=t_arr, anchor_logits=logits, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump,
    )
    got = classifier_induced_score(
        committed=committed, committed_idx=committed_idx, **common
    )
    np.testing.assert_allclose(
        np.asarray(got)[0][uncommitted],
        np.asarray(want)[uncommitted],
        rtol=1e-4,
        atol=1e-5,
    )

    # Without the override the contaminated logits must yield a different
    # score at the uncommitted neighbors (the leak the fix removes).
    got_no_override = classifier_induced_score(**common)
    diff = np.max(
        np.abs(np.asarray(got_no_override)[0][uncommitted] - np.asarray(want)[uncommitted])
    )
    assert diff > 1e-3, (
        "contaminated committed-site logits did not perturb neighbor scores; "
        "the contamination fixture is degenerate."
    )
