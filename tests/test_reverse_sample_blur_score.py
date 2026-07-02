"""Bit-exactness + W-aware behavior of the reverse_sample path (blur fix).

Fills the pre-fix coverage gap: nothing exercised `reverse_sample` /
`simple_generate` (the CIFAR/Text8 FID drift) at all. Stub linear classifier,
tiny grids, CPU-pinned (conftest.py) so int32 outputs are deterministic and
`np.array_equal` is valid (cf. the TF32 caveat in sjd_sudoku_wsweep.sbatch).

Covered:
  - kernel=None: blur_score True vs False are bit-identical (the in-repo G1
    proxy — the W branch adds zero ops without a kernel);
  - kernel attached + blur_score=False strips W bit-exactly (legacy A/B arm);
  - a real kernel with blur_score=True changes the samples;
  - eta != 1 with a kernel hard-errors unless blur_score=False;
  - known/clamped sites flow through the committed delta override
    (conditional_generate_board smoke; clues preserved).

Both geometries are exercised: 5-D image (B, 2, 2, 1, d) with a 4x4
gaussian_2d kernel (the layout every CIFAR arm uses) and 3-D sequence (B, 6)
with a 1-D gaussian kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sticky.models.sjd import sampling as sjd_sampling
from sticky.models.sjd.anchors import AnchorTable
from sticky.models.sjd.blur import (
    gaussian_2d_position_kernel,
    gaussian_position_kernel,
)
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sampler import SamplerConfig
from sticky.models.sjd.sdes import make_beta

_L = 5  # anchors
_D = 3  # anchor dim
_B = 2  # batch


class _StubModel:
    """Deterministic linear classifier: logits = y @ proj, any site layout."""

    def __init__(self, proj: np.ndarray):
        self._proj = jnp.asarray(proj, dtype=jnp.float32)  # (d, L)

    def apply(self, variables, y, *, t, train=False):
        del variables, t, train
        return jnp.einsum("...d,dk->...k", y, self._proj), None

    def anchor_table(self):  # pragma: no cover - not used by simple_generate
        raise NotImplementedError


def _make_harness():
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    rng = np.random.default_rng(0)
    a_table = jnp.asarray(rng.standard_normal((_L, _D)), dtype=jnp.float32)
    anchors = AnchorTable(table_float=a_table)
    model = _StubModel(rng.standard_normal((_D, _L)))
    return beta, hazard, anchors, model


IMAGE_SHAPE = (2, 2, 1)  # 5-D y layout, kernel over the 2x2 grid
SEQ_SHAPE = (6,)  # 3-D y layout


def _kernel_for(shape):
    if len(shape) == 3:
        return gaussian_2d_position_kernel(
            shape[0], shape[1], sigma=1.5, normalization="rowstoch"
        )
    return gaussian_position_kernel(shape[0], sigma=2.0, include_self=True)


def _run(
    *,
    shape,
    blur_kernel,
    blur_score: bool,
    eta: float = 1.0,
    seed: int = 42,
    n_steps: int = 4,
):
    beta, hazard, anchors, model = _make_harness()
    jump = VPMatchedGaussianJump(beta=beta, eta=eta, blur_kernel=blur_kernel)
    out = sjd_sampling.simple_generate(
        rng=jax.random.PRNGKey(seed),
        params={},
        model=model,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        batch_size=_B,
        shape=shape,
        cfg=SamplerConfig(n_steps=n_steps, tau_grid_size=8, blur_score=blur_score),
    )
    return (
        np.asarray(out.k_filled),
        np.asarray(out.k),
        np.asarray(out.committed),
    )


@pytest.mark.parametrize("shape", [IMAGE_SHAPE, SEQ_SHAPE], ids=["image5d", "seq3d"])
def test_no_kernel_blur_score_toggle_is_bit_exact(shape):
    """G1 in-repo proxy: without a kernel, the blur_score toggle must not
    change a single op — outputs are bit-identical."""
    on = _run(shape=shape, blur_kernel=None, blur_score=True)
    off = _run(shape=shape, blur_kernel=None, blur_score=False)
    for a, b in zip(on, off):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("shape", [IMAGE_SHAPE, SEQ_SHAPE], ids=["image5d", "seq3d"])
def test_blur_score_false_strips_kernel_bit_exactly(shape):
    """The legacy A/B arm: kernel attached + blur_score=False must be
    bit-identical to no kernel at all (the strip happens before tracing)."""
    stripped = _run(shape=shape, blur_kernel=_kernel_for(shape), blur_score=False)
    bare = _run(shape=shape, blur_kernel=None, blur_score=False)
    for a, b in zip(stripped, bare):
        assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "shape,n_steps",
    [(IMAGE_SHAPE, 8), (SEQ_SHAPE, 4)],
    ids=["image5d", "seq3d"],
)
def test_real_kernel_changes_samples(shape, n_steps):
    """The int32 outputs are a coarse observable; (kernel, n_steps) per
    geometry were chosen empirically so the drift delta flips at least one
    committed index on the pinned CPU backend (deterministic thereafter)."""
    kernel = _kernel_for(shape)
    on = _run(shape=shape, blur_kernel=kernel, blur_score=True, n_steps=n_steps)
    off = _run(shape=shape, blur_kernel=kernel, blur_score=False, n_steps=n_steps)
    assert not all(np.array_equal(a, b) for a, b in zip(on, off)), (
        "W-aware score produced bit-identical samples to the legacy score; "
        "the kernel/model fixture is too weak to exercise the branch."
    )


def test_eta_guard_raises_with_kernel_unless_legacy():
    kernel = _kernel_for(SEQ_SHAPE)
    with pytest.raises(ValueError, match="exact only at eta=1"):
        _run(shape=SEQ_SHAPE, blur_kernel=kernel, blur_score=True, eta=0.97)
    # blur_score=False is the sanctioned legacy escape hatch at any eta.
    _run(shape=SEQ_SHAPE, blur_kernel=kernel, blur_score=False, eta=0.97)


def test_known_sites_flow_through_committed_override():
    """conditional_generate_board with a kernel ON: clamped clues are
    committed from step 0, must survive to the output board, and the run
    must complete without shape errors through the delta override."""
    beta, hazard, anchors, model = _make_harness()
    kernel = _kernel_for(SEQ_SHAPE)
    jump = VPMatchedGaussianJump(beta=beta, eta=1.0, blur_kernel=kernel)

    rng = np.random.default_rng(3)
    tokens = np.zeros((_B,) + SEQ_SHAPE, dtype=np.int32)
    mask = np.zeros((_B,) + SEQ_SHAPE, dtype=bool)
    for b in range(_B):
        pos = rng.choice(SEQ_SHAPE[0], size=2, replace=False)
        mask[b, pos] = True
        tokens[b, pos] = rng.integers(1, _L + 1, size=2)

    board, _metrics = sjd_sampling.conditional_generate_board(
        rng=jax.random.PRNGKey(7),
        params={},
        model=model,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        known_tokens=jnp.asarray(tokens),
        known_token_mask=jnp.asarray(mask),
        cfg=SamplerConfig(n_steps=4, tau_grid_size=8, blur_score=True),
    )
    board = np.asarray(board)
    assert np.array_equal(board[mask], tokens[mask])
    assert board.min() >= 1 and board.max() <= _L
