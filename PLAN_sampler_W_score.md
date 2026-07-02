# PLAN_sampler_W_score.md — W-aware classifier-induced score in the SJD sampler

> Status: IMPLEMENTED (2026-07-02), pending job launches. All §4 code changes and §7
> tests are in the working tree (uncommitted); full pytest suite passes (198 tests);
> campaign gates 1/2/3a/4 PASS with the updated Gate-4 invariant; a 3-lens adversarial
> review found no blocker/major issues (its 4 minor findings are fixed). G1 evidence so
> far: pre-fix vs post-fix `reverse_sample` jaxprs are textually identical and jitted
> outputs bit-identical at blur=none (stub level). Still HELD, per §8 ordering, until
> explicitly launched: the operational G1 run on the real `cifar_base_seed1` checkpoint
> (`scripts/g1_dump_kfilled.py`), Phase A (`scripts/eval_reeval_fid1k.sbatch`, includes
> G3+G4), Phase B (`scripts/reeval_select.py`), Phase C
> (`scripts/eval_endpoint_reselected.sbatch`), and Phase D (additive arm). §9-R0 was
> executed at plan approval: the additive arm's checkpoints are snapshotted read-only at
> `runs/cifar_additive_s1.5_seed0/checkpoints_snapshot_20260702/` (40 files, verified
> byte-equal). Known follow-up (review Lens 3): `text_valid_word` offline reports carry
> no blur provenance — add before launching any OWT/Text8 W runs.

## 0. Context

The SJD forward corruption blends anchor embeddings through a fixed site matrix W:
`sample_pair` uses `jump.apply_blur(x0_anchor)` ([corruption.py:111]) → `blur_means`
(einsum `'ij,bjcd->bicd'` per color channel, [blur.py:305]); W is built by
`build_blur_kernel` from the `forward/blur` config group and attached to
`VPMatchedGaussianJump.blur_kernel` by `tasks/factory.py:_build_forward_schedule`
(L77–112). The reverse sampler's score, however, is computed with the W=I formula:
`classifier_induced_score` ([corruption.py:415–564]) carries explicit `TODO(phase-2)`
banners "this inference path currently assumes W = I". Every W≠I run samples with a
biased reverse drift. Training is unaffected (CE on corrupted pairs).

Empirical smoking gun: the additive-σ1.5 CIFAR arm's best in-training FID is ~295 vs
~42–44 for every rowstoch/base arm.

Scope: sampler/eval-side fix at η=1 (all CIFAR arms use η=1.0), validation gates,
then re-evaluation of the CIFAR blur campaign. η<1 ∧ W≠I is rejected with an assertion.
The forward corruption (`sample_pair`), losses, training loop, and configs are not
touched (see §10-D1 for the one file-boundary nuance).

Two prompt assumptions were corrected by investigation:
- "The sampler has no access to the blur config" — partially true. The Sudoku eval
  passes `task.forward.jump` (kernel attached) but the score ignores it; the CIFAR/text
  FID builders (`build_sampling_fns` [training/sampling.py:158] and
  `build_multi_fid_sampling_fns` [:301]) **re-instantiate the jump from
  `cfg.forward.jump` and drop the kernel entirely**. Both gaps are fixed here.
- Suspected Sudoku/Text8 W-run bias: **no Sudoku or Text8 W-runs exist on disk**
  (verified: `~/scratch/sticky-diffusion/runs` all blur-none or empty `.hydra`; both
  wandb trees blur-disabled). Only config templates exist. Nothing to re-evaluate there.

## 1. Correct formula (η=1, any fixed W; per color channel)

For off-anchor (uncommitted) position i at time t:

    s_i(y,t) = −( y_i − α(t)·μ̂_i(y,t) ) / v(t),      v(t) = max(σ²(t), std_floor²)
    μ̂_i     = Σ_j W_ij m_j(y,t)
    m_j      = Σ_a P_θ^{(j)}(a|y,t) E(a)      (uncommitted j)
    m_j      = E(k_j)   (delta on committed anchor)   (committed j)

Same einsum orientation as the corruption (left-multiplication `'ij,bjcd->bicd'`,
row-major flat p = r·W_grid + c, per channel). Justification: μ_i(X₀)=Σ_j W_ij E(X₀_j)
is linear in the embedded sequence; at η=1 every unstick-time component shares mean
α(t)μ_i and variance σ²(t), so E[μ_i(X₀)|y]=(W m(y))_i by the tower property with
per-position **marginal** joint-posterior means (no independence assumption; verified
adversarially). v(t) uses std_floor=1e-3 because at η=1.0 the existing τ-quadrature
collapses exactly: `deficit=(1−η²)(…)=0.0` in float, `v=max(σ²−0, std_floor²)`,
`e_l=1/v` constant ⇒ current code ≡ −(y−αm)/v at W=I. (Note: the quadrature's
`e=n/l` equals 1/v only algebraically, ~1e-7 ULP drift — no gate depends on
W=I-as-matrix bitwise equality; only blur=none must be bit-exact, and it is, by a
trace-time Python bypass.)

Hazard λ̂ / commit allocation: **unchanged at η=1, verified in code.** In
`plugin_intensity.dhm_log_ratio`, numerator `vp_jump_logpdf_all_anchors` (var
`max((η σ)², floor²)`) and denominator `mixture_logpdf_all_anchors` (v =
`max(σ²−0, floor²)`) are bitwise-equal at η=1 and share the same `dist2` — the
Gaussian factor cancels, the log-ratio is state/anchor-independent, allocation reduces
to the tempered classifier posterior, and λ̂ depends on t only. This cancellation holds
for the true (W-blurred) densities identically, so leaving `plugin_intensity` and the
three `mixture_logpdf*` functions untouched is exactly correct (cf. vp_matched.yaml
"eta=1 cancels state dependency"). Residuals: a pre-existing `+1e-12` denominator
asymmetry and LSE rounding — unchanged by this fix.

Out of scope: η<1 ∧ W≠I (τ-mixture weights couple to μ; needs the quadrature
generalization). Rejected by assertions (§4.2, §4.3).

## 2. Call-site inventory (spec item 1)

| Path | Chain | Jump carries W? | Score W-aware? | Verdict |
|---|---|---|---|---|
| CIFAR in-training FID trajectory (N=1k, NFE 256 = sampler.n_steps, EMA via `params_for_sampling`) | loop_helpers → `build_eval_logger` (eval.py:161/501) → `build_sampling_fns` (sampling.py:150) → `simple_generate` → `reverse_sample` → `classifier_induced_score` | **No** (:158 drops it) | No | **Biased for W≠I — fix** |
| CIFAR in-training image logging | same builder | No | No | same fix |
| Endpoint eval array (50k, `eval_checkpoint`) | offline_eval.py:727–758 → same builders (+ multi-arm :301) | No | No | **Biased — fix** |
| Text8 / OWT generation (valid-word, genppl; eval.py:277, offline_eval.py:510) | same `build_sampling_fns` | No | No | Shared path; all trained runs W=I → numerically unaffected; future W runs (e.g. untrained `sjd_openwebtext_blur`, gaussian_1d η=1) fixed for free |
| Sudoku `predictor_only` (kind: sampler) | sudoku.py:934–969 → `conditional_generate_board` → `reverse_sample` | **Yes** (task.forward.jump) | No | **Same bug, same fix** (clues = committed handled by the delta override) |
| Sudoku policy (kind: policy) | `board_sampling.conditional_generate` — independent inline η=1 score (:307) with opt-in `blur_score` (η=1-guarded) | Yes | opt-in | Semantics unchanged; refactor to shared helper only (bit-identical delegation) |
| `state_dependency.py` diagnostics | mixture_logpdf / jump.logpdf | — | No | W=I diagnostics only; out of scope, noted |

All CIFAR/Text8/Sudoku-predictor sampling funnels through **one** function
(`classifier_induced_score` via `reverse_sample`); the Sudoku policy sampler is the
only second implementation and already has the correct gated fix.

## 3. W reconstruction + fingerprint at sampling (spec item 2)

- Kernel source at sampling: prefer `task.forward.blur_kernel` — the exact array the
  task factory built (same builder as training; also reflects any
  `offline_eval.jump_eta`-era config state since the task is built from the
  already-overridden effective config). Fallback: rebuild via
  `build_blur_kernel(cfg.forward.blur, seq_len=prod(data_shape),
  grid_shape=data_shape[:2])`, mirroring factory.py:98–110 (covers duck-typed tasks).
- Endpoint/re-eval jobs run with `offline_eval.use_run_config=true` (already the
  campaign convention, eval_endpoint.sbatch): the run's archived config from
  `run_context.json` drives the rebuild, so trained σ/normalization are reproduced by
  construction; G3 verifies it.
- New `kernel_fingerprint(kernel)` in blur.py: sha256 over `"float32:<shape>"` header +
  row-major float32 bytes (host numpy via `jax.device_get`; `None → None`).
  **Honest guarantee**: the training-time array was never persisted, so G3 certifies
  "deterministic builder + config equality" — rebuild-from-archived-config vs
  kernel-actually-attached — not byte identity with the historical array. Softmax bit
  patterns are backend-sensitive → compare fingerprints only within the same
  job/backend; also record the blur config tuple (enabled/kind/sigma/normalization) and
  kernel shape for loose cross-platform checks.
- Recording: (a) `offline_eval_metrics.json` — extend `_extract_forward_config_metadata`
  (offline_eval.py:153–179, currently omits blur entirely) with the blur config block,
  plus `forward_blur_kernel_fingerprint`, `kernel_shape`, and the effective
  `sampler_blur_score`; (b) future runs — `write_run_context` gains an optional
  `forward_blur` payload (config + fingerprint + shape) computed in `loop_helpers`
  inside try/except (provenance must never abort training).

## 4. Exact diff sketch (spec item 4)

blur=none stays a strict bypass throughout: `build_blur_kernel` returns `None`, all new
branches are trace-time `if kernel is not None` Python checks → zero new ops in the
jaxpr → bit-exact (same mechanism as the existing `apply_blur` identity bypass).

### 4.1 `src/sticky/models/sjd/blur.py` — shared helper + fingerprint
Append after `blur_means` (L310); add both names to `__all__`:
```python
def blurred_posterior_mean(*, probs_mean, committed_mask, committed_idx,
                           a_table, kernel):
    """Ehat = committed one-hot anchor embedding at committed sites, else the
    classifier posterior mean; returns blur_means(Ehat, kernel) = W @ Ehat.
    Shape-generic ((B,N,d) and (B,H,W,C,d)); -1 sentinel in committed_idx clips
    to 0 and is discarded by the mask select."""
    safe_idx = jnp.clip(committed_idx, 0, a_table.shape[0] - 1)
    committed_vec = a_table[safe_idx]
    e_hat = jnp.where(committed_mask[..., None], committed_vec, probs_mean)
    return blur_means(e_hat, kernel)

def kernel_fingerprint(kernel) -> str | None:
    # sha256 of "float32:<shape>" + row-major float32 bytes; None -> None.
    # Backend-sensitive: compare only within one job/backend (see §3).
```
`blurred_posterior_mean` is a verbatim move of `board_sampling.blur_score_mean`
(L166–184) — identical op sequence, Sudoku policy path stays bit-identical.

### 4.2 `src/sticky/models/sjd/corruption.py` — W-aware branch in the score
Only `classifier_induced_score` changes; `sample_pair` and the three `mixture_logpdf*`
functions are untouched (see §10-D1). New trailing kwargs
`committed: Array | None = None, committed_idx: Array | None = None`; branch inserted
after the `alpha_t/sigma_t` block (L476), before `eta = float(jump.eta)` (L478) — all
needed values (`probs_flat`, `y_flat`, `a_table`, `B,S,d,site_shape`) already in scope:
```python
    blur_kernel = getattr(jump, "blur_kernel", None)
    if blur_kernel is not None:
        if abs(float(jump.eta) - 1.0) > 1e-9:
            raise ValueError("... exact only at eta=1; got eta=... "
                             "Sample with eta=1 or set sampler blur_score=false.")
        if (committed is None) != (committed_idx is None):
            raise ValueError("committed and committed_idx must both be set or None.")
        m_flat = jnp.einsum("bsl,ld->bsd", probs_flat, a_table)      # (B,S,d)
        # committed=None => all-uncommitted (zeros mask, -1 idx)
        mu = blurred_posterior_mean(
            probs_mean=m_flat.reshape(y.shape),
            committed_mask=committed_site, committed_idx=committed_idx_site,
            a_table=a_table, kernel=blur_kernel)                     # W @ Ehat
        v_b = jnp.maximum(jnp.square(sigma_t),
                          jnp.square(jnp.float32(jump.std_floor)))   # (B,)
        score_flat = -(1.0 / v_b)[:, None, None] * (
            y_flat - alpha_t[:, None, None] * mu.reshape((B, S, d)))
        return score_flat.reshape(y.shape)
    # kernel is None: fall through to the untouched quadrature path (bit-exact)
```
`posterior_temperature` (symmetric_temperature) semantics preserved: the branch reuses
the already-tempered `probs_flat`. Update the banner above the function (resolved at
η=1; η<1∧W≠I raises); annotate the three logpdf banners with the η=1 cancellation note.

### 4.3 `src/sticky/models/sjd/sampler.py` — toggle, guard, plumbing
- `SamplerConfig`: append `blur_score: bool = True` (True = W-aware score when a kernel
  is attached; False = legacy pre-fix behavior; no-op when kernel is None).
- `reverse_sample` entry (after the logit_temperature check, L131) — **strip first,
  then guard** (blur_score=false + η<1 is the legal legacy arm and must not raise):
```python
    if getattr(jump, "blur_kernel", None) is not None:
        if not bool(cfg.blur_score):
            jump = replace(jump, blur_kernel=None)   # legacy A/B path
        elif abs(float(jump.eta) - 1.0) > 1e-9:
            raise ValueError("reverse_sample with jump.blur_kernel requires eta=1 ...")
```
- `_predictive_stats(logits, y_state, t_img_state, committed_state, k_idx_state)`
  passes `committed=..., committed_idx=...` into the score; call site (L224) becomes
  `_predictive_stats(logits_score, y_for_model, t_img, committed, k_idx)` — the pre-EM
  carry snapshot, consistent with the state the classifier saw. `committed`/`k_idx` are
  already traced carry values; the None path performs no op on them (acceptance
  criterion for G1).

### 4.4 `src/sticky/training/sampling.py` — kernel attach + spec key
- New helper `_attach_blur_kernel(jump, *, cfg, task)`: prefer
  `task.forward.blur_kernel`; else rebuild from `cfg.forward.blur` with
  `task.spec.data_shape` (mirrors factory); return jump unchanged when no blur
  (strict bypass). Called after both `hydra.utils.instantiate(cfg.forward.jump, ...)`
  sites (`build_sampling_fns` L158, `build_multi_fid_sampling_fns` L301).
- `_base_sampler_spec_from_cfg`: add `"blur_score": bool(cfg_sampler.get("blur_score",
  True))`. **Load-bearing**: `build_multi_fid_sampling_fns`' override merge only
  accepts keys already in the base spec (L325 `if k in spec`).
- `_sjd_sampler_cfg_from_dict`: map `spec["blur_score"]` → `SamplerConfig`.

### 4.5 `src/sticky/models/sjd/board_sampling.py` — delegation only
`blur_score_mean` body delegates to `blur.blurred_posterior_mean` (keep the exported
name and keyword signature — `tests/test_blur_score_eval.py` imports it); fix its
docstring (it is already shape-generic beyond `(B,N,d)`). No semantic change; policy
default stays `blur_score=False`.

### 4.6 `src/sticky/eval/sudoku.py` — escapable guard for sampler-kind
With `blur_score` defaulting True, a *future* η≠1 Sudoku W-run (e.g. `sjd_sudoku_blur`,
η=0.97) would hard-error at its first in-training predictor_only eval — fail-loud
replacing silent bias, which is intended, but there must be an escape hatch: plumb
`blur_score` through the sampler-kind spec into the `SamplerConfig` built at
sudoku.py:935–950 (replacing the stale rejection at :368–372 for sampler-kind entries)
and record `blur_score_effective` for sampler-kind rows (mirroring :1008–1015).
No behavior change for any existing run (none has W).

### 4.7 Metadata: `offline_eval.py`, `persistence.py`, `loop_helpers.py`
Per §3: blur block + fingerprint + `sampler_blur_score` in `offline_eval_metrics.json`;
optional `forward_blur` payload in `run_context.json` for future runs (try/except).

### 4.8 Campaign scripts (`/home/x-pjutrasdube/projects/cifar_blur_campaign/scripts/`)
- `run_gates.py` Gate 4a currently asserts `sampler.py`/`sampling.py` contain **no**
  blur references — inverted by this fix by design. Update Gate 4 to the new invariant
  (sampler threads W; `board_sampling` policy default still False; η guard present).
- New `verify_w_fingerprint.py` (G3) and `eval_reeval_fid1k.sbatch` (§6), plus a small
  `g1_dump_kfilled.py` driver (§5-G1).
- Update the stale comment in `config/forward/blur/gaussian_2d.yaml` ("The sampler does
  NOT use W") — comment-only config change, flagged for the no-configs rule (§10-D1).

## 5. Committed positions (spec item 3) — CONFIRMED DIFFERENT, fixed

`classifier_induced_score` has no committed argument today; m_j at a committed site is
the raw classifier softmax mean at the clamped y — under W=I harmless (committed
scores are discarded by the `~committed` drift mask; no cross-site coupling), under
W≠I it leaks into uncommitted neighbors' μ̂. The fix threads `committed`/`k_idx` from
the sampler carry and applies the delta override (§4.2). Committed y_j equals
`a_table[k_j]` bit-exactly (sampler.py:339 + `clamp_known_state`), and committed sites
always have valid `k_idx ≥ 0`, so the override is well-defined and also covers
inpainting clues. Forward-consistency check (adversarial review): stuck sites hold the
raw unblurred anchor while unstuck means blend **all** sites' clean anchors through W —
Ehat(delta at committed, posterior mean elsewhere) is exactly the conditional-mean
plug-in for that structure. CIFAR shapes at the call site: y `(B,32,32,3,8)`, committed
`(B,32,32,3)` bool, k_idx int32 with −1 sentinel; kernel `(1024,1024)` applied per
channel by the 5-D `blur_means` path — identical contraction to the forward.

## 6. Validation gates (spec item 5) — all must pass before any re-evaluation

- **G1 bit-exactness (blur=none)**: `cifar_base_seed1/checkpoints/best/checkpoint_75000`,
  same seed (offline harness derives `fold_in(make_rng(seed+12345), restored_step)`),
  pre-fix vs post-fix code → `np.array_equal` on dumped `k_filled` (new
  `g1_dump_kfilled.py`; the offline CLI emits only FID scalars). Same device/precision
  both runs (`JAX_DEFAULT_MATMUL_PRECISION=float32` on GPU). Acceptance criteria for
  the diff: None-branch first, no new RNG, no ops on `committed` in the None path.
  Permanent in-repo proxy: stub-model `reverse_sample` tests (§7) — this also fills the
  existing coverage gap (nothing currently tests `reverse_sample` at all).
- **G2 analytic score** (`tests/test_w_score_analytic.py`): 2 anchors, 2×2 grid (N=4),
  d=2, known **non-uniform, correlated** p₀ over all 16 X₀ (random simplex), η=1,
  **hand-built asymmetric** row-stochastic 4×4 W (random rows — a symmetric/gaussian W
  would let a W-transpose bug pass), uniform anchor weights (log_w=0, so unstick
  factors are X₀-independent on the all-unstuck slice and drop from ∇_y log p_t — the
  slice is the a.e.-correct density), mid-t ∈ {0.3, 0.7}, y placed away from score≈0.
  Reference in float64 via scoped `jax.experimental.enable_x64()`: enumerate X₀ →
  `log p_t = logsumexp`, autograd `jax.grad`; formula with exact Bayes per-position
  marginal posteriors. Assert rtol ≤ 1e−5 (f64 identity has ~1e-12 headroom).
  **Mutation checks**: the same gate must FAIL under `kernel.T` and under
  `kernel=I`. **Committed variant**: one site clamped to an anchor, enumeration
  restricted to consistent X₀, autograd w.r.t. uncommitted coords only, formula with
  the delta override — rtol 1e-5. **Production cross-checks** (float32 code path):
  feed `classifier_induced_score` the exact posteriors as logits — rtol 1e-4/atol 1e-5
  (fp32 cast at corruption.py:447-450 makes 1e-5 marginal there; the spec's 1e-5 gate
  is the f64 formula-vs-autograd identity above); plus a 5-D layout equivalence case
  `(B,2,2,1,d)` vs `(B,4,d)` (the image path is the one all 13 arms use).
- **G3 consistency**: `verify_w_fingerprint.py` — fingerprint of the kernel rebuilt
  from the run's `resolved_config.yaml` == `forward_blur_kernel_fingerprint` recorded
  in that run's fresh `offline_eval_metrics.json`, run inside the same job/backend
  (appended to the re-eval sbatch). Targets: `cifar_rowstoch_s1.5_seed1` (rowstoch) and
  the additive config (`sjd_cifar10_blur` + `normalization=additive sigma=1.5` vs the
  additive run's archived config).
- **G4 effect probe**: FID-1k, NFE 256, EMA, `cifar_rowstoch_s1.5_seed1/best/checkpoint_75000`,
  old vs new **through the same offline harness** — old = `+offline_eval.sampler_overrides.blur_score=false`
  (same checkpoint, same RNG: `fold_in(seed+12345, restored_step)` identical across the
  two arms) — record both, no interpretation. Zero-change control:
  `cifar_base_seed1/best/checkpoint_75000` both toggles → identical FID (W=I ⇒
  bit-exact path). Optional extra probe, record-only: `cifar_rowstoch_s2.5_seed0`
  (strongest finished W; the s1.5 delta may sit inside FID-1k noise).

## 7. New/updated tests (permanent, CPU-pinned by conftest.py)

- `tests/test_w_score_analytic.py` — G2 (above).
- `tests/test_reverse_sample_blur_score.py` — stub linear model (modeled on
  `test_blur_score_eval.py:135–252`), n_steps=4, both geometries (5-D `(2,2,1)` with
  `gaussian_2d_position_kernel`, 3-D `(6,)` with `gaussian_position_kernel`):
  (1) kernel=None: blur_score True vs False → identical outputs (G1 proxy);
  (2) kernel attached + blur_score=False == no kernel (bit-exact strip);
  (3) kernel ON ≠ OFF (real σ);
  (4) `kernel=eye(N)` + ON ≈ no-kernel (allclose; different code paths);
  (5) η=0.97 + kernel + ON → ValueError; with OFF → no raise;
  (6) known-sites smoke via `conditional_generate_board` (clues preserved).
- `tests/test_blur_score_units.py` — score-level η guard + committed/None arg
  validation; committed override vs hand-computed numpy reference (and ≠ the
  contaminated version); η=1 identity-kernel vs kernel=None allclose (closed form vs
  quadrature collapse); `kernel_fingerprint` determinism/shape/value sensitivity;
  `blur_score_mean` ≡ `blurred_posterior_mean` delegation; spec plumbing round-trip
  (`_base_sampler_spec_from_cfg` / `_sjd_sampler_cfg_from_dict`).

## 8. Re-evaluation matrix and ordering (spec item 6)

The 50k endpoint array stays **HELD** (confirmed: never launched; only
`endpoint_smoke.json` on base_seed0 exists). Ordering:

0. (Immediately, see §9-R0) snapshot the additive arm's surviving checkpoints.
1. Land the fix; pass G1–G3 (G4 runs as part of Phase A below).
2. **Phase A — FID-1k matrix** (new `eval_reeval_fid1k.sbatch`, cloned from
   `eval_endpoint.sbatch`; `--array=0-11`; ~3×140 s of sampling per arm):
   per arm, patched sampler, `use_run_config=true use_ema=true sample_timesteps=256
   eval.fid_num_samples=1000`, three cells:
   `checkpoint_source=best` (best-by-old-FID, 50k/75k), `checkpoint_source=final`
   (200k), and — recommended cheap extension (§10-D2) — `checkpoint_source=root
   checkpoint_step=175000` (the third surviving state). Legacy-toggle arm
   (`blur_score=false`) added for `cifar_rowstoch_s1.5_seed1` + `cifar_base_seed1`
   (= G4) and optionally s2.5_seed0. G3 verifier appended in-job. W=I base arms are
   live zero-change controls.
   Noise note: cells across checkpoints use different `fold_in(..., restored_step)`
   RNG; old-vs-new at the SAME checkpoint is exactly seed-matched. FID-1k noise is
   comparable to within-arm checkpoint gaps (~0.3–1.5) — report paired deltas; §10-D3
   offers an N=5k selection option.
3. **Phase B — re-selection**: per arm, min eval/fid over the evaluated cells →
   `reeval_selection.json` (source + step). Never compare against the logged
   in-training FID trajectory (different RNG derivation/harness).
4. **Phase C — 50k endpoint array** on re-selected checkpoints only:
   `eval_endpoint_reselected.sbatch` = eval_endpoint.sbatch + per-arm
   `checkpoint_source`/`checkpoint_step` read from `reeval_selection.json`
   (the current script hardcodes `checkpoint_source=best`). NFE 256 primary,
   NFE 128 secondary, unchanged otherwise. Do NOT mutate `best/` dirs or
   `best_metric.json` (keeps `collect_results.py` and provenance intact).
5. **Phase D — additive arm** (`cifar_additive_s1.5_seed0`, ~110k/200k): do NOT restart
   training (its in-training selection keeps using the old sampler — acceptable; the
   post-hoc re-selection covers whatever is retained). After it finishes: Phases A→C
   with `--array=12`, including the legacy-toggle arm for the headline before/after
   delta. Do not eval its `best/` while the trainer can overwrite it (eval after
   completion, or eval the §9-R0 snapshots).
6. **Sudoku / Text8** — enumerated implications, NOT scheduled (decision pending
   review): no trained W-runs exist, so there is nothing to re-evaluate. Forward-looking:
   future `sjd_sudoku_wsweep` (η=1) predictor_only columns become W-aware automatically;
   its policy `wscore` column is already correct; `sjd_sudoku_blur` (η=0.97) stays in
   the excluded η<1∧W≠I set (predictor_only eval will fail loud with an escape hatch,
   §4.6) until the quadrature generalization; Text8 has no W configs; the untrained OWT
   blur config (gaussian_1d, η=1) would be fully covered by this fix.

## 9. Risks / unknowns (spec item 7)

- **R0 (time-sensitive) — additive-arm evidence destruction.** The arm trains now with
  `checkpoint_keep: 2`: periodic ckpts are pruned as it advances and `best/` (75k,
  FID 295, selected by the broken metric) is overwritten whenever the broken metric
  improves. Resolution: immediately after plan approval, `cp -r` its current
  `checkpoints/{best,checkpoint_*}` to a read-only `checkpoints_snapshot_<date>/`
  (~2–5 GiB). This is the only action proposed before the fix lands.
- **R1 — campaign Gate 4 inverts.** `run_gates.py` asserts the sampler contains no blur
  refs; fails by design post-fix. Resolution: update Gate 4 (§4.8) in the same change.
- **R2 — pruned checkpoints bound re-selection.** Only {best-old, 175k, 200k} survive;
  the true best step under the corrected sampler may be gone (25k–150k pruned; best-old
  itself was selected under the biased sampler). Unavoidable; affects all arms equally;
  state in the campaign writeup.
- **R3 — FID-1k selection noise** (~ within-arm gaps). Mitigations: paired same-seed
  old/new deltas; optional N=5k selection metric (§10-D3); endpoint 50k is the final
  arbiter.
- **R4 — fingerprint is backend-sensitive** (softmax bytes differ CPU/GPU). G3 compares
  within one job/backend; config tuple + shape recorded for cross-platform sanity. The
  guarantee is deterministic-builder + config equality (training array never persisted).
- **R5 — v-floor consistency.** New branch uses v = max(σ², std_floor²=1e-6), the exact
  η=1 quadrature limit — deliberately NOT board_sampling's 1e-12 floor; difference
  matters only at t→0. Documented in the branch docstring.
- **R6 — η-override interaction.** `offline_eval.jump_eta` mutates the config before
  task construction, so kernel and η stay consistent and the guard fires correctly;
  η-sweeps on W arms now require `sampler_overrides: {blur_score: false}` — documented.
- **R7 — Sudoku sampler-kind semantics change** for future W-runs (silently-biased →
  W-aware at η=1 / fail-loud at η≠1). Deliberate; escape hatch plumbed (§4.6); no
  existing run affected. CSV provenance via `blur_score_effective` column.
- **R8 — GPU TF32.** The new einsum is precision-sensitive; bit-exactness gates run
  CPU-pinned (tests) or with `JAX_DEFAULT_MATMUL_PRECISION=float32` (G1 on GPU).
  No gate requires W=I-as-matrix bitwise equality.
- **R9 — latent config breakage** (pre-existing, noted only): `eval_checkpoint.yaml`
  default `override eval: sudoku_sjd_report` references a non-existent file; CIFAR jobs
  pass `eval=cifar10` explicitly and are unaffected.
- **R10 — memory/perf**: negligible — one `(1024,1024)×(B,1024,3,8)` einsum per NFE
  step; the closed form replaces a 32-node scan on W-runs (slightly cheaper).

## 10. Decision points for the reviewer (defaults chosen, flag to override)

- **D1 — file boundary**: the score lives in `corruption.py`, the same *file* as the
  forward corruption the spec says not to touch. Default: edit only
  `classifier_induced_score` in that file (single canonical score used by CIFAR and
  Sudoku predictor paths; `sample_pair`/`mixture_logpdf*` verifiably untouched by the
  diff). Strict-reading alternative: implement the branch in `sampler.py`'s
  `_predictive_stats` and leave corruption.py untouched — rejected by default because it
  leaves the public score API silently biased for other callers. Also within this rule:
  the `gaussian_2d.yaml` stale-comment update (§4.8) is a comment-only config edit —
  drop it if "no configs" is meant literally.
- **D2 — 175k in the re-selection set**: spec says best+final; default adds
  `checkpoint_175000` (third surviving state, one extra FID-1k per arm, ~140 s).
- **D3 — selection-metric sample count**: spec says N=1k (default). Option: N=5k for
  the Phase-A selection metric only (~700 s/cell) to cut selection noise.
- **D4 — deliverable location**: default repo root
  (`sticky-diffusion/PLAN_sampler_W_score.md`); alternative: campaign dir.

## 11. Verification summary (how the whole change is proven)

1. `pytest tests/` (new + existing; CPU-pinned) — unit + analytic + regression.
2. G1 driver on `cifar_base_seed1` (old vs new code, bit-equal `k_filled`).
3. Phase A sbatch → G3 verifier + G4 numbers recorded; base arms zero-change.
4. Phase B/C only after G1–G4 pass; Phase D after the additive arm finishes.
