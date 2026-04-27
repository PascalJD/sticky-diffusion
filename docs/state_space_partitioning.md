# State-space partitioning for SJD models

The standard SJD setup treats the state as a single tensor of *anchored* sites:
every site participates in the SJD forward kernel (paired corruption with the
unstick / commit dynamics) and contributes to the cross-entropy NLL on the
classifier output. Recent work (the constraint-slack augmentation for Sudoku)
adds a second class of site that has *no anchor table* and follows a pure
VP-SDE; the joint state is the concatenation of an anchored axis and one or
more unanchored axes. This document describes the abstraction the codebase
uses to express that partitioning, and the dynamics dial it surfaces.

---

## The two-axis-class framing

A `StateLayout` (in [`models/sjd/state_layout.py`](../src/sticky/models/sjd/state_layout.py))
is an ordered tuple of `AxisSpec`s. Each axis declares:

| Field | Meaning |
|---|---|
| `name` | dict key in the state, prefix in metric names |
| `site_count` | number of sites (e.g., 81 for cells, 9 for row slacks) |
| `embedding_dim` | per-site continuous-state width (must match `vocab_size` for anchored axes so the per-anchor logit head aligns) |
| `anchor_table_name` | name of the anchor family, or `None` for unanchored |
| `dynamics` | `"sjd"` / `"vp"` / `"deterministic"` (see below) |
| `contributes_to_nll` | whether the loss includes this axis's NLL |

Two reference layouts ship with the code:

- `SUDOKU_SLACK_LAYOUT` — `(cells, row_slacks, col_slacks, box_slacks)`, one
  anchored + three unanchored axes; total length 108.
- `SUDOKU_CELL_ONLY_LAYOUT` — single `cells` axis; total length 81.

Other tasks (CIFAR / ImageNet64 / OpenWebText) currently use the legacy
single-tensor SJD path; wrapping them in a one-axis `StateLayout` is
mechanical but intentionally out of scope for the PR that introduced this
abstraction. See "Migration status" below.

---

## How the layout flows through the model

```
state: dict[axis_name -> (B, site_count, embedding_dim)]
        │
        ▼
[MultiAxisInputProj]   per-axis Dense(feature_dim) + per-axis site-type emb
        │
        ▼
[SudokuStructuralAdapter]   (optional, init-zero)  row/col/box for cells,
        │                                           group_idx for slacks
        ▼
concat in declared order ─► (B, total_site_count, feature_dim)
        │
        ▼
[ContinuousClassifier]   sequence-mode backbone (GPT-2-like) + lm_head
        │
        ▼
(B, total_site_count, vocab_size)
        │
        ▼
slice per anchored axis ─► {axis_name -> (B, site_count, vocab_size)}
```

The slicing is driven by `StateLayout.slice_of(name)`, so anchored-axis
indices are derived from the layout's declared axis order rather than
hard-coded. Unanchored axes still pass through the classifier (so their
representations contribute to attention over neighboring cells) but their
logits are discarded.

`SJD.apply_layout(state, t)` (see [`sjd_model.py`](../src/sticky/models/sjd/sjd_model.py))
is the entry point. It returns `(per_axis_logits, aux)` where
`per_axis_logits` only contains anchored axes. The legacy
`SJD.__call__(y_t, t, slack_y_t=...)` path is preserved for backward
compatibility — the slack task still uses it (see "Migration status").

---

## Dynamics dial

`AxisSpec.dynamics` selects the forward law for that axis. The loss
([`losses_multi_axis.py`](../src/sticky/models/sjd/losses_multi_axis.py))
dispatches on this field per axis:

| Setting | Forward law | NLL? | Status |
|---|---|---|---|
| `"sjd"` | `sample_pair` (paired SJD corruption) | yes | implemented |
| `"vp"` | `sample_slack_pair` (pure VP perturbation) | no | implemented |
| `"deterministic"` | value at `t > 0` is computed from other axes via a per-layout callback | no | **stubbed** — raises `NotImplementedError` |

Per the current loss, `contributes_to_nll=True` is only legal with
`dynamics="sjd"`; this is enforced at `AxisSpec` construction time.

### What `"vp"` does and doesn't give you

Under `"vp"`, the slack-axis forward law is independent of the cell-axis
forward law given `t`. The joint marginal `p^ac(y_cells, y_slacks | a_p)`
factors into `p^ac_cells(y_cells | a_p) * p^vp_slacks(y_slacks)`, which means
the per-cell DHM spatial factor `r_a(y_p) / p^ac(y_p | a_p)` does **not**
literally depend on the slack — the constraint signal flows into the per-cell
commit rate `Λ*_plug(a | y) = λ̂(y, a) · P_θ(a | y)` only through `P_θ`. The
λ̂ side is unchanged from the cell-only model.

This is genuinely useful: a constraint-aware classifier, plugged into
`Λ*_plug`, biases the *plug-in rate* toward feasible commits. Empirically the
plan is to confirm this gives meaningful gains over the cell-only baseline.
But the stronger reading — "the spatial factor literally encodes the
constraint penalty" — is **not** delivered by `"vp"` slack dynamics.

### What `"deterministic"` would give you

If slacks are deterministically tied to cells throughout the trajectory
(e.g., `s_G(t) = sum_{c ∈ G} cell_t[c]` enforced at every `t`, not just
inference time), then `p^ac(y_cells, y_slacks | a_p)` no longer factors and
the per-cell spatial factor `r_a/p^ac` literally depends on slack. This is
the framing the original motivation pointed at. The implementation needs a
per-layout callback that draws `(y_p, Δs_R, Δs_C, Δs_B)` jointly when cell
`p` unsticks at `τ` and commits to `e_v`, with slack increments correlated
to `y_p − α(τ) e_v`.

This is the **Phase D** design summarized in the TODO at the top of
[`slack_corruption.py`](../src/sticky/models/sjd/slack_corruption.py). The
`dynamics="deterministic"` enum value is reserved for that mode; the loss
currently raises `NotImplementedError` on it so a layout that opts in fails
loudly rather than silently using `"vp"`.

### Inference-time slack projection (the cheap alternative)

A weaker but cheap approximation: leave the training law as `"vp"`, and at
inference time overwrite the slack state after each reverse-time predictor
step with the deterministic group-sum readout
`s_G = sum_{c ∈ G} cell_state_c`. This re-ties the slack to the cell state
between Euler-Maruyama steps, so the classifier sees a constraint-consistent
slack throughout sampling, even though training dynamics didn't enforce it.

The helper `compute_slack_from_cells` in
[`data/sudoku/slack.py`](../src/sticky/data/sudoku/slack.py) is the readout.
Wiring it into the slack-aware sampler is Phase 4 of the broader plan; until
then, the helper is a unit-tested standalone.

---

## Migration status

| Component | Path | Status |
|---|---|---|
| Anchored axis state passed as tensor (`y_t`) | legacy `SJD.__call__` | retained |
| Anchored + unanchored as state dict | `SJD.apply_layout` | new (this PR) |
| Cell-only Sudoku task | `SudokuInpaintSJDTask` | unchanged (legacy `__call__`) |
| Slack-augmented Sudoku task | `SudokuInpaintSJDSlackTask` | unchanged (legacy `SudokuJointInputProj`) |
| CIFAR / ImageNet64 / OpenWebText | their respective tasks | unchanged (legacy `__call__`) |

The decision to **not** migrate `SudokuInpaintSJDSlackTask` to the new
abstraction in this PR is deliberate. The legacy path uses a single shared
`slack_proj` Dense across all 27 slack sites; the new multi-axis layout
splits them into three per-axis Denses (`row_slacks_proj`, `col_slacks_proj`,
`box_slacks_proj`). The two are not parameter-tree-compatible, which means
strict bit-equivalence at the same RNG seed is not achievable across the
migration boundary. Doing the migration as a separate PR keeps the
"structural embeddings init-zero ⇒ no forward change at init" claim honest:
within the legacy `SudokuJointInputProj`, that claim holds today; the new
multi-axis path is verified independently by its own test suite.

The migration itself is small once we're ready to take the bit-equivalence
hit:

1. `SudokuInpaintSJDSlackTask.loss_fn`: build a state dict
   `{"cells": x0_anchor, "row_slacks": slack_x0[:, :9], "col_slacks":
   slack_x0[:, 9:18], "box_slacks": slack_x0[:, 18:27]}` and call
   `model.apply(..., state=state)` instead of `slack_y_t=...`.
2. Swap `ce_allocation_loss_with_slack` for `ce_allocation_loss_multi_axis`.
3. Plumb a layout name through Hydra (e.g., `model.state_layout: sudoku_slack`)
   and resolve it to `SUDOKU_SLACK_LAYOUT` in the SJD factory.
4. Set `model.use_sudoku_structural: true`.

After migration, `SudokuJointInputProj` and `ce_allocation_loss_with_slack`
become dead code and can be deleted.

---

## Pointer index

- Layout: [`src/sticky/models/sjd/state_layout.py`](../src/sticky/models/sjd/state_layout.py)
- Generic input projection: [`src/sticky/models/sjd/multi_axis_input.py`](../src/sticky/models/sjd/multi_axis_input.py)
- Sudoku structural adapter: [`src/sticky/models/sjd/sudoku_structural.py`](../src/sticky/models/sjd/sudoku_structural.py)
- Multi-axis loss: [`src/sticky/models/sjd/losses_multi_axis.py`](../src/sticky/models/sjd/losses_multi_axis.py)
- SJD module entry: [`src/sticky/models/sjd/sjd_model.py`](../src/sticky/models/sjd/sjd_model.py) (`apply_layout`)
- Slack readout helper: [`src/sticky/data/sudoku/slack.py`](../src/sticky/data/sudoku/slack.py) (`compute_slack_from_cells`)
- Phase D TODO: [`src/sticky/models/sjd/slack_corruption.py`](../src/sticky/models/sjd/slack_corruption.py) (top-of-file docstring)
