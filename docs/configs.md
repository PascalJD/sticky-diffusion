# Config Layout

`config/` is the Hydra root. The tree exposes exactly one canonical default
per (method × task) — no aliases, no `_report` variants, no flat duplicates.

## Tree

```
config/
├── config.yaml, eval_checkpoint.yaml         (top-level entrypoints)
├── dataset/                                  (per-dataset)
├── eval/{cifar10,imagenet64,openwebtext,sudoku}/   (per-task subfolders)
├── experiment/{cifar10,imagenet64,openwebtext,sudoku}/
├── forward/
│   ├── beta/vp_linear.yaml
│   ├── hazard/{cosine_alpha, poly_alpha}.yaml      (only the two used schedules)
│   └── jump/{vp_matched, vp_matched_sudoku}.yaml
├── hydra/, runtime/, wandb/
├── model/
│   ├── anchor/, anchor/transform/
│   ├── backbone/                             (single source for backbones)
│   ├── baseline/                             (per-method × per-dataset)
│   ├── cadd_latent/
│   └── sjd/                                  (per-task SJD model)
├── offline_eval/, optim/
├── sampler/
│   ├── base.yaml                             (shared, referenced as /sampler/base@_here_)
│   ├── cifar10/                              (per-task subfolder)
│   ├── openwebtext/
│   └── sudoku/
└── training/
```

## Canonical experiments

```bash
python -m sticky.cli.train experiment=cifar10/sjd_cifar10
python -m sticky.cli.train experiment=imagenet64/sjd_imagenet64
python -m sticky.cli.train experiment=sudoku/sjd_sudoku
python -m sticky.cli.train experiment=openwebtext/sjd_openwebtext
```

Baseline experiment bundles (no SJD) live alongside:

- CIFAR-10: `bitdiff_cifar10`, `cadd_cifar10`, `candi_cifar10`,
  `d3pm_{absorb,gaussian,uniform}_cifar10`, `ddpm_cifar10`, `md4_cifar10`,
  `mdlm_cifar10`
- ImageNet64: `bitdiff_imagenet64`, `ddpm_imagenet64`, `md4_imagenet64`,
  `mdlm_imagenet64`
- OpenWebText: `md4_openwebtext`, `mdlm_openwebtext`
- Sudoku: SJD only — non-SJD Sudoku configs were removed in the cleanup pass.

## Composition pattern (SJD)

```yaml
defaults:
  - /forward/beta: vp_linear
  - /forward/hazard: poly_alpha           # or cosine_alpha
  - /forward/jump: vp_matched             # or vp_matched_sudoku
  - /sampler/<task>@sampler: sjd_<task>   # e.g. /sampler/cifar10@sampler: sjd_cifar10
  - /eval/<task>@_global_.eval: <name>    # e.g. /eval/cifar10@_global_.eval: cifar10
  - /training: sjd_<task>
  - _self_
```

The `@<group>` redirects keep each subfolder's contents loadable from the
top-level entry point. ImageNet64 experiments cross-reference
`/sampler/cifar10@sampler: <name>_cifar10` (the image-baseline samplers are
shared across CIFAR-10 and ImageNet64).

## Forward / sampler conventions

- **`forward.tau_grid_size`** is the only quadrature knob exposed; no
  canonical experiment overrides it (default `32`).
- The legacy `forward.corruption` / `forward.dhm_denom` switches are gone —
  `sample_pair` is the only training-side corruption sampler and the SJD
  off-anchor mixture is the only DHM denominator.
- All baseline samplers in `sampler/cifar10/` and `sampler/openwebtext/`
  inherit from `sampler/base.yaml` via `- /sampler/base@_here_` in their
  `defaults` list.
