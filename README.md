# Sticky Jump Diffusion

**Status:** 🚧 active research

## Installation

Clone and move in
```
git clone https://github.com/PascalJD/sticky-diffusion.git
cd sticky-diffusion
```

Create the Conda environment
```
conda env create -f environment.yml
conda activate sticky
```

## Quick Start 

Train:
```bash
python -m sticky.entrypoints.train
```

Train SJD on Sudoku:
```bash
python -m sticky.entrypoints.train experiment=sjd_sudoku eval=sjd_sudoku
```

Offline Sudoku checkpoint evaluation:
```bash
python -m sticky.entrypoints.eval_checkpoint \
  experiment=sjd_sudoku \
  eval=sjd_sudoku \
  offline_eval.run_dir=/absolute/path/to/run \
  offline_eval.checkpoint_source=best \
  offline_eval.use_ema=true
```

Sudoku dataset setup:
- Training now auto-downloads missing Sudoku files from [Google Drive](https://drive.google.com/drive/folders/1TluiZjYl-zLdbxjVmhfWl-WyX_OvD7UW).
- If `SCRATCH` is set and `dataset.data_dir` is left at the default `data/sudoku`, files are downloaded to `$SCRATCH/sticky-diffusion/data/sudoku`.
- You can still override `dataset.data_dir`, `dataset.train_file`, and `dataset.test_file`.
- Supported sequence ordering is configured via `dataset.seq_order` with values: `dataset`, `fixed`, `random`.
- The `sjd_sudoku` preset uses tuned defaults: 6M GPT-2-like backbone (`3` layers, `12` heads, hidden dim `384`), `batch_size=256`, `learning_rate=3e-4` (warmup `4000`), `grad_clip_norm=1.0`, jump `eta=0.6`, `logit_temperature=0.8`, and `50` reverse sampling steps.
- In SJD sampling, `alloc_mode` now defaults to `sample`; `score_scale` controls the reverse score strength, `logit_temperature` only affects jump-time anchor allocation, and the default end cleanup is a forced final plug-in jump on the last positive slice rather than a separate `t=0` classifier projection.
- Optional manual prefetch:
```bash
python -m sticky.scripts.prepare_sudoku_data --data-dir data/sudoku
```

Offline checkpoint evaluation (FID/IS):
```bash
python -m sticky.entrypoints.eval_checkpoint \
  experiment=md4_cifar10 \
  offline_eval.run_dir=outputs/2026-02-24/12-00-00_md4_cifar10_md4_md4_cifar10 \
  offline_eval.checkpoint_source=best \
  eval.fid_enabled=true \
  eval.is_enabled=true \
  wandb.enabled=false
```

Evaluate periodic checkpoint at a specific step:
```bash
python -m sticky.entrypoints.eval_checkpoint \
  experiment=sjd_cifar10 \
  offline_eval.checkpoint_dir=/absolute/path/to/checkpoints \
  offline_eval.checkpoint_source=periodic \
  offline_eval.checkpoint_step=50000 \
  offline_eval.use_ema=true \
  eval.fid_enabled=true \
  eval.is_enabled=true \
  wandb.enabled=false
```

Notes:
- `offline_eval.checkpoint_source=best|final|periodic` maps to CheckpointWriter layout.
- `offline_eval.run_dir=...` auto-discovers checkpoint paths from `run_context.json` when available.
- JSON metrics are written to `offline_eval.output_path` (default: `offline_eval_metrics.json` in Hydra output dir).

## Contribute 

Below is the general format for commit log messages:

```
ABBR: Commit message.
```

**ABBR options:**

- `IMP`: Development and implementation of a new feature.
- `FIX`: A fix of an existing bug.
- `OPT`: Any optimization performed.
- `REF`: Any refactors to code.
