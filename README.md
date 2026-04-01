# Sticky Jump Diffusions and Discrete Diffusion Baselines

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

Train (defaults to the paper-faithful TrainForWorst MDLM Sudoku setup):
```bash
python -m sticky.cli.train
```

Train SJD on Sudoku:
```bash
python -m sticky.cli.train experiment=sudoku/sjd_sudoku eval=sjd_sudoku
```

Train MD4 on ImageNet64:
```bash
python -m sticky.cli.train experiment=imagenet64/md4_imagenet64 eval=imagenet64
```

Train the paper-faithful TrainForWorst MDLM Sudoku setup:
```bash
python -m sticky.cli.train experiment=sudoku/mdlm_sudoku_tfw_top_prob_margin eval=sudoku_mdlm
```

Train vanilla MDLM on Sudoku with uniform reveal order:
```bash
python -m sticky.cli.train experiment=sudoku/mdlm_sudoku_uniform eval=sudoku_mdlm
```

Train vanilla MDLM on Sudoku with top-probability-margin reveal order:
```bash
python -m sticky.cli.train experiment=sudoku/mdlm_sudoku_top_prob_margin eval=sudoku_mdlm
```

Launch the default Sudoku training job on Anvil:
```bash
sbatch scripts/anvil_sudoku_train.sbatch
```

The Sudoku Anvil launcher enables Weights & Biases by default and now defaults
to the paper-faithful `mdlm_sudoku_tfw_top_prob_margin` experiment; override
`EXPERIMENT=...` or `WANDB_ENABLED=false` if needed.

Offline Sudoku checkpoint evaluation:
```bash
python -m sticky.cli.eval_checkpoint \
  experiment=sudoku/sjd_sudoku \
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
- `dataset.seq_order=dataset` means “use the token order stored in the `.npy` file”, i.e. the solver-decomposed / dataset order.
- The `sjd_sudoku` preset uses tuned defaults: `3` layers, `12` heads, `feature_dim=32`, `anchor.dim=64`, `batch_size=256`, `learning_rate=3e-4` (warmup `4000`), `grad_clip_norm=1.0`, jump `eta=0.6`, `logit_temperature=0.8`, and `50` reverse sampling steps.
- The Sudoku MDLM presets use a non-causal GPT-2-like sequence backbone with `3` layers, `12` heads, model dim `384`, MLP hidden dim `1792`, dropout `0.1`, `time_features=none`, `noise_schedule_type=loglinear`, and `50` reverse diffusion steps.
- The paper-faithful `mdlm_sudoku_tfw_top_prob_margin` preset matches TrainForWorst Appendix D.2 more closely: `batch_size=128`, `learning_rate=1e-3`, `300` epochs derived from the actual train-set size, `sampler.method=top_prob_margin`, `sampling_grid=loglinear`, and oracle Gumbel noise `0.5`.
- Sudoku checkpoint selection now tracks the strict solve-rate metric (`eval/solve_rate`), which requires exact board reconstruction rather than only row/column/box validity. Best-checkpoint updates can refresh on equal metrics so long zero-solve warm-up phases do not pin `best/` to the first evaluation forever.
- Sudoku evaluation logs reverse-process diagnostics including the mean masked-unknown count per step, mean reveal count per step, mean selected top-probability margin, the final masked-unknown fraction before decode, and which checkpoint source was evaluated.
- In SJD sampling, `alloc_mode` now defaults to `sample`; `score_scale` controls the reverse score strength, `logit_temperature` only affects jump-time anchor allocation, and the default end cleanup is a forced final plug-in jump on the last positive slice rather than a separate `t=0` classifier projection.
- Same-checkpoint sampler comparison:
```bash
scripts/compare_sudoku_mdlm_samplers.sh /absolute/path/to/run best
```
- Tiny overfit bring-up configs:
```bash
python -m sticky.cli.train experiment=sudoku/mdlm_sudoku_overfit_512 eval=sudoku_mdlm
python -m sticky.cli.train experiment=sudoku/mdlm_sudoku_overfit_2048 eval=sudoku_mdlm
```
- Optional manual prefetch:
```bash
python tools/prepare_sudoku_data.py --data-dir data/sudoku
```

Offline checkpoint evaluation (FID/IS):
```bash
python -m sticky.cli.eval_checkpoint \
  experiment=cifar10/md4_cifar10 \
  offline_eval.run_dir=outputs/2026-02-24/12-00-00_md4_cifar10_md4_md4_cifar10 \
  offline_eval.checkpoint_source=best \
  eval.fid_enabled=true \
  eval.is_enabled=true \
  wandb.enabled=false
```

Offline ImageNet64 checkpoint evaluation (FID/IS):
```bash
python -m sticky.cli.eval_checkpoint \
  experiment=imagenet64/md4_imagenet64 \
  eval=imagenet64_report \
  offline_eval.run_dir=/absolute/path/to/run \
  offline_eval.checkpoint_source=best \
  eval.fid_tfds_data_dir=/absolute/path/to/tfds \
  eval.fid_cache_dir=/absolute/path/to/fid_stats \
  wandb.enabled=false
```

Launch the ImageNet64 offline-report flow on Anvil:
```bash
BEST_CHECKPOINT_DIR=/absolute/path/to/checkpoints/best \
sbatch scripts/anvil_imagenet64_report.sbatch
```

Evaluate periodic checkpoint at a specific step:
```bash
python -m sticky.cli.eval_checkpoint \
  experiment=cifar10/sjd_cifar10 \
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

Anvil SJD temperature sweep:
```bash
sbatch scripts/anvil_sjd_logit_temperature_sweep.sbatch
```

The sweep script evaluates `logit_temperature` over `0.70, 0.75, 0.80, 0.85, 0.90, 0.95` with `offline_eval.jump_eta=0.8`, `eval.fid_num_samples=10000`, and `eval.is_enabled=false`. It defaults to the SJD checkpoint / FID stats / TFDS paths used on Anvil, auto-reuses `run_context.json` when available, and writes one `offline_eval_metrics.json` per temperature under `/home/x-pjutrasdube/scratch/sticky-diffusion/evals`.

To rank completed runs:
```bash
python tools/collect_sjd_temperature_sweep.py \
  /home/x-pjutrasdube/scratch/sticky-diffusion/evals/sjd_logit_temperature_sweep_<jobid>
```

See [docs/repo_layout.md](docs/repo_layout.md) for the repository layout and
[docs/configs.md](docs/configs.md) for the Hydra layout,
[docs/datasets.md](docs/datasets.md) for dataset notes, and
[docs/adding_a_baseline.md](docs/adding_a_baseline.md) for the
baseline-extension conventions used in the canonical tree.

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
