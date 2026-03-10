# Anvil Slurm Launchers

Fresh layout with one job = one model.

## Files

- `submit_train.sh`: generic submit entrypoint for any model.
- `train_model.slurm`: generic runtime script that executes one training run.
- `submit_cadd.sh`: CADD wrapper with paper-style defaults.
- `submit_md4.sh`: MD4 wrapper with paper-style defaults.
- `submit_sjd.sh`: SJD wrapper that submits the default `sjd_cifar10` experiment config.
- `submit_sjd_anchor_short_sweep.sh`: consumes a prescreen manifest and submits one short SJD anchor-study run per candidate/seed.
- `eval_anchor_checkpoint.slurm`: anchor-study-safe offline eval runtime that reuses the stored training run config.
- `submit_sjd_anchor_eval_sweep.sh`: submits one anchor-study offline eval job per candidate/seed/NFE budget.
- `eval_checkpoint.slurm`: runtime script for offline checkpoint evaluation (FID/IS + optional sampler probes).
- `submit_sjd_fid_sweep.sh`: submits an eta/tau sweep for SJD checkpoint evaluation as separate Slurm jobs.

## Recommended usage

### CADD baseline

```bash
ACCOUNT=<allocation> \
CONDA_ENV=/anvil/scratch/$USER/envs/sticky \
EXCLUDE=<optional_bad_nodes> \
RUN_TAG=cadd_baseline_$(date +%Y%m%d_%H%M%S) \
bash scripts/slurm/anvil/submit_cadd.sh
```

### MD4 baseline

```bash
ACCOUNT=<allocation> \
CONDA_ENV=/anvil/scratch/$USER/envs/sticky \
EXCLUDE=<optional_bad_nodes> \
RUN_TAG=md4_baseline_$(date +%Y%m%d_%H%M%S) \
bash scripts/slurm/anvil/submit_md4.sh
```

### SJD baseline

```bash
ACCOUNT=<allocation> \
CONDA_ENV=/anvil/scratch/$USER/envs/sticky \
EXCLUDE=<optional_bad_nodes> \
RUN_TAG=sjd_baseline_$(date +%Y%m%d_%H%M%S) \
bash scripts/slurm/anvil/submit_sjd.sh
```

### SJD anchor short sweep

```bash
ACCOUNT=<allocation> \
CONDA_ENV=/anvil/scratch/$USER/envs/sticky \
PRESCREEN_MANIFEST=/path/to/prescreen_anchors/manifest.jsonl \
RUN_TAG_PREFIX=sjd_anchor_short \
TRAIN_STEPS=50000 \
SEEDS="0 1 2" \
DRY_RUN=1 \
bash scripts/slurm/anvil/submit_sjd_anchor_short_sweep.sh
```

### SJD eta/tau FID sweep (offline checkpoint eval)

```bash
ACCOUNT=<allocation> \
PARTITION=ai \
CONDA_ENV=/anvil/scratch/$USER/envs/sticky \
CHECKPOINT_PATH=/home/$USER/scratch/sticky-diffusion/outputs/sjd_run/checkpoints/checkpoint_350000 \
ETA_VALUES="0.5 0.6 0.8 1.0" \
TAU_VALUES="0.8 1.0 1.2" \
FID_NUM_SAMPLES=10000 \
IS_ENABLED=false \
SAMPLER_PROBE_BATCHES=32 \
bash scripts/slurm/anvil/submit_sjd_fid_sweep.sh
```

Outputs for each combination are saved under:
- `<run_dir>/eval_sweeps/<sweep_tag>/eta_*__tau_*/offline_eval_metrics.json`
- `<run_dir>/eval_sweeps/<sweep_tag>/eta_*__tau_*/summary.json`
- `<run_dir>/eval_sweeps/<sweep_tag>/submit_manifest.tsv`

This path intentionally overrides the backbone and SJD hazard/jump settings for the legacy eta/tau sweep. Do not use it for the anchor study.

### SJD anchor-study offline eval sweep

```bash
ACCOUNT=<allocation> \
CONDA_ENV=/anvil/scratch/$USER/envs/sticky \
TRAINING_MANIFEST=/path/to/anchor_short_submission_manifest.tsv \
NFE_BUDGETS="64 128 256 512" \
FID_NUM_SAMPLES=10000 \
DRY_RUN=1 \
bash scripts/slurm/anvil/submit_sjd_anchor_eval_sweep.sh
```

Outputs are organized under:
- `<sweep_root>/<candidate>/seed_<seed>/nfe_<budget>/offline_eval_metrics.json`
- `<sweep_root>/<candidate>/seed_<seed>/nfe_<budget>/summary.json`
- `<sweep_root>/eval_manifest.tsv`
- `<sweep_root>/aggregate/raw_results.tsv`
- `<sweep_root>/aggregate/candidate_budget_summary.tsv`
- `<sweep_root>/aggregate/candidate_best.tsv`

## Generic usage

Run any model from one command path:

```bash
MODEL=sjd \
EXPERIMENT_CFG=sjd_cifar10 \
EVAL_CFG=sjd_cifar10 \
ACCOUNT=<allocation> \
CONDA_ENV=/anvil/scratch/$USER/envs/sticky \
TIME_LIMIT=24:00:00 \
GPUS_PER_NODE=2 \
PLATFORM=pmap \
REQUIRED_LOCAL_DEVICES=2 \
RUN_TAG=sjd_run_$(date +%Y%m%d_%H%M%S) \
EXTRA_OVERRIDES="experiment.dataset.augment.enabled=false" \
bash scripts/slurm/anvil/submit_train.sh
```

Or with one Hydra override per line:

```bash
cat > /tmp/sticky_overrides.txt <<'EOF'
experiment.model.anchor.learnable=false
experiment.model.anchor.transform.scale=0.75
EOF

MODEL=sjd \
EXPERIMENT_CFG=sjd_anchor_study_cifar10 \
EVAL_CFG=sjd_anchor_study_cifar10 \
ACCOUNT=<allocation> \
CONDA_ENV=/anvil/scratch/$USER/envs/sticky \
EXTRA_OVERRIDES_FILE=/tmp/sticky_overrides.txt \
bash scripts/slurm/anvil/submit_train.sh
```

## Key environment knobs

- `MODEL`: model key (`cadd`, `md4`, `sjd`, or custom with explicit `EXPERIMENT_CFG`/`EVAL_CFG`).
- `ACCOUNT`, `PARTITION`, `QOS`, `CONSTRAINT`, `EXCLUDE`, `NODELIST`: Slurm placement/accounting.
- `GPUS_PER_NODE`, `CPUS_PER_TASK`, `MEMORY`, `TIME_LIMIT`: Slurm resources.
- `CONDA_ENV`, `ANVIL_MODULES`: Python environment setup.
- `RUN_TAG`, `OUTPUT_ROOT`, `DATA_DIR`: output/data location.
- `PLATFORM`: `single`, `pmap`, or `auto`.
- `REQUIRED_LOCAL_DEVICES`: fail-fast JAX preflight threshold in `pmap`.
- `BATCH_SIZE`, `EVAL_BATCH_SIZE`, `TRAIN_STEPS`, `CHECKPOINT_EVERY`, `LOG_IMAGES_EVERY`.
- `WANDB_ENABLED`, `EVAL_ENABLED`, `SAVE_FINAL_CHECKPOINT`.
- `BASELINE_ARCH_114M`, `DISABLE_AUGMENT`, `DISABLE_CORRECTOR`.
- `EXTRA_OVERRIDES`: extra Hydra overrides (space-separated).
- `EXTRA_OVERRIDES_FILE`: path to a file containing one Hydra override per line; blank lines and `#` comments are ignored.
- `CHECKPOINT_PATH`, `CHECKPOINT_DIR`, `CHECKPOINT_STEP`: select the checkpoint to evaluate.
- `ETA_VALUES`, `TAU_VALUES`: whitespace-separated sweep grids for SJD `forward.jump.eta` and `sampler.logit_temperature`.
- `FID_NUM_SAMPLES`: use `10000` for quick probes, `50000` for final comparisons.
- `SAMPLER_PROBE_BATCHES`, `SAMPLER_PROBE_BATCH_SIZE`: optional SJD sampler diagnostics (state-dependency proxy metrics).
- `PRESCREEN_MANIFEST`, `RUN_TAG_PREFIX`, `SEEDS`, `TRAIN_STEPS`, `LEARNABLE_OVERRIDE`: knobs for `submit_sjd_anchor_short_sweep.sh`.
- `TRAINING_MANIFEST`, `RUN_DIR_LIST`, `NFE_BUDGETS`, `CANDIDATES`, `DEPEND_ON_TRAIN`: knobs for `submit_sjd_anchor_eval_sweep.sh`.

## Notes

- The runtime script performs a JAX device preflight when `PLATFORM=pmap` and aborts if not enough local devices are visible.
- The wrappers are thin defaults only; override any variable at submit time.
- `submit_sjd.sh` now leaves SJD hyperparameters untouched by default and only changes scheduler/resource defaults, including a `48:00:00` default time limit.
- `submit_sjd_anchor_short_sweep.sh` writes candidate override files under its submission root and records one TSV manifest row per submitted candidate/seed pair.
- `submit_sjd_anchor_eval_sweep.sh` reuses each run's stored `run_context.json` during offline eval, so hazard, jump, backbone, and anchor settings stay aligned with the training run unless you explicitly override `ETA_OVERRIDE` or `TAU_OVERRIDE`.
