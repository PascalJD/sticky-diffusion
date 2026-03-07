# Anvil Slurm Launchers

Fresh layout with one job = one model.

## Files

- `submit_train.sh`: generic submit entrypoint for any model.
- `train_model.slurm`: generic runtime script that executes one training run.
- `submit_cadd.sh`: CADD wrapper with paper-style defaults.
- `submit_md4.sh`: MD4 wrapper with paper-style defaults.
- `submit_sjd.sh`: SJD wrapper that submits the default `sjd_cifar10` experiment config.
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
- `CHECKPOINT_PATH`, `CHECKPOINT_DIR`, `CHECKPOINT_STEP`: select the checkpoint to evaluate.
- `ETA_VALUES`, `TAU_VALUES`: whitespace-separated sweep grids for SJD `forward.jump.eta` and `sampler.logit_temperature`.
- `FID_NUM_SAMPLES`: use `10000` for quick probes, `50000` for final comparisons.
- `SAMPLER_PROBE_BATCHES`, `SAMPLER_PROBE_BATCH_SIZE`: optional SJD sampler diagnostics (state-dependency proxy metrics).

## Notes

- The runtime script performs a JAX device preflight when `PLATFORM=pmap` and aborts if not enough local devices are visible.
- The wrappers are thin defaults only; override any variable at submit time.
- `submit_sjd.sh` now leaves SJD hyperparameters untouched by default and only changes scheduler/resource defaults, including a `48:00:00` default time limit.
