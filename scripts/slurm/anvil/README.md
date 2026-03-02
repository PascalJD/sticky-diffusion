# Anvil Slurm Launchers

Fresh layout with one job = one model.

## Files

- `submit_train.sh`: generic submit entrypoint for any model.
- `train_model.slurm`: generic runtime script that executes one training run.
- `submit_cadd.sh`: CADD wrapper with paper-style defaults.
- `submit_md4.sh`: MD4 wrapper with paper-style defaults.

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

## Notes

- The runtime script performs a JAX device preflight when `PLATFORM=pmap` and aborts if not enough local devices are visible.
- The wrappers are thin defaults only; override any variable at submit time.
