# ANVIL Slurm Scripts (SJD + CADD CIFAR-10)

This folder contains a Stage-1-only sweep for SJD on CIFAR-10 (`eta x p`) using Slurm arrays.

It also includes a CADD smoke-test launcher that defaults dataset/cache, logs,
and run outputs to `scratch`.

## CADD smoke test (recommended first run)

```bash
PARTITION=gpu \
ACCOUNT=<your-allocation> \
CONDA_ENV=sticky \
bash scripts/slurm/anvil/submit_cadd_smoke.sh
```

Default smoke settings:
- single GPU (`GPUS_PER_NODE=1`, `platform=single`)
- `num_train_steps=200`
- no FID/IS (`eval.enabled=false`)
- no checkpointing (`checkpoint_every_steps=0`)
- data dir: `$SCRATCH/sticky-diffusion/data/cifar10`
- run dir: `$SCRATCH/sticky-diffusion/outputs/smoke/<run_tag>`

Useful overrides:

```bash
# 4-GPU pmap smoke test
GPUS_PER_NODE=4 \
CPUS_PER_TASK=64 \
MEMORY=0 \
PLATFORM=pmap \
BATCH_SIZE=512 \
EVAL_BATCH_SIZE=512 \
TIME_LIMIT=01:00:00 \
PARTITION=gpu \
ACCOUNT=<your-allocation> \
bash scripts/slurm/anvil/submit_cadd_smoke.sh
```

## One-command submit (Stage 1 only)

```bash
mkdir -p logs manifests

PARTITION=gpu \
ACCOUNT=<your-allocation> \
QOS=<optional-qos> \
CONDA_ENV=sticky \
bash scripts/slurm/anvil/submit_sjd_stage1_sweep.sh
```

Default Stage-1 grid:
- `eta`: `0.9,0.85,0.8,0.75`
- `p`: `0.5,1,2,3`
- fixed anchors
- `temperature=1.0`

## Baseline enforced in `train_sjd_array.slurm`

- `experiment.model.image_backbone=adm_unet5d`
- `experiment.sampler.n_steps=512`
- `experiment/forward/jump=vp_matched`
- `experiment/forward/hazard=poly_alpha`
- `experiment.optim.learning_rate=1e-4`
- `experiment.optim.b2=0.999` (`b1=0.9` fixed in code)
- `experiment.dataset.augment.enabled=false`
- `experiment.model.dropout_rate=0.1`
- Proxy FID during sweep: `eval.fid_every=50000`, `eval.fid_num_samples=10000`
- Checkpoint cadence: `experiment.training.checkpoint_every_steps=10000`

## Useful overrides

```bash
STAGE1_ETAS="0.9,0.85,0.8,0.75" \
STAGE1_P_VALUES="0.5,1,2,3" \
STAGE1_SEEDS="0,1" \
ARRAY_MAX_PARALLEL=16 \
TIME_LIMIT=24:00:00 \
bash scripts/slurm/anvil/submit_sjd_stage1_sweep.sh
```

## Manual flow

1) Generate manifest:

```bash
python scripts/slurm/anvil/generate_sjd_manifest.py \
  --output manifests/sjd_stage1_eta_p.txt
```

2) Submit array from manifest:

```bash
PARTITION=gpu \
ACCOUNT=<your-allocation> \
CONDA_ENV=sticky \
bash scripts/slurm/anvil/submit_sjd_tuning.sh manifests/sjd_stage1_eta_p.txt
```
