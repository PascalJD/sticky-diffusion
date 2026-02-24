# ANVIL Slurm Scripts (SJD CIFAR-10 Stage-1)

This folder contains a Stage-1-only sweep for SJD on CIFAR-10 (`eta x p`) using Slurm arrays.

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
