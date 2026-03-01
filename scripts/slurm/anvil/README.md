# ANVIL Slurm Scripts (CADD + MD4 Baseline)

This folder contains one clean workflow for Anvil:
- allocate 1 node with 2 GPUs on `ai`
- train **CADD first**, then **MD4** in order
- enforce the shared 114M-ish ADM UNet baseline
- disable augmentation, FID/IS, and CADD corrector

By default, submit uses two Slurm jobs with dependency:
- CADD job runs first
- MD4 job is submitted with `afterok:<cadd_jobid>`

This avoids long single-job walltime limits on Anvil.

## Submit

```bash
ACCOUNT=<your-allocation> \
CONDA_ENV=sticky \
bash scripts/slurm/anvil/submit_cadd_md4_sequential.sh
```

`CONSTRAINT` is optional and unset by default. Only set it if you have a confirmed
valid feature string on your Anvil allocation.

## Defaults enforced by the job

- architecture:
  - `adm_unet5d`
  - `feature_dim=96`
  - `ch_mult=[3,4,4]`
  - `adm_num_res_blocks=4`
  - `adm_attention_resolutions=[2,4]`
  - `adm_num_heads=4`
  - `adm_num_head_channels=64`
  - `adm_num_heads_upsample=-1`
  - `adm_conv_resample=true`
  - `adm_use_scale_shift_norm=true`
  - `adm_resblock_updown=false`
  - `adm_use_conv_skip=false`
- optimization/training:
  - AdamW, `lr=1e-4`, `warmup_steps=100`, `b2=0.99`, `weight_decay=0.01`
  - `num_train_steps=500000`
  - `batch_size=256` (default for both CADD and MD4)
- logging/sampling:
  - `log_every_steps=1000`
  - `log_images_every_steps=25000`
  - `checkpoint_every_steps=10000`
  - `save_final_checkpoint=true`
  - `model.timesteps=512`
  - `training.sample_timesteps=512`
  - `eval.enabled=false` (no FID/IS)
- other:
  - `wandb.enabled=true`
  - `dataset.augment.enabled=false`
  - CADD corrector disabled (`corrector_enabled=false`, `corrector_steps=0`)

## Common overrides

```bash
# Job resources (split mode, default)
TIME_LIMIT_CADD=24:00:00 \
TIME_LIMIT_MD4=24:00:00 \
CPUS_PER_TASK=48 \
MEMORY=480G \
bash scripts/slurm/anvil/submit_cadd_md4_sequential.sh

# Force single-allocation mode (legacy behavior)
SPLIT_JOBS=0 \
TIME_LIMIT=24:00:00 \
bash scripts/slurm/anvil/submit_cadd_md4_sequential.sh

# Paths
DATA_DIR=/anvil/scratch/$USER/sticky-diffusion/data/cifar10 \
OUTPUT_ROOT=/anvil/scratch/$USER/sticky-diffusion/outputs/baselines \
RUN_TAG=my_baseline_run \
bash scripts/slurm/anvil/submit_cadd_md4_sequential.sh

# Extra Hydra overrides (space-separated)
EXTRA_OVERRIDES_COMMON="experiment.training.seed=1" \
EXTRA_OVERRIDES_CADD="experiment.model.dropout_rate=0.0" \
EXTRA_OVERRIDES_MD4="experiment.model.dropout_rate=0.0" \
bash scripts/slurm/anvil/submit_cadd_md4_sequential.sh

# Per-phase batch sizes
BATCH_SIZE_CADD=256 \
BATCH_SIZE_MD4=256 \
bash scripts/slurm/anvil/submit_cadd_md4_sequential.sh
```
