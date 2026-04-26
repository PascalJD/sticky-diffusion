# Adding a Baseline

This repository keeps baseline model families under `sticky.models.baselines`.

## Where New Baselines Should Live

- Python code: `src/sticky/models/baselines/<family>/`
- Tests: `tests/models/baselines/`
- Model config: `config/model/baseline/`
- Experiment config:
  - `config/experiment/cifar10/` for CIFAR-10 baselines
  - `config/experiment/sudoku/` for Sudoku baselines
  - `config/experiment/openwebtext/` for text baselines

## Recommended Checklist

1. Add the model family package under `sticky.models.baselines.<family>`.
2. Add or update the explicit builder entry in `sticky.models.factory`.
3. Add or update the task mapping if the baseline needs a new task preset.
4. Add model and sampler configs under the grouped config tree.
5. Add a grouped experiment config that composes the dataset, model, optimizer,
   sampler, training, and runtime entries.
6. Add a focused smoke test plus any characterization tests for behavior-
   sensitive paths.

## Config Examples

Use the existing grouped configs as templates:

- CIFAR-10 baseline model configs: `config/model/baseline/md4_cifar10.yaml`
  and `config/model/baseline/mdlm_cifar10.yaml`
- CIFAR-10 sampler configs: `config/sampler/md4_cifar10.yaml` and
  `config/sampler/mdlm_cifar10.yaml`
- Sudoku baseline configs: `config/model/baseline/mdlm_sudoku.yaml` with
  `config/sampler/mdlm_sudoku_base.yaml`
- OpenWebText baseline configs: `config/model/baseline/md4_openwebtext.yaml`,
  `config/model/baseline/mdlm_openwebtext.yaml`, `config/sampler/md4_openwebtext.yaml`,
  and `config/sampler/mdlm_openwebtext.yaml`

## Stability Rules

- Do not rename existing task names or model names inside configs.
- Keep checkpoint format and sampler behavior stable unless the change is an
  intentional research result.
