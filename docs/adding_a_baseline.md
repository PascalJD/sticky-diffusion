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
4. Add model config under the grouped config tree.
5. Add a focused smoke test plus any characterization tests for behavior-
   sensitive paths.

## Stability Rules

- Do not rename existing task names or model names inside configs.
- Keep checkpoint format and sampler behavior stable unless the change is an
  intentional research result.
