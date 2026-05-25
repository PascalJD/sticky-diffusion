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
2. Create `src/sticky/models/factories/<new_family>.py` with `build_model(cfg,
   *, data_shape, vocab_size)` and optionally an init function decorated with
   `@register_init("<new_family>")`.
3. Add the new family to imports and `MODEL_BUILDERS` dict in
   `src/sticky/models/factories/__init__.py`.
4. Add or update the task mapping in `src/sticky/tasks/factory.py` if the
   baseline needs a new task preset.
5. Add model and sampler configs under the grouped config tree.
6. Add a grouped experiment config that composes the dataset, model, optimizer,
   sampler, training, and runtime entries.
7. Update `EXPECTED_FAMILIES` in `tests/test_registry_smoke.py` if adding an
   init function.
8. Add a focused smoke test plus any characterization tests for behavior-
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
