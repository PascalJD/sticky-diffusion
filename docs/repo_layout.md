# Repository Layout

This repository keeps the Python package namespace as `sticky` and uses a
single canonical source/config tree.

## Top Level

- `README.md`, `LICENSE`, `environment.yml`, `pyproject.toml`
- `config/`
  - Hydra root grouped by concern: `dataset/`, `model/`, `sampler/`,
    `training/`, `eval/`, `runtime/`, `offline_eval/`, and `experiment/`.
  - Experiments are grouped under `experiment/cifar10/`, `experiment/imagenet64/`,
    `experiment/openwebtext/`, and `experiment/sudoku/`.
  - Models are grouped under `model/backbone/`, `model/baseline/`, and `model/sjd/`.
- `docs/`: repository, config, dataset, and baseline-extension notes.
- `tests/`: Smoke tests for SJD loss, corruption sampler, and Hydra config resolution.
- `tools/`
  - `prepare_imagenet64.py`: ImageNet 64x64 TFDS-cache builder.
  - `prepare_openwebtext.py`: OpenWebText preprocessing.
  - `prepare_sudoku_data.py`: Sudoku preprocessing.
  - `extract_gpt2_embeddings.py`: exports the GPT-2 token embedding table for
    the OpenWebText SJD anchor config.
- `src/sticky/`
  - `cli/`: canonical train/eval entrypoints.
  - `core/`: shared path, metrics, sampling-loop, and runtime helpers.
  - `data/`: dataset loaders and iterator packages, including `data/sudoku/`.
  - `tasks/`: task adapters, SJDTaskBase (shared loss_fn), and task factory.
  - `models/`
    - `_registry.py`: init function registry for per-family initialization.
    - `backbones/`: shared sequence/image backbone implementations.
    - `common/`: shared model helpers and math utilities.
    - `baselines/`: baseline diffusion families grouped by family.
    - `factories/`: per-family `build_model` and init functions; `_helpers.py`
      for shared factory utilities.
    - `sjd/`: sticky-jump-diffusion code, including `schedule.py`
      (ForwardSchedule for bundled forward design axes).
  - `training/`, `eval/`.
- `tests/`: smoke tests for model initialization and config resolution.

## Naming Conventions

- Use canonical import paths only.
- Use grouped Hydra experiment paths such as `experiment=sudoku/...` and
  `experiment=cifar10/...` and `experiment=imagenet64/...`.
- Keep `sticky.models.sjd` first-class.
- Keep `sticky.models.backbones`, `sticky.models.common`, and
  `sticky.models.baselines.*` as the canonical code locations.
- Model factory code lives in `sticky.models.factories.<family>` per family.
- Init functions are registered via `@register_init("<family>")` decorator in
  factories.

## Notes

- `config/` remains the Hydra root.
- Legacy default filesystem paths that embed `sticky-diffusion` remain in place
  where changing them would alter runtime behavior.
