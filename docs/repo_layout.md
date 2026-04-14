# Repository Layout

This repository keeps the Python package namespace as `sticky` and uses a
single canonical source/config tree.

## Top Level

- `README.md`, `LICENSE`, `environment.yml`, `pyproject.toml`
- `config/`
  - Hydra root grouped by concern: `dataset/`, `model/`, `sampler/`,
    `training/`, `eval/`, `runtime/`, `offline_eval/`, and `experiment/`.
  - Experiments are grouped under `experiment/cifar10/`, `experiment/imagenet64/`,
    and `experiment/sudoku/`.
  - Models are grouped under `model/backbone/`, `model/baseline/`, and `model/sjd/`.
- `docs/`: repository, config, dataset, and baseline-extension notes.
- `cluster/`
  - Shell, launcher, and cluster scripts.
  - `.sbatch` launchers stay here.
- `tools/`: one-off Python utilities, reporting helpers, and data-prep scripts.
- `src/sticky/`
  - `cli/`: canonical train/eval entrypoints.
  - `core/`: shared path, metrics, sampling-loop, and runtime helpers.
  - `data/`: dataset loaders and iterator packages, including `data/sudoku/`.
  - `tasks/`: task adapters and task factory.
  - `models/`
    - `backbones/`: shared sequence/image backbone implementations.
    - `common/`: shared model helpers and math utilities.
    - `baselines/`: baseline diffusion families grouped by family.
    - `sjd/`: sticky-jump-diffusion code.
  - `training/`, `eval/`.

## Naming Conventions

- Use canonical import paths only.
- Use grouped Hydra experiment paths such as `experiment=sudoku/...` and
  `experiment=cifar10/...` and `experiment=imagenet64/...`.
- Keep `sticky.models.sjd` first-class.
- Keep `sticky.models.backbones`, `sticky.models.common`, and
  `sticky.models.baselines.*` as the canonical code locations.

## Notes

- `config/` remains the Hydra root.
- Legacy default filesystem paths that embed `sticky-diffusion` remain in place
  where changing them would alter runtime behavior.
