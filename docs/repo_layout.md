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
- `cluster/`
  - Training launchers: `anvil_cadd_cifar10_train.sbatch`,
    `anvil_imagenet64_train.sbatch`, `anvil_sudoku_train.sbatch`, and
    `gautschi_imagenet64_train.sbatch`.
  - Environment helpers: `anvil_python_env.sh`, `gauss_python_env.sh`,
    `gautschi_python_env.sh`, `debug_python_env.sh`, and their Anvil debug
    launcher.
  - Submission entry points: `submit_anvil.sh`, `submit_imagenet64_all.sh`,
    and `submit_gautschi_imagenet64_all.sh`.
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
