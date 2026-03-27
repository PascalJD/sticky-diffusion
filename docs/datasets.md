# Dataset Notes

## Canonical dataset configs

- `config/dataset/cifar10_discrete.yaml`
- `config/dataset/openwebtext_gpt2_1024.yaml`
- `config/dataset/sudoku.yaml`

## Sudoku data layout

The Sudoku data implementation lives under `sticky.data.sudoku.*`.

Sudoku training still auto-downloads missing files when
`dataset.auto_download=true`, and the default scratch-path behavior remains
unchanged at `$SCRATCH/sticky-diffusion/data/sudoku` when `dataset.data_dir`
keeps its default value.
