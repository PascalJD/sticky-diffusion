# Dataset Notes

## Canonical dataset configs

- `config/dataset/cifar10_discrete.yaml`
- `config/dataset/imagenet64_discrete.yaml`
- `config/dataset/openwebtext_gpt2_1024.yaml`
- `config/dataset/sudoku.yaml`

TFDS-backed image dataset configs store their builder name in `dataset.tfds_name`
and split policy in `dataset.train_split` / `dataset.eval_split`.

`imagenet64_discrete` starts with the unconditional
`downsampled_imagenet/64x64` benchmark path and keeps `include_label: false` so
the task code remains unconditional while still using the generic TFDS image
loader.

OpenWebText uses GPT-2 tokenization with sequence length 1024. Prepare local
text shards with `tools/prepare_openwebtext.py` before launching the OWT
experiments.

## Sudoku data layout

The Sudoku data implementation lives under `sticky.data.sudoku.*`.

Sudoku training still auto-downloads missing files when
`dataset.auto_download=true`, and the default scratch-path behavior remains
unchanged at `$SCRATCH/sticky-diffusion/data/sudoku` when `dataset.data_dir`
keeps its default value.

For an explicit local preparation step, use `tools/prepare_sudoku_data.py`.
