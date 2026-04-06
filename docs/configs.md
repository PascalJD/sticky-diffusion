# Config Layout

`config/` is the Hydra root.

## Grouped Configs

- `config/experiment/cifar10/`
- `config/experiment/imagenet64/`
- `config/experiment/sudoku/`
- `config/model/backbone/`
- `config/model/baseline/`
- `config/model/sjd/`

Use grouped experiment paths directly, for example:

- `experiment=cifar10/md4_cifar10`
- `experiment=cifar10/sjd_cifar10`
- `experiment=imagenet64/md4_imagenet64`
- `experiment=imagenet64/sjd_imagenet64`
- `experiment=sudoku/mdlm_sudoku`
- `experiment=sudoku/sjd_sudoku`

Direct model overrides should use grouped paths too, such as
`model/backbone@...`, `model/anchor@...`, or `model/cadd_latent@...`.

`config/forward/` stays in place because it is SJD-specific and already fits the
current training/eval flow.
