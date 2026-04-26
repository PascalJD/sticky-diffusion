# Sticky Jump Diffusions and Discrete Diffusion Baselines

**Status:** 🚧 active research

## Installation

```bash
git clone https://github.com/PascalJD/sticky-diffusion.git
cd sticky-diffusion
conda env create -f environment.yml
conda activate sticky
```

## Quickstart

The four canonical SJD experiments carry their forward, sampler, eval, and
training defaults in the experiment bundle.

```bash
python -m sticky.cli.train experiment=cifar10/sjd_cifar10
python -m sticky.cli.train experiment=imagenet64/sjd_imagenet64
python -m sticky.cli.train experiment=sudoku/sjd_sudoku
python -m sticky.cli.train experiment=openwebtext/sjd_openwebtext
```

One-line map:

- `cifar10/sjd_cifar10`: canonical CIFAR-10 discrete-image SJD.
- `imagenet64/sjd_imagenet64`: ImageNet-64 SJD with ImageNet data, model, and eval defaults.
- `sudoku/sjd_sudoku`: board-level Sudoku SJD with the compact sampler comparison eval.
- `openwebtext/sjd_openwebtext`: GPT-2-token OpenWebText SJD with lightweight text sampling eval.

## What's in the Box

- `corruption.py`: draws SJD training pairs and evaluates mixture scores for corrupted anchors.
- `convolution.py`: implements the VP-matched Gaussian convolution used by the SJD mixture.
- `hazard.py`: defines forward unstick hazards and survival functions.
- `jump.py`: implements the anchor-to-continuous unstick kernel.
- `losses.py`: contains the classifier-centered SJD training objective and diagnostics.
- `plugin_intensity.py`: converts classifier logits into reverse jump intensities and anchor allocations.
- `sampler.py`: runs the reverse SJD sampler from continuous states back to anchors.
- `sjd_model.py`: bundles anchor tables, classifier backbones, and model-facing helpers.
- `anchors.py`: defines token anchor tables, transforms, and known-token clamping helpers.
- `classifier.py`: adapts shared image and sequence backbones to SJD anchor classification.

See [docs/repo_layout.md](docs/repo_layout.md), [docs/configs.md](docs/configs.md),
[docs/datasets.md](docs/datasets.md), and [docs/adding_a_baseline.md](docs/adding_a_baseline.md)
for further details.

## Contribute

Commit log format:

```
ABBR: Commit message.
```

**ABBR options:**

- `IMP`: Development and implementation of a new feature.
- `FIX`: A fix of an existing bug.
- `OPT`: Any optimization performed.
- `REF`: Any refactors to code.
