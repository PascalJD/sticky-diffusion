# Sticky Jump Diffusions and Discrete Diffusion Baselines

**Status:** 🚧 active research

## Installation

```bash
git clone https://github.com/PascalJD/sticky-diffusion.git
cd sticky-diffusion
conda env create -f environment.yml
conda activate sticky
```

## Quick Start

Train the canonical board-level SJD Sudoku experiment:
```bash
python -m sticky.cli.train experiment=sudoku/sjd_sudoku eval=sudoku_sjd
```

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
