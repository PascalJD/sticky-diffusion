# Smoke test: every experiment YAML resolves via Hydra without errors.
import glob
import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config")
)
_EXPERIMENT_DIR = os.path.join(_CONFIG_DIR, "experiment")

_all_yamls = sorted(
    glob.glob(os.path.join(_EXPERIMENT_DIR, "**", "*.yaml"), recursive=True)
)

_params = []
for _y in _all_yamls:
    rel = os.path.relpath(_y, _EXPERIMENT_DIR).replace(".yaml", "")
    # skip sweep configs: they use Hydra sweeper syntax, not compose-compatible
    if "sweeps" in rel.replace(os.sep, "/"):
        continue
    _params.append(rel)


@pytest.mark.parametrize("experiment_rel", _params)
def test_config_resolves(experiment_rel):
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[f"experiment={experiment_rel}"],
            )
        exp = cfg.experiment
        assert "dataset" in exp, f"'dataset' key missing for {experiment_rel}"
        assert "model" in exp, f"'model' key missing for {experiment_rel}"
    finally:
        GlobalHydra.instance().clear()
