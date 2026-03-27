from __future__ import annotations

import importlib
import sys
import types


def _install_tensorflow_stub(monkeypatch):
    tf_stub = types.ModuleType("tensorflow")
    tf_stub.config = types.SimpleNamespace(
        set_visible_devices=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "tensorflow", tf_stub)


def _install_md4_optional_dependency_stubs(monkeypatch):
    distrax_mod = types.ModuleType("distrax")
    distrax_mod.Distribution = object
    monkeypatch.setitem(sys.modules, "distrax", distrax_mod)

    matplotlib_mod = types.ModuleType("matplotlib")
    pyplot_mod = types.ModuleType("matplotlib.pyplot")
    matplotlib_mod.pyplot = pyplot_mod
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_mod)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot_mod)

    orbax_mod = types.ModuleType("orbax")
    checkpoint_mod = types.ModuleType("orbax.checkpoint")
    checkpoint_mod.CheckpointManager = object
    orbax_mod.checkpoint = checkpoint_mod
    monkeypatch.setitem(sys.modules, "orbax", orbax_mod)
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", checkpoint_mod)

    monkeypatch.setitem(sys.modules, "seaborn", types.ModuleType("seaborn"))


def test_train_entrypoint_import_smoke(monkeypatch):
    _install_tensorflow_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "sticky.cli.train", raising=False)

    module = importlib.import_module("sticky.cli.train")
    assert callable(module.main)


def test_eval_checkpoint_cli_import_smoke(monkeypatch):
    _install_tensorflow_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "sticky.cli.eval_checkpoint", raising=False)

    module = importlib.import_module("sticky.cli.eval_checkpoint")
    assert callable(module.main)


def test_canonical_model_package_imports(monkeypatch):
    _install_md4_optional_dependency_stubs(monkeypatch)

    arch = importlib.import_module("sticky.models.backbones.sequence")
    md4 = importlib.import_module("sticky.models.baselines.md4.md4_model")
    mdlm_sampling = importlib.import_module("sticky.models.baselines.mdlm.sampling")

    assert callable(arch.TransformerBackbone)
    assert callable(arch.GPT2LikeBackbone)
    assert callable(md4.MD4)
    assert callable(mdlm_sampling.simple_generate)


def test_canonical_sudoku_data_imports():
    download = importlib.import_module("sticky.data.sudoku.download")
    dataset = importlib.import_module("sticky.data.sudoku.dataset")

    assert callable(download.ensure_sudoku_data_available)
    assert callable(dataset.make_sudoku_iterator)
