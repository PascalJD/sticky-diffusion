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


def test_train_entrypoint_import_smoke(monkeypatch):
    _install_tensorflow_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "sticky.entrypoints.train", raising=False)

    module = importlib.import_module("sticky.entrypoints.train")
    assert callable(module.main)


def test_eval_checkpoint_entrypoint_import_smoke(monkeypatch):
    _install_tensorflow_stub(monkeypatch)
    monkeypatch.delitem(sys.modules, "sticky.entrypoints.eval_checkpoint", raising=False)

    module = importlib.import_module("sticky.entrypoints.eval_checkpoint")
    assert callable(module.main)
