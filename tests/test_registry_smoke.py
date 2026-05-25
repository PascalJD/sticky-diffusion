# tests/test_registry_smoke.py
"""Smoke tests that the init registry and model-builder dispatch table are
fully populated after importing ``sticky.models.factories``.

If any family module has a broken import (e.g., a typo in anchor_config), these
tests fail with a clear ImportError rather than a confusing
``Unknown model.name=...`` at runtime.
"""
from __future__ import annotations

EXPECTED_FAMILIES = frozenset(
    {"md4", "mdlm", "mdm", "d3pm", "sjd", "cadd", "bitdiff", "candi", "ddpm"}
)


def test_init_registry_has_expected_families():
    """After importing factories, _INIT_REGISTRY contains exactly the 9 model families."""
    import sticky.models.factories  # noqa: F401 — triggers @register_init decorators
    from sticky.models._registry import _INIT_REGISTRY

    missing = EXPECTED_FAMILIES - set(_INIT_REGISTRY)
    assert not missing, f"Init registry missing families: {sorted(missing)}"
    extra = set(_INIT_REGISTRY) - EXPECTED_FAMILIES
    assert not extra, f"Init registry has unexpected families: {sorted(extra)}"


def test_model_builders_has_expected_families():
    """MODEL_BUILDERS dict from factories.__init__ contains all 9 model families."""
    from sticky.models.factories import MODEL_BUILDERS

    missing = EXPECTED_FAMILIES - set(MODEL_BUILDERS)
    assert not missing, f"MODEL_BUILDERS missing families: {sorted(missing)}"
    assert set(MODEL_BUILDERS) == EXPECTED_FAMILIES, (
        f"MODEL_BUILDERS has unexpected families: "
        f"{sorted(set(MODEL_BUILDERS) - EXPECTED_FAMILIES)}"
    )
