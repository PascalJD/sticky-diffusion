"""Guard against accidental reintroduction of the legacy-vs-corrected
switch. After Prompt A, the SJD module exports paper-faithful names with no
denom / dhm_denom / corruption flag.
"""
from __future__ import annotations

import importlib

import pytest


def test_renamed_public_symbols_are_importable():
    """The corrected paper-faithful names must be importable from
    `sticky.models.sjd`."""
    from sticky.models.sjd import (  # noqa: F401
        classifier_induced_score,
        dhm_log_ratio,
        dhm_target,
        lambda_off_star,
        lambda_time_star,
        mixture_logpdf,
        plugin_hazard_and_allocation,
        plugin_hazard_and_anchor,
        sample_pair,
    )


def test_legacy_symbols_are_not_present():
    """The legacy names that named the math mistake (or that were transitional
    safety nets in steps 1-7) must not be re-exported."""
    sjd = importlib.import_module("sticky.models.sjd")
    forbidden = (
        "anchor_log_ratio",
        "dhm_target_intensity",
        "lam_off_star",
        "lam_time_star",
        "plugin_intensity_and_choice",
        "plugin_intensity_and_probs",
        "sample_sjd_pair",
        "sjd_corruption",  # the file rename
    )
    for name in forbidden:
        assert not hasattr(sjd, name), (
            f"`sticky.models.sjd` re-exports legacy symbol {name!r}; the "
            "Prompt A rename should have purged it."
        )


def test_module_renamed_to_corruption():
    """`sticky.models.sjd.sjd_corruption` must not import; the module is now
    `sticky.models.sjd.corruption`."""
    importlib.import_module("sticky.models.sjd.corruption")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sticky.models.sjd.sjd_corruption")


def test_no_denom_or_corruption_flags_in_public_signatures():
    """Spot-check that the public corrected functions don't accept legacy
    flag arguments. (Behavioral guard against silent reintroduction.)"""
    import inspect

    from sticky.models.sjd import (
        classifier_induced_score,
        dhm_log_ratio,
        dhm_target,
        plugin_hazard_and_allocation,
        plugin_hazard_and_anchor,
    )
    from sticky.models.sjd.losses import ce_allocation_loss

    forbidden_params = {"denom", "dhm_denom", "corruption", "n_tau"}
    for fn in (
        classifier_induced_score,
        dhm_log_ratio,
        dhm_target,
        plugin_hazard_and_allocation,
        plugin_hazard_and_anchor,
        ce_allocation_loss,
    ):
        params = set(inspect.signature(fn).parameters)
        bad = params & forbidden_params
        assert not bad, (
            f"{fn.__name__} re-exposes legacy flag arguments: {sorted(bad)}"
        )
