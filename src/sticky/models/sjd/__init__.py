"""SJD model package: corrected DHM training + sampling.

This is the public API surface for paper-faithful imports:

    from sticky.models.sjd import (
        sample_pair,
        mixture_logpdf,
        classifier_induced_score,
        dhm_log_ratio,
        dhm_target,
        plugin_hazard_and_allocation,
        plugin_hazard_and_anchor,
        lambda_off_star,
        lambda_time_star,
    )

Function names mirror the paper notation directly. There is no `denom`,
`dhm_denom`, or `corruption` flag — the SJD off-anchor mixture is the only
denominator and `sample_pair` is the only training-side corruption sampler.
"""

from .corruption import (
    classifier_induced_score,
    mixture_logpdf,
    sample_pair,
)
from .hazard import (
    HazardSchedule,
    lambda_off_star,
    lambda_time_star,
)
from .plugin_intensity import (
    dhm_log_ratio,
    dhm_target,
    plugin_hazard_and_allocation,
    plugin_hazard_and_anchor,
)

__all__ = [
    "HazardSchedule",
    "classifier_induced_score",
    "dhm_log_ratio",
    "dhm_target",
    "lambda_off_star",
    "lambda_time_star",
    "mixture_logpdf",
    "plugin_hazard_and_allocation",
    "plugin_hazard_and_anchor",
    "sample_pair",
]
