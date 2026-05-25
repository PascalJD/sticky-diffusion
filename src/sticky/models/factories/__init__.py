"""Per-family model factory modules.

Importing this package triggers all ``@register_init`` decorators so the
registry is populated before ``init_state`` looks up by ``cfg.model.name``.
"""
from sticky.models.factories import (  # noqa: F401
    bitdiff,
    cadd,
    candi,
    d3pm,
    ddpm,
    discrete_baselines,
    md4,
    mdlm,
    mdm,
    sjd,
)

# Build-model dispatch table — one entry per model.name key.
# Note: "mdm" maps to the inpaint variant (the original _build_mdm was dead code).
MODEL_BUILDERS = {
    "md4": md4.build_model,
    "mdlm": mdlm.build_model,
    "mdm": mdm.build_model,
    "d3pm": d3pm.build_model,
    "sjd": sjd.build_model,
    "cadd": cadd.build_model,
    "bitdiff": bitdiff.build_model,
    "candi": candi.build_model,
    "ddpm": ddpm.build_model,
}
