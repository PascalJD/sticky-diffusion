"""Per-family model factory modules.

Importing this package triggers all ``@register_init`` decorators so the
registry is populated before ``init_state`` looks up by ``cfg.model.name``.
"""
from sticky.models.factories import discrete_baselines, mdm, sjd  # noqa: F401
