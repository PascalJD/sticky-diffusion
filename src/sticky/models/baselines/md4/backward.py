"""Backward classifier re-export used by MD4.

MD4 now reuses shared architecture classifiers from `sticky.models.backbones`.
"""

from sticky.models.backbones import DiscreteClassifier

__all__ = ["DiscreteClassifier"]
