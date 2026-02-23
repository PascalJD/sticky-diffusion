"""Backward classifier compatibility shim.

MD4 now reuses shared architecture classifiers from `sticky.models.architectures`.
"""

from sticky.models.architectures import DiscreteClassifier

__all__ = ["DiscreteClassifier"]

