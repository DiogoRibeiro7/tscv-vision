"""Domain-specific encoders and feature extractors."""

from __future__ import annotations

from . import (
    astronomy,
    audio,
    climate,
    finance,
    healthcare,
    iot,
    manufacturing,
)
from .adapter import (
    DomainAdapter,
    PrototypicalClassifier,
    classification_metrics,
    uncertainty_sampling,
)

__all__ = [
    "finance",
    "healthcare",
    "iot",
    "audio",
    "astronomy",
    "climate",
    "manufacturing",
    "DomainAdapter",
    "PrototypicalClassifier",
    "classification_metrics",
    "uncertainty_sampling",
]
