"""Domain-specific encoders and feature extractors."""

from __future__ import annotations

from typing import Any

from . import (
    astronomy,
    audio,
    climate,
    finance,
    healthcare,
    iot,
    manufacturing,
)

DomainAdapter: Any
PrototypicalClassifier: Any
classification_metrics: Any
uncertainty_sampling: Any

try:  # optional dependency
    from .adapter import (
        DomainAdapter as _DomainAdapter,
    )
    from .adapter import (
        PrototypicalClassifier as _PrototypicalClassifier,
    )
    from .adapter import (
        classification_metrics as _classification_metrics,
    )
    from .adapter import (
        uncertainty_sampling as _uncertainty_sampling,
    )
    DomainAdapter = _DomainAdapter
    PrototypicalClassifier = _PrototypicalClassifier
    classification_metrics = _classification_metrics
    uncertainty_sampling = _uncertainty_sampling
except Exception:  # pragma: no cover - optional dependency
    DomainAdapter = PrototypicalClassifier = classification_metrics = uncertainty_sampling = None

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
