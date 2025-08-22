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

__all__ = [
    "finance",
    "healthcare",
    "iot",
    "audio",
    "astronomy",
    "climate",
    "manufacturing",
]
