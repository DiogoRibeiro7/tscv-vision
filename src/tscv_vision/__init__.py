"""Top-level package for tscv-vision."""

from . import (
    aggregation,
    analysis,
    analytics,
    automl,
    dataset,
    domains,
    encoders,
    features,
    fusion,
    io,
    irregular,
    ml_integration,
    mlops,
    multimodal,
    neural,
    nextgen,
    parallel,
    research,
    streaming,
)
from .dataset import WindowedDataset

__all__ = [
    "aggregation",
    "analysis",
    "analytics",
    "automl",
    "domains",
    "dataset",
    "encoders",
    "features",
    "fusion",
    "io",
    "irregular",
    "ml_integration",
    "mlops",
    "multimodal",
    "nextgen",
    "neural",
    "parallel",
    "research",
    "streaming",
    "WindowedDataset",
]

__version__ = "0.10.0"
