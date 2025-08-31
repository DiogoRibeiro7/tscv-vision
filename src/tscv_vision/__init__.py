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
    pipeline,
    research,
    streaming,
)
from .dataset import WindowedDataset

__all__ = [
    "aggregation",
    "analysis",
    "analytics",
    "automl",
    "dataset",
    "domains",
    "encoders",
    "features",
    "fusion",
    "io",
    "irregular",
    "ml_integration",
    "mlops",
    "multimodal",
    "neural",
    "nextgen",
    "parallel",
    "pipeline",
    "research",
    "streaming",
    "WindowedDataset",
]

__version__ = "0.10.0"
