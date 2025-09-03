"""Top-level package for tscv-vision."""

try:  # pragma: no cover - optional dependency paths
    from . import domains
except Exception:
    domains = None  # type: ignore[assignment]

from . import (
    aggregation,
    analysis,
    analytics,
    automl,
    dataset,
    encoders,
    features,
    fusion,
    gpu,
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
from .automl import AutoTSCV
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
    "aggregation",
    "analysis",
    "analytics",
    "automl",
    "dataset",
    "domains",
    "encoders",
    "features",
    "fusion",
    "gpu",
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
    "AutoTSCV",
]

__version__ = "0.10.0"
