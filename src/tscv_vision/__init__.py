"""Top-level package for tscv-vision."""

from . import aggregation, encoders, features, fusion, io, ml_integration, parallel
from .dataset import WindowedDataset

__all__ = [
    "aggregation",
    "encoders",
    "features",
    "fusion",
    "io",
    "ml_integration",
    "parallel",
    "WindowedDataset",
]

__version__ = "0.5.0"
