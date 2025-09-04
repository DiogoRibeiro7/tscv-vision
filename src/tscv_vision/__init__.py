"""Top-level package for :mod:`tscv_vision`.

This module exposes subpackages lazily to avoid hard dependencies on optional
third-party libraries (e.g. scikit-learn, torch).  Attributes listed in
``__all__`` are imported on first access via :func:`importlib.import_module`.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, cast

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


def __getattr__(name: str) -> ModuleType | Any:  # pragma: no cover - exercised in tests
    """Dynamically import submodules on first access.

    This prevents optional heavy dependencies from being imported unless the
    corresponding submodule is actually used.
    """

    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:  # Optional export that may require extra dependencies
    AutoTSCV = importlib.import_module(".automl", __name__).AutoTSCV
except Exception:  # pragma: no cover - optional
    AutoTSCV = cast(Any, None)

try:
    WindowedDataset = importlib.import_module(".dataset", __name__).WindowedDataset
except Exception:  # pragma: no cover - optional
    WindowedDataset = cast(Any, None)


__version__ = "0.10.0"

