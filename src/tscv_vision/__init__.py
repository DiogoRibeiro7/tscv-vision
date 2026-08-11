"""Top-level package for :mod:`tscv_vision`.

This module exposes subpackages lazily to avoid hard dependencies on optional
third-party libraries (e.g. scikit-learn, torch).  Attributes listed in
``__all__`` are imported on first access via :func:`importlib.import_module`.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

__all__ = [
    "aggregation",
    "analysis",
    "analytics",
    "automl",
    "dataset",
    "domains",
    "encoders",
    "evaluation",
    "features",
    "fusion",
    "gpu",
    "io",
    "irregular",
    "ml_integration",
    "mlops",
    "multimodal",
    "multivariate",
    "neural",
    "parallel",
    "pipeline",
    "representations",
    "research",
    "stats",
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
        module_name = name
        attr_name: str | None = None
        if name == "AutoTSCV":
            module_name, attr_name = "automl", "AutoTSCV"
        elif name == "WindowedDataset":
            module_name, attr_name = "dataset", "WindowedDataset"
        try:
            module = importlib.import_module(f".{module_name}", __name__)
        except ImportError as exc:
            msg = (
                f"Optional submodule '{module_name}' could not be imported. "
                "Install the required extras to use this feature."
            )
            raise AttributeError(msg) from exc
        if attr_name is not None:
            obj = getattr(module, attr_name)
            globals()[name] = obj
            return obj
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.2.0"

