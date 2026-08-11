"""Every module must import with only the core dependencies installed.

The core package promises a NumPy-only footprint, with optional dependencies
raising a directed ``ImportError`` at the point of use rather than at import.
Developer machines have the extras installed, so this is exactly the kind of
breakage that reaches CI unnoticed — as it did when ``pipeline`` aliased both
scikit-learn base classes to ``object`` and became a duplicate-base
``TypeError`` on any machine without scikit-learn.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pytest

#: Modules that must import without any optional dependency present.
CORE_MODULES = [
    "tscv_vision.aggregation",
    "tscv_vision.analysis",
    "tscv_vision.automl",
    "tscv_vision.dataset",
    "tscv_vision.encoders",
    "tscv_vision.evaluation",
    "tscv_vision.features",
    "tscv_vision.fusion",
    "tscv_vision.io",
    "tscv_vision.irregular",
    "tscv_vision.ml_integration",
    "tscv_vision.mlops",
    "tscv_vision.multimodal",
    "tscv_vision.parallel",
    "tscv_vision.pipeline",
    "tscv_vision.representations",
    "tscv_vision.research",
    "tscv_vision.sliding",
    "tscv_vision.stats",
    "tscv_vision.streaming",
]

#: Optional packages hidden while importing.
OPTIONAL_PACKAGES = (
    "sklearn",
    "torch",
    "torchvision",
    "matplotlib",
    "seaborn",
    "shap",
    "lime",
    "umap",
    "pywt",
    "numba",
    "cupy",
    "dask",
    "pyarrow",
    "h5py",
    "redis",
    "kafka",
    "pika",
    "onnx",
    "pyts",
    "fastapi",
    "prometheus_client",
    "feast",
    "yaml",
    "tensorflow",
)


@contextmanager
def _hidden(packages: Sequence[str]) -> Iterator[None]:
    """Make ``packages`` unimportable for the duration of the block."""

    blocked = set(packages)
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in blocked:
            raise ImportError(f"{name} is hidden by test_optional_dependency_isolation")
        return real_import(name, *args, **kwargs)

    saved = {
        key: value
        for key, value in sys.modules.items()
        if key.split(".")[0] in blocked or key.startswith("tscv_vision")
    }
    for key in list(saved):
        del sys.modules[key]
    builtins.__import__ = fake_import
    try:
        yield
    finally:
        builtins.__import__ = real_import
        for key in [k for k in sys.modules if k.startswith("tscv_vision")]:
            del sys.modules[key]
        sys.modules.update(saved)


@pytest.mark.parametrize("module", CORE_MODULES)
def test_module_imports_without_optional_dependencies(module: str) -> None:
    with _hidden(OPTIONAL_PACKAGES):
        importlib.import_module(module)


def test_pipeline_classes_are_usable_without_sklearn() -> None:
    """Regression: both fallback bases were ``object``, a duplicate-base error."""

    with _hidden(OPTIONAL_PACKAGES):
        pipeline = importlib.import_module("tscv_vision.pipeline")
        assert pipeline.FeatureSelector is not None
        # Constructing it must raise a directed ImportError, not TypeError.
        with pytest.raises(ImportError, match="scikit-learn"):
            pipeline.FeatureSelector()


def test_optional_features_raise_directed_import_errors() -> None:
    with _hidden(OPTIONAL_PACKAGES):
        features = importlib.import_module("tscv_vision.features")
        with pytest.raises(ImportError, match="pywt"):
            features.wavelet_stats(__import__("numpy").zeros((4, 4)))

        parallel = importlib.import_module("tscv_vision.parallel")
        with pytest.raises(ImportError, match="dask"):
            parallel.map_dask(lambda x: x, [1, 2])


def test_top_level_package_stays_lazy() -> None:
    """Importing the package must not pull in any optional dependency."""

    with _hidden(OPTIONAL_PACKAGES):
        importlib.import_module("tscv_vision")
        leaked = sorted(
            name
            for name in sys.modules
            if name.split(".")[0] in set(OPTIONAL_PACKAGES)
        )
        assert not leaked, f"importing tscv_vision pulled in {leaked}"
