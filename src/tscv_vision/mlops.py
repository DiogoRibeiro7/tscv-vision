"""MLOps utilities for production feature pipelines.

This optional module provides helpers for deploying tscv-vision in large scale
settings. All functionality is gated behind optional dependencies so the core
package remains NumPy-only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .encoders import gaf
from .features import extract_feature_vector

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

Array = NDArray[np.float64]


def validate_features(features: Array) -> None:
    """Validate a feature array.

    Parameters
    ----------
    features:
        Feature matrix ``(N, D)`` or vector ``(D,)``.

    Raises
    ------
    ValueError
        If the array contains NaN or infinite values.
    """

    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinite values")


@dataclass
class DriftDetector:
    """Detect distribution drift via histogram comparison."""

    bins: int = 32
    threshold: float = 0.1

    def has_drift(self, baseline: Array, current: Array) -> bool:
        """Return ``True`` if ``current`` drifts from ``baseline``.

        Uses KL divergence between histograms normalised to sum to one.
        ``baseline`` and ``current`` should be 1D feature vectors.
        """

        edges = np.histogram_bin_edges(
            np.concatenate([baseline, current]), bins=self.bins
        )
        p_hist, _ = np.histogram(baseline, bins=edges, density=True)
        q_hist, _ = np.histogram(current, bins=edges, density=True)
        # Avoid division by zero by adding a small epsilon
        eps = 1e-12
        p = p_hist + eps
        q = q_hist + eps
        kl = float(np.sum(p * np.log(p / q)))
        return kl > self.threshold


def assign_variant(key: Any) -> str:
    """Assign ``key`` deterministically to variant "A" or "B"."""

    h = hashlib.sha256(str(key).encode()).hexdigest()
    # Use the last hex digit to keep distribution even
    return "B" if int(h[-1], 16) % 2 else "A"


@dataclass
class ResourceScaler:
    """Utility to scale workers based on throughput metrics."""

    max_replicas: int = 100

    def required_replicas(self, throughput: float, target: float) -> int:
        """Return number of replicas to meet ``target`` qps.

        Parameters
        ----------
        throughput:
            Observed queries per second of a single worker.
        target:
            Desired total queries per second.
        """

        if throughput <= 0:
            raise ValueError("throughput must be positive")
        replicas = int(np.ceil(target / throughput))
        return min(replicas, self.max_replicas)


def create_feature_service() -> FastAPI:  # pragma: no cover - requires fastapi
    """Create a FastAPI application exposing feature extraction.

    The app provides one ``/extract`` endpoint accepting a list of floats and
    returning a feature vector. Prometheus metrics are exposed at ``/metrics``
    if :mod:`prometheus_client` is available.
    """

    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title="tscv-vision")

    try:  # optional monitoring
        from prometheus_client import Counter, make_asgi_app

        counter = Counter("tscv_requests_total", "Total feature extraction requests")
        app.mount("/metrics", make_asgi_app())
    except Exception:  # pragma: no cover - metrics optional
        counter = None

    @app.post("/extract")  # type: ignore[misc]
    def extract(data: list[float] = Body(...)) -> dict[str, Any]:  # noqa: B008
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 1 or arr.size < 2:
            raise HTTPException(status_code=400, detail="series must be 1D with >=2 samples")
        img = gaf(arr)
        feats = extract_feature_vector(img)
        if counter is not None:
            counter.inc()
        return {"features": feats.tolist()}

    return app


class FeastWriter:
    """Light-weight wrapper around :mod:`feast` feature store."""

    def __init__(self, repo_path: str | None = None) -> None:
        try:
            from feast import FeatureStore
        except Exception as exc:  # pragma: no cover - dependency optional
            raise ImportError("feast is required for FeastWriter") from exc
        self._store = FeatureStore(repo_path or ".")

    def push(self, entity: str, features: dict[str, Iterable[Any]]) -> None:
        """Push features for ``entity`` to the store."""

        from feast import EntityData, FeatureData  # pragma: no cover - dependency optional

        fd = FeatureData.from_dict(features)
        ed = EntityData(entity, {})
        self._store.push("fs", [ed], [fd])


__all__ = [
    "validate_features",
    "DriftDetector",
    "assign_variant",
    "ResourceScaler",
    "create_feature_service",
    "FeastWriter",
]
