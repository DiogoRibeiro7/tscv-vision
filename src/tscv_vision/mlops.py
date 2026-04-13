"""MLOps utilities for production feature pipelines.

This optional module provides helpers for deploying tscv-vision in large scale
settings. All functionality is gated behind optional dependencies so the core
package remains NumPy-only.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import erfc, sqrt
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .encoders import gaf
from .features import extract_feature_vector

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

Array = NDArray[np.float64]

logger = logging.getLogger(__name__)


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


class ModelRegistry:
    """Thread-safe in-memory registry for encoder versions and metrics.

    Examples
    --------
    >>> reg = ModelRegistry()
    >>> reg.register("gaf", "1.0", {"acc": 0.9})
    >>> reg.set_status("gaf", "1.0", "deployed")
    >>> reg.latest("gaf")
    '1.0'
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, dict[str, dict[str, Any]]] = {}

    def register(
        self, name: str, version: str, metrics: dict[str, float], status: str = "staging"
    ) -> None:
        """Register a new model ``version`` with ``metrics``."""

        with self._lock:
            logger.info("registering %s version %s", name, version)
            self._store.setdefault(name, {})[version] = {
                "metrics": dict(metrics),
                "status": status,
            }

    def update_metrics(self, name: str, version: str, metrics: dict[str, float]) -> None:
        """Update stored ``metrics`` for ``name`` and ``version``."""

        with self._lock:
            self._store[name][version]["metrics"].update(metrics)

    def set_status(self, name: str, version: str, status: str) -> None:
        """Set deployment ``status`` (e.g. ``staging`` or ``deployed``)."""

        with self._lock:
            self._store[name][version]["status"] = status

    def latest(self, name: str) -> str:
        """Return latest version string for ``name``."""

        with self._lock:
            versions = self._store.get(name)
            if not versions:
                raise KeyError(f"{name!r} not found in registry")
            return sorted(versions)[-1]

    def get(self, name: str, version: str | None = None) -> dict[str, Any]:
        """Return metadata for ``name`` and ``version`` (latest if ``None``)."""

        with self._lock:
            versions = self._store.get(name)
            if not versions:
                raise KeyError(f"{name!r} not found in registry")
            if version is None:
                version = sorted(versions)[-1]
            return versions[version]


@dataclass
class ABTestResult:
    """Result of an A/B test."""

    p_value: float
    lift: float


class ABTester:
    """Collect metrics for two variants and compute a t-test."""

    def __init__(self) -> None:
        self._a: list[float] = []
        self._b: list[float] = []

    def add(self, variant: str, value: float) -> None:
        """Record ``value`` for variant ``'A'`` or ``'B'``."""

        (self._a if variant.upper() == "A" else self._b).append(float(value))

    def compare(self) -> ABTestResult:
        """Return p-value and lift of variant ``B`` over ``A``.

        Raises
        ------
        ValueError
            If fewer than two samples have been added per variant.
        """

        a = np.asarray(self._a, dtype=float)
        b = np.asarray(self._b, dtype=float)
        if a.size < 2 or b.size < 2:
            raise ValueError("need >=2 samples for each variant")
        mean_a = float(a.mean())
        mean_b = float(b.mean())
        var_a = float(a.var(ddof=1))
        var_b = float(b.var(ddof=1))
        n_a = a.size
        n_b = b.size
        t = (mean_a - mean_b) / np.sqrt(var_a / n_a + var_b / n_b)
        z = abs(t)
        p = float(erfc(z / sqrt(2.0)))
        return ABTestResult(p_value=p, lift=mean_b - mean_a)


def safe_encode(
    series: Array,
    primary: Callable[[Array], Array],
    fallback: Callable[[Array], Array],
    timeout: float = 0.1,
) -> Array:
    """Encode ``series`` with ``primary`` falling back on errors or latency.

    Parameters
    ----------
    series:
        1D input array.
    primary:
        Encoder attempted first.
    fallback:
        Encoder used when ``primary`` fails or exceeds ``timeout`` seconds.
    timeout:
        Maximum allowed processing time in seconds.
    """

    start = time.perf_counter()
    try:
        out = primary(series)
        if time.perf_counter() - start > timeout:
            raise TimeoutError("primary encoder timed out")
        return out
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("primary encoder failed: %s; using fallback", exc)
        return fallback(series)


def batch_process(
    data: Iterable[Array],
    func: Callable[[Array], Iterable[Array]],
    batch_size: int = 64,
    start: int = 0,
    progress: Callable[[int], None] | None = None,
) -> list[Array]:
    """Process ``data`` in batches with optional resumption and progress.

    Parameters
    ----------
    data:
        Iterable of 1D arrays.
    func:
        Function applied to each batch (stacked into ``(B, N)``).
    batch_size:
        Number of samples per batch.
    start:
        Index to resume from.
    progress:
        Optional callback receiving the number of processed samples.
    """

    outputs: list[Array] = []
    batch: list[Array] = []
    for idx, series in enumerate(data):
        if idx < start:
            continue
        batch.append(series)
        if len(batch) == batch_size:
            arr = np.stack(batch)
            outputs.extend(func(arr))
            if progress:
                progress(len(outputs))
            batch.clear()
    if batch:
        arr = np.stack(batch)
        outputs.extend(func(arr))
        if progress:
            progress(len(outputs))
    return outputs


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
    def extract(data: list[float] = Body(..., embed=True)) -> dict[str, Any]:  # noqa: B008
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 1 or arr.size < 2:
            raise HTTPException(status_code=400, detail="series must be 1D with >=2 samples")
        img = gaf(arr)
        feats = extract_feature_vector(img)
        if counter is not None:
            counter.inc()
        return {"features": feats.tolist()}

    return app


def create_monitoring_app(
    detector: DriftDetector | None = None,
) -> FastAPI:  # pragma: no cover - requires fastapi
    """Create a FastAPI app exposing health and drift metrics.

    Parameters
    ----------
    detector:
        Optional :class:`DriftDetector` used to flag distribution shift.

    Examples
    --------
    >>> app = create_monitoring_app(DriftDetector())
    """

    from fastapi import Body, FastAPI

    app = FastAPI(title="tscv-monitor")

    try:  # optional Prometheus metrics
        from prometheus_client import Counter, Gauge, make_asgi_app

        drift_counter = Counter(
            "tscv_drift_events_total", "Number of detected drift events"
        )
        quality_gauge = Gauge(
            "tscv_feature_quality", "Latest feature quality score"
        )
        health_gauge = Gauge(
            "tscv_system_health", "System health", ["component"]
        )
        app.mount("/metrics", make_asgi_app())
    except Exception:  # pragma: no cover - metrics optional
        drift_counter = quality_gauge = health_gauge = None

    @app.get("/health")  # type: ignore[misc]
    def health() -> dict[str, str]:
        if health_gauge is not None:
            health_gauge.labels(component="api").set(1.0)
        return {"status": "ok"}

    @app.post("/drift")  # type: ignore[misc]
    def drift(
        baseline: list[float] = Body(..., embed=True),  # noqa: B008
        current: list[float] = Body(..., embed=True),  # noqa: B008
    ) -> dict[str, bool]:
        base_arr = np.asarray(baseline, dtype=float)
        cur_arr = np.asarray(current, dtype=float)
        has_drift = detector.has_drift(base_arr, cur_arr) if detector else False
        if has_drift and drift_counter is not None:
            drift_counter.inc()
        return {"drift": bool(has_drift)}

    @app.post("/quality")  # type: ignore[misc]
    def quality(score: float = Body(..., embed=True)) -> dict[str, float]:  # noqa: B008
        if quality_gauge is not None:
            quality_gauge.set(score)
        return {"score": float(score)}

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
    "ModelRegistry",
    "ABTester",
    "ABTestResult",
    "safe_encode",
    "batch_process",
    "create_feature_service",
    "create_monitoring_app",
    "FeastWriter",
]
