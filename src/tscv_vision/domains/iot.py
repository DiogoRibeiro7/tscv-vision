"""IoT and sensor-network utilities."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

FUSION_WEIGHTS: Array = np.array([0.5, 0.5])
ANOMALY_THRESHOLD = 3.0


def fuse_sensors(sensors: Array, weights: Array | None = None) -> Array:
    """Fuse multiple sensors via weighted average.

    Parameters
    ----------
    sensors : Array
        Array of shape ``(S, N)`` where ``S`` is the number of sensors.
    weights : Array, optional
        Fusion weights of shape ``(S,)``. Defaults to equal weights.

    Returns
    -------
    Array
        Fused signal of shape ``(N,)``.
    """
    data = np.asarray(sensors, dtype=float)
    if data.ndim != 2:
        raise ValueError("sensors must be 2D")
    default = np.full(data.shape[0], 1 / data.shape[0])
    w = np.asarray(weights if weights is not None else default, dtype=float)
    if w.shape[0] != data.shape[0]:
        raise ValueError("weights shape mismatch")
    w = w / w.sum()
    return cast(Array, w @ data)


def anomaly_score(signal: Array, threshold: float = ANOMALY_THRESHOLD) -> float:
    """Return maximum z-score as anomaly score."""
    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 1:
        raise ValueError("signal must be 1D")
    z = (sig - sig.mean()) / (sig.std() + 1e-12)
    score = float(np.max(np.abs(z)))
    return float(score / threshold)


def generate_sensor_series(n: int = 100, *, seed: int = 0) -> Array:
    """Generate a synthetic sensor signal with Gaussian noise."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=n)


def augment_dropout(signal: Array, drop_prob: float = 0.1, *, seed: int = 0) -> Array:
    """Randomly drop samples to simulate sensor outages."""
    rng = np.random.default_rng(seed)
    sig = np.asarray(signal, dtype=float).copy()
    mask = rng.random(sig.shape) < drop_prob
    sig[mask] = 0.0
    return sig


def iot_features(signal: Array) -> Array:
    """Return basic statistics for an IoT signal.

    Parameters
    ----------
    signal : Array
        1D input sequence.

    Returns
    -------
    Array
        ``[mean, std, min, max]`` of the signal.
    """

    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 1:
        raise ValueError("signal must be 1D")
    stats = [sig.mean(), sig.std(), sig.min(), sig.max()]
    return cast(Array, np.array(stats, dtype=float))
