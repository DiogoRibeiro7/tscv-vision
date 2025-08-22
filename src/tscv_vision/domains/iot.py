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
