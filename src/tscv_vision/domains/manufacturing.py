"""Manufacturing and process-monitoring utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
QUALITY_WEIGHTS: Array = np.array([0.5, 0.3, 0.2])

__all__ = [
    "vibration_features",
    "quality_score",
    "generate_vibration_series",
    "augment_spike",
    "QUALITY_WEIGHTS",
]


def vibration_features(signal: Array) -> Array:
    """Compute RMS, peak and kurtosis of a vibration signal."""
    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 1 or sig.size == 0:
        raise ValueError("signal must be 1D and non-empty")
    rms = np.sqrt(np.mean(sig ** 2))
    peak = np.max(np.abs(sig))
    kurt = np.mean((sig - sig.mean()) ** 4) / (sig.var() ** 2 + 1e-12)
    return np.array([rms, peak, kurt])


def quality_score(signal: Array, coeffs: Array = QUALITY_WEIGHTS) -> float:
    """Estimate quality using weighted vibration features."""
    feats = vibration_features(signal)
    return float(feats @ coeffs)


def generate_vibration_series(n: int = 100, *, seed: int = 0) -> Array:
    """Generate a synthetic vibration signal."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=n)


def augment_spike(
    signal: Array,
    probability: float = 0.01,
    scale: float = 1.0,
    *,
    seed: int = 0,
) -> Array:
    """Inject random spikes to simulate faults."""
    rng = np.random.default_rng(seed)
    sig = np.asarray(signal, dtype=float).copy()
    mask = rng.random(sig.shape) < probability
    sig[mask] += rng.normal(0.0, scale, size=mask.sum())
    return sig
