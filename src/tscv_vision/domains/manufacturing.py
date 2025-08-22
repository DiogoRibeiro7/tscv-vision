"""Manufacturing and process-monitoring utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
QUALITY_WEIGHTS: Array = np.array([0.5, 0.3, 0.2])


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
