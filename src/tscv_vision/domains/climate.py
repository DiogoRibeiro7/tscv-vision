"""Climate and weather analysis utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
EXTREME_THRESHOLD = 3.0

__all__ = [
    "seasonal_trend_features",
    "extreme_event_score",
    "generate_temperature_series",
    "augment_regime_shift",
    "EXTREME_THRESHOLD",
]


def seasonal_trend_features(series: Array, period: int) -> Array:
    """Compute seasonal variance, trend slope, and extreme deviation."""
    s = np.asarray(series, dtype=float)
    if s.ndim != 1 or s.size < period or period <= 0:
        raise ValueError("invalid series or period")
    seasonal = s[:-period] - s[period:]
    seasonal_var = seasonal.var()
    slope = np.polyfit(np.arange(s.size), s, 1)[0]
    extreme = np.max(np.abs(s - s.mean()))
    return np.array([seasonal_var, slope, extreme])


def extreme_event_score(series: Array, period: int, threshold: float = EXTREME_THRESHOLD) -> float:
    """Score extreme events relative to threshold."""
    _, _, extreme = seasonal_trend_features(series, period)
    return float(extreme / threshold)


def generate_temperature_series(
    n: int = 365,
    amplitude: float = 10.0,
    noise: float = 0.5,
    *,
    seed: int = 0,
) -> Array:
    """Generate a synthetic seasonal temperature series."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = amplitude * np.sin(2 * np.pi * t / n)
    return seasonal + rng.normal(0.0, noise, size=n)


def augment_regime_shift(series: Array, delta: float = 1.0) -> Array:
    """Add a mean shift to the second half of the series."""
    s = np.asarray(series, dtype=float).copy()
    mid = s.size // 2
    s[mid:] += delta
    return s
