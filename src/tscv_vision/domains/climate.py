"""Climate and weather analysis utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
EXTREME_THRESHOLD = 3.0


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
