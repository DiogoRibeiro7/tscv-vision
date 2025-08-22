"""Utilities for irregularly sampled time series."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def resample_irregular(t: Array, x: Array, step: float) -> tuple[Array, Array]:
    """Resample an irregular series ``(t, x)`` onto a regular grid."""

    t_arr = np.asarray(t, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if t_arr.ndim != 1 or x_arr.ndim != 1 or t_arr.shape[0] != x_arr.shape[0]:
        raise ValueError("t and x must be 1D arrays of the same length")
    if step <= 0:
        raise ValueError("step must be positive")
    t_new = np.arange(t_arr[0], t_arr[-1] + step, step)
    x_new = np.interp(t_new, t_arr, x_arr)
    return t_new, x_new


__all__ = ["resample_irregular"]

