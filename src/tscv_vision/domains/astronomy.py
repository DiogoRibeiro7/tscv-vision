"""Astronomy and astrophysics utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
VARIABLE_STAR_THRESHOLD = 0.1


def periodicity_features(signal: Array, fs: float = 1.0) -> Array:
    """Return dominant frequency and its amplitude."""
    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 1 or sig.size == 0:
        raise ValueError("signal must be 1D and non-empty")
    spectrum = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(sig.size, d=1.0 / fs)
    idx = int(np.argmax(spectrum[1:])) + 1
    return np.array([freqs[idx], spectrum[idx]])


def variable_star_score(signal: Array, fs: float = 1.0) -> float:
    """Score variability using dominant amplitude."""
    _, amp = periodicity_features(signal, fs)
    return float(amp / VARIABLE_STAR_THRESHOLD)
