"""Healthcare-specific encoders and features."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Coefficients for a toy arrhythmia risk model: intercept, SDNN weight
ARRHYTHMIA_COEFFS: Array = np.array([-1.0, 2.0])


def ecg_features(signal: Array, fs: float = 250.0) -> Array:
    """Extract simple heart-rate and variability features from ECG signal.

    Parameters
    ----------
    signal : Array
        ECG waveform of shape ``(N,)``.
    fs : float, optional
        Sampling rate in Hz. Default is ``250``.

    Returns
    -------
    Array
        ``[heart_rate, sdnn]`` where ``sdnn`` is the standard deviation of RR
        intervals. If fewer than two peaks are detected, zeros are returned.
    """
    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 1 or sig.size < 3:
        raise ValueError("signal must be 1D with at least three points")
    peaks = np.where((sig[1:-1] > sig[:-2]) & (sig[1:-1] > sig[2:]))[0] + 1
    if peaks.size < 2:
        return np.zeros(2)
    rr = np.diff(peaks) / fs
    heart_rate = 60.0 / rr.mean()
    sdnn = rr.std()
    return np.array([heart_rate, sdnn])


def arrhythmia_risk(signal: Array, fs: float = 250.0, coeffs: Array = ARRHYTHMIA_COEFFS) -> float:
    """Estimate arrhythmia risk using a linear model on HRV."""
    _, sdnn = ecg_features(signal, fs)
    features = np.array([1.0, sdnn])
    return float(features @ coeffs)


def generate_ecg(n: int = 250, fs: float = 250.0, *, seed: int = 0) -> Array:
    """Generate a synthetic ECG-like waveform."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * 1.0 * t)
    return signal + 0.01 * rng.normal(size=n)


def augment_noise(signal: Array, scale: float = 0.01, *, seed: int = 0) -> Array:
    """Add Gaussian noise to an ECG signal."""
    rng = np.random.default_rng(seed)
    sig = np.asarray(signal, dtype=float)
    return sig + rng.normal(0.0, scale, size=sig.shape)
