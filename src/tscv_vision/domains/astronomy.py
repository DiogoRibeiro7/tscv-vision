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


def generate_light_curve(
    n: int = 100,
    period: int = 50,
    noise: float = 0.01,
    *,
    seed: int = 0,
) -> Array:
    """Generate a synthetic sinusoidal light curve.

    Parameters
    ----------
    n : int, optional
        Number of samples. Default is ``100``.
    period : int, optional
        Period of the underlying sinusoid. Default is ``50``.
    noise : float, optional
        Standard deviation of Gaussian noise added to the signal.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    Array
        Generated light curve of shape ``(n,)``.

    Examples
    --------
    >>> lc = generate_light_curve(10)
    >>> lc.shape
    (10,)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    signal = np.sin(2 * np.pi * t / period)
    return signal + rng.normal(0.0, noise, size=n)


def augment_noise(signal: Array, scale: float = 0.01, *, seed: int = 0) -> Array:
    """Inject Gaussian noise into a light curve.

    Parameters
    ----------
    signal : Array
        Original light curve of shape ``(N,)``.
    scale : float, optional
        Standard deviation of the added noise.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    Array
        Augmented light curve.
    """
    rng = np.random.default_rng(seed)
    sig = np.asarray(signal, dtype=float)
    return sig + rng.normal(0.0, scale, size=sig.shape)
