"""Audio and speech utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
RHYTHM_WEIGHTS: Array = np.ones(13) / 13


def _dct(x: Array) -> Array:
    """Compute a Type-II DCT using FFT symmetry."""
    n = x.size
    extended = np.concatenate([x, x[::-1]])
    return np.real(np.fft.fft(extended)[:n])


def mfcc_features(signal: Array, sr: float = 16_000.0, n_mfcc: int = 13) -> Array:
    """Compute simple MFCC-like coefficients."""
    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 1 or sig.size == 0:
        raise ValueError("signal must be 1D and non-empty")
    spectrum = np.abs(np.fft.rfft(sig)) ** 2
    mel = np.log(spectrum[: n_mfcc] + 1e-12)
    return _dct(mel)[:n_mfcc]


def rhythm_score(signal: Array, sr: float = 16_000.0) -> float:
    """Estimate rhythmicity via weighted MFCC sum."""
    coeffs = mfcc_features(signal, sr)
    return float(coeffs @ RHYTHM_WEIGHTS)


def generate_tone(
    n: int = 16_000,
    freq: float = 440.0,
    sr: float = 16_000.0,
    *,
    seed: int = 0,
) -> Array:
    """Generate a sinusoidal tone for testing.

    Parameters
    ----------
    n : int, optional
        Number of samples. Default is ``16000``.
    freq : float, optional
        Frequency in Hz. Default is ``440``.
    sr : float, optional
        Sampling rate. Default is ``16000``.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    Array
        Generated waveform.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    tone = np.sin(2 * np.pi * freq * t)
    return tone.astype(float) + rng.normal(0.0, 0.0, size=n)


def augment_time_shift(signal: Array, shift: int) -> Array:
    """Time-shift an audio signal by ``shift`` samples."""
    sig = np.asarray(signal, dtype=float)
    return np.roll(sig, shift)
