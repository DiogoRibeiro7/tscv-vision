from __future__ import annotations
import math
from typing import Literal
import numpy as np

Array = np.ndarray


def _minmax_scale(x: Array) -> Array:
    """Scale to [-1, 1] robustly. Adds small eps to avoid zero-division.

    Args:
        x: 1D array-like time series.
    Returns:
        Scaled array in [-1, 1].
    """
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise ValueError("Input contains non-finite values.")
    if xmax == xmin:
        return np.zeros_like(x)
    x01 = (x - xmin) / (xmax - xmin)
    return x01 * 2.0 - 1.0


def gaf(x: Array, method: Literal["summation", "difference"] = "summation") -> Array:
    """Gramian Angular Field (GAF) encoding.

    Maps a 1D series to a Gramian image using polar transform and cosine rule.

    Args:
        x: 1D time series (N,).
        method: "summation" (GASF) or "difference" (GADF).
    Returns:
        (N, N) image with values in [-1, 1].
    """
    x = _minmax_scale(x)
    # Polar encoding
    phi = np.arccos(np.clip(x, -1.0, 1.0))  # angle
    # Time radii scaled to [0,1]
    n = x.shape[0]
    r = np.linspace(0.0, 1.0, n)

    # Outer sums/diffs of angles
    phi_i = phi[:, None]
    phi_j = phi[None, :]
    if method == "summation":
        return np.cos(phi_i + phi_j)
    elif method == "difference":
        return np.sin(phi_i - phi_j)
    else:
        raise ValueError("method must be 'summation' or 'difference'")


def recurrence_plot(
    x: Array, metric: Literal["euclidean", "manhattan"] = "euclidean", eps: float | None = None
) -> Array:
    """Binary/real-valued recurrence plot.

    Args:
        x: 1D series (N,).
        metric: distance metric.
        eps: optional threshold; if provided returns binary RP (0/1), else distances normalized to [0,1].
    Returns:
        (N, N) array RP.
    """
    x = np.asarray(x, dtype=float)
    x = x.reshape(-1, 1)
    diffs = x - x.T  # (N,N)
    if metric == "euclidean":
        dist = np.sqrt(diffs * diffs)
    elif metric == "manhattan":
        dist = np.abs(diffs)
    else:
        raise ValueError("Unsupported metric")
    dist = dist / (np.nanmax(dist) + 1e-12)
    if eps is None:
        return 1.0 - dist  # similarity map
    return (dist <= eps).astype(float)


def spectrogram(
    x: Array, win: int = 64, hop: int | None = None, window: Literal["hann", "rect"] = "hann"
) -> Array:
    """Very small STFT-based magnitude spectrogram using NumPy only.

    Args:
        x: 1D array (N,)
        win: window length (>=8)
        hop: hop length (defaults to win//4)
        window: window type ("hann" or "rect")
    Returns:
        (F, T) magnitude spectrogram scaled to [0,1]
    """
    x = np.asarray(x, dtype=float)
    if hop is None:
        hop = max(1, win // 4)
    if win < 8:
        raise ValueError("win must be >= 8")
    if window == "hann":
        w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)
    elif window == "rect":
        w = np.ones(win)
    else:
        raise ValueError("Unsupported window")

    n_frames = 1 + max(0, (len(x) - win) // hop)
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, win),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    )
    frames = frames * w[None, :]
    fft = np.fft.rfft(frames, n=win, axis=1)
    mag = np.abs(fft).T  # (F,T)
    mag = mag / (np.max(mag) + 1e-12)
    return mag
