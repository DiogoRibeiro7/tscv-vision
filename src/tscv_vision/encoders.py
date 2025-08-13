from __future__ import annotations

import math
from typing import Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


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
    # Outer sums/diffs of angles
    phi_i = phi[:, None]
    phi_j = phi[None, :]
    if method == "summation":
        return cast(Array, np.cos(phi_i + phi_j))
    if method == "difference":
        return cast(Array, np.sin(phi_i - phi_j))
    raise ValueError("method must be 'summation' or 'difference'")


def recurrence_plot(
    x: Array, metric: Literal["euclidean", "manhattan"] = "euclidean", eps: float | None = None
) -> Array:
    """Binary/real-valued recurrence plot.

    Args:
        x: 1D series (N,).
        metric: Distance metric.
        eps: Optional threshold; if set returns binary RP (0/1), else
            distances normalized to [0, 1].
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
        return cast(Array, 1.0 - dist)  # similarity map
    return cast(Array, (dist <= eps).astype(float))


def spectrogram(
    x: Array, win: int = 64, hop: int | None = None, window: Literal["hann", "rect"] = "hann"
) -> Array:
    """Very small STFT-based magnitude spectrogram using NumPy only.

    Pads the signal with zeros so all samples are covered by a window.

    Args:
        x: 1D array ``(N,)``.
        win: Window length (``>=8``).
        hop: Hop length (defaults to ``win//4``).
        window: Window type (``"hann"`` or ``"rect"``).
    Returns:
        Magnitude spectrogram ``(win//2 + 1, n_frames)`` scaled to ``[0, 1]`` where
        ``n_frames = ceil((N - win) / hop) + 1``.
    """
    x = np.asarray(x, dtype=float)
    if hop is None:
        hop = max(1, win // 4)
    if win < 8:
        raise ValueError("win must be >= 8")
    if len(x) < win:
        raise ValueError("len(x) must be >= win")
    if window == "hann":
        w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)
    elif window == "rect":
        w = np.ones(win)
    else:
        raise ValueError("Unsupported window")

    n_frames = 1 + math.ceil((len(x) - win) / hop)
    total_len = (n_frames - 1) * hop + win
    if len(x) < total_len:
        x = np.pad(x, (0, total_len - len(x)))

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
    return cast(Array, mag)


# ---------------------------------------------------------------------------
# Encoder registry

ENCODER_REGISTRY: dict[str, Callable[..., Array]] = {
    "gaf": gaf,
    "gadf": lambda x, **k: gaf(x, method="difference"),
    "rp": recurrence_plot,
    "spec": spectrogram,
}


def register_encoder(name: str, func: Callable[..., Array]) -> None:
    """Register a new encoder function.

    Parameters
    ----------
    name:
        Unique encoder identifier.
    func:
        Callable with signature ``(x: Array, **kwargs) -> Array``.
    """

    if name in ENCODER_REGISTRY:
        raise ValueError(f"Encoder '{name}' already registered")
    ENCODER_REGISTRY[name] = func


__all__ = [
    "gaf",
    "recurrence_plot",
    "spectrogram",
    "ENCODER_REGISTRY",
    "register_encoder",
]
