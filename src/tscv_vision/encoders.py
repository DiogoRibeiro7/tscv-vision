from __future__ import annotations

import math
from typing import Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _validate_series(x: Array) -> Array:
    """Return ``x`` as a 1D ``float64`` array after validation.

    Parameters
    ----------
    x:
        Input array.

    Raises
    ------
    ValueError
        If ``x`` is not 1D, empty or contains NaN/inf values.
    """

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    if x.size == 0:
        raise ValueError("x cannot be empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN or infinite values")
    return x


def _minmax_scale(x: Array) -> Array:
    """Scale a validated series to ``[-1, 1]``.

    Adds a small epsilon to avoid zero division.

    Parameters
    ----------
    x:
        1D time series.

    Returns
    -------
    ndarray
        Scaled array in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If the dynamic range of ``x`` is not finite.
    """

    x = _validate_series(x)
    xmin, xmax = x.min(), x.max()
    span = xmax - xmin
    if not np.isfinite(span):
        raise ValueError("Input range is too large")
    if span == 0:
        return np.zeros_like(x)
    x01 = (x - xmin) / span
    return cast(Array, x01 * 2.0 - 1.0)


def gaf(x: Array, method: Literal["summation", "difference"] = "summation") -> Array:
    """Gramian Angular Field (GAF) encoding.

    Maps a 1D series to a Gramian image using a polar transform and the
    cosine rule.

    Parameters
    ----------
    x:
        1D time series ``(N,)``.
    method:
        ``"summation"`` (GASF) or ``"difference"`` (GADF).

    Returns
    -------
    ndarray
        ``(N, N)`` image with values in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If ``x`` is empty, not 1D or contains NaN/inf values.
    ValueError
        If ``method`` is not one of ``{"summation", "difference"}``.
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

    Parameters
    ----------
    x:
        1D time series ``(N,)``.
    metric:
        Distance metric to use.
    eps:
        Optional threshold; if set, returns a binary RP (0/1), otherwise
        distances normalised to ``[0, 1]``.

    Returns
    -------
    ndarray
        ``(N, N)`` recurrence matrix.

    Raises
    ------
    ValueError
        If ``x`` is empty, not 1D or contains NaN/inf values.
    ValueError
        If ``metric`` is not ``"euclidean"`` or ``"manhattan"``.
    ValueError
        If ``eps`` is provided and is not finite or outside ``[0, 1]``.
    """

    x = _validate_series(x)
    x = x.reshape(-1, 1)
    diffs = x - x.T  # (N,N)
    if metric == "euclidean":
        dist = np.abs(diffs)
    elif metric == "manhattan":
        dist = np.abs(diffs)
    else:
        raise ValueError("metric must be 'euclidean' or 'manhattan'")
    if not np.all(np.isfinite(dist)):
        raise ValueError("Input range is too large")
    dist = dist / (np.nanmax(dist) + 1e-12)
    if eps is None:
        return cast(Array, 1.0 - dist)  # similarity map
    if not np.isfinite(eps) or not 0.0 <= eps <= 1.0:
        raise ValueError("eps must be in [0, 1]")
    return cast(Array, (dist <= eps).astype(float))


def spectrogram(
    x: Array, win: int = 64, hop: int | None = None, window: Literal["hann", "rect"] = "hann"
) -> Array:
    """Very small STFT-based magnitude spectrogram using NumPy only.

    Pads the signal with zeros so all samples are covered by a window.

    Parameters
    ----------
    x:
        1D array ``(N,)``.
    win:
        Window length (``>=8``).
    hop:
        Hop length (defaults to ``win//4``).
    window:
        Window type (``"hann"`` or ``"rect"``).

    Returns
    -------
    ndarray
        Magnitude spectrogram ``(win//2 + 1, n_frames)`` scaled to ``[0, 1]`` where
        ``n_frames = ceil((N - win) / hop) + 1``.

    Raises
    ------
    ValueError
        If ``x`` is empty, not 1D or contains NaN/inf values.
    ValueError
        If ``win < 8``.
    ValueError
        If ``hop`` is non-positive or greater than ``win``.
    ValueError
        If ``len(x) < win``.
    ValueError
        If ``window`` is not ``'hann'`` or ``'rect'``.
    """
    x = _validate_series(x)
    if hop is None:
        hop = max(1, win // 4)
    if win < 8:
        raise ValueError("win must be >= 8")
    if hop <= 0:
        raise ValueError("hop must be > 0")
    if hop > win:
        raise ValueError("hop must be <= win")
    if len(x) < win:
        raise ValueError("len(x) must be >= win")
    if window == "hann":
        w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)
    elif window == "rect":
        w = np.ones(win)
    else:
        raise ValueError("window must be 'hann' or 'rect'")

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
    if not np.all(np.isfinite(mag)):
        raise ValueError("Input range is too large")
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
