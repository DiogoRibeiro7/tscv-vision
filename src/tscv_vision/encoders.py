from __future__ import annotations

import math
from typing import Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

EncoderFunc = Callable[..., Array]
ENCODER_REGISTRY: dict[str, EncoderFunc] = {}


def register_encoder(name: str, func: EncoderFunc) -> None:
    """Register a new encoder under ``name``.

    Parameters
    ----------
    name:
        Identifier used with :func:`get_encoder`.
    func:
        Callable taking a 1D array and returning an image array.
    """

    ENCODER_REGISTRY[name] = func


def get_encoder(name: str) -> EncoderFunc:
    """Return a registered encoder by ``name``.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """

    return ENCODER_REGISTRY[name]


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


def cwt(x: Array, scales: Array, wavelet: Literal["morlet"] = "morlet") -> Array:
    """Continuous Wavelet Transform using a Morlet mother wavelet.

    Parameters
    ----------
    x:
        Input 1D series.
    scales:
        Positive scales at which to compute the transform.
    wavelet:
        Currently only ``"morlet"`` is supported.

    Returns
    -------
    Array
        ``(len(scales), len(x))`` transform magnitudes scaled to ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``scales`` are invalid or ``wavelet`` unsupported.
    """

    x = _validate_series(x)
    scales_arr = np.asarray(scales, dtype=float)
    if scales_arr.ndim != 1 or np.any(scales_arr <= 0):
        raise ValueError("scales must be a 1D array of positive values")
    if wavelet != "morlet":
        raise ValueError("only 'morlet' wavelet is supported")

    n = x.size
    fft_len = int(2 ** math.ceil(math.log2(n * 2)))
    fft_x = np.fft.fft(x, fft_len)
    freqs = np.fft.fftfreq(fft_len)
    out = np.empty((scales_arr.size, n), dtype=float)
    for i, s in enumerate(scales_arr):
        psi_hat = np.exp(-0.5 * (s * 2 * np.pi * freqs - 5.0) ** 2)
        conv = np.fft.ifft(fft_x * psi_hat)
        out[i] = np.abs(conv[:n])
    out = out / (np.max(out) + 1e-12)
    return cast(Array, out)


def persistence_image(x: Array, bins: int = 32) -> Array:
    """Simple persistence diagram histogram.

    Approximates 0D persistent homology by pairing consecutive extrema and
    accumulating their ``(birth, persistence)`` values into a 2D histogram.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    bins:
        Number of histogram bins per axis ``>=1``.

    Returns
    -------
    ndarray
        ``(bins, bins)`` persistence image.

    Raises
    ------
    ValueError
        If ``bins`` is not positive or ``x`` invalid.
    """

    x = _validate_series(x)
    if bins < 1:
        raise ValueError("bins must be >= 1")
    dx = np.diff(x)
    sign = np.sign(dx)
    extrema_idx = np.where(np.diff(sign) != 0)[0] + 1
    if extrema_idx.size < 2:
        return np.zeros((bins, bins), dtype=float)
    births = x[extrema_idx[:-1]]
    deaths = x[extrema_idx[1:]]
    pers = np.abs(deaths - births)
    b_norm = (births - x.min()) / (x.max() - x.min() + 1e-12)
    p_norm = pers / (pers.max() + 1e-12)
    H, _, _ = np.histogram2d(b_norm, p_norm, bins=bins, range=[[0, 1], [0, 1]])
    return cast(Array, H)


def mtf(x: Array, bins: int = 8) -> Array:
    """Markov Transition Field encoding.

    Quantises ``x`` into ``bins`` states and uses the state transition
    probabilities to build an ``N×N`` field.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    bins:
        Number of quantisation bins ``>=2``.

    Returns
    -------
    ndarray
        ``(N, N)`` matrix of transition probabilities.

    Raises
    ------
    ValueError
        If ``bins < 2`` or ``x`` is invalid.
    """

    x = _validate_series(x)
    if bins < 2:
        raise ValueError("bins must be >= 2")
    q = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    states = np.digitize(x, q, right=False)
    trans = np.zeros((bins, bins), dtype=float)
    for i in range(len(states) - 1):
        trans[states[i], states[i + 1]] += 1.0
    # normalise rows to probabilities
    trans /= np.maximum(trans.sum(axis=1, keepdims=True), 1e-12)
    img = trans[states[:, None], states[None, :]]
    return cast(Array, img)


def multi_scale_rp(x: Array, scales: Array) -> Array:
    """Multi-scale Recurrence Plot.

    Computes recurrence plots on downsampled versions of ``x`` and upsamples
    them back to the original size, stacking the results along the first axis.

    Parameters
    ----------
    x:
        Input series ``(N,)``.
    scales:
        1D array of positive integer scales.

    Returns
    -------
    ndarray
        ``(len(scales), N, N)`` stacked recurrence plots.

    Raises
    ------
    ValueError
        If ``scales`` are invalid or exceed ``len(x)``.
    """

    x = _validate_series(x)
    s = np.asarray(scales, dtype=int)
    if s.ndim != 1 or np.any(s <= 0):
        raise ValueError("scales must be a 1D array of positive integers")
    if np.any(s > x.size):
        raise ValueError("scales cannot exceed length of x")
    rps: list[Array] = []
    for scale in s:
        xs = x[::scale]
        rp = recurrence_plot(xs)
        rp_us = np.repeat(np.repeat(rp, scale, axis=0), scale, axis=1)
        rps.append(rp_us[: x.size, : x.size])
    return cast(Array, np.stack(rps, axis=0))


def dtw_matrix(x: Array) -> Array:
    """Dynamic Time Warping cost matrix."""

    x = _validate_series(x)
    n = x.size
    cost = np.empty((n, n), dtype=float)
    cost[0, 0] = 0.0
    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + abs(x[i] - x[0])
        cost[0, i] = cost[0, i - 1] + abs(x[0] - x[i])
    for i in range(1, n):
        xi = x[i]
        for j in range(1, n):
            d = abs(xi - x[j])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    cost = cost / (np.max(cost) + 1e-12)
    return cast(Array, 1.0 - cost)


def sax(x: Array, segments: int = 8, alphabet: int = 8) -> Array:
    """Symbolic Aggregate approXimation image."""

    x = _validate_series(x)
    if segments <= 0 or alphabet < 2:
        raise ValueError("invalid segments or alphabet")
    segs = np.array_split(x, segments)
    means = np.array([seg.mean() for seg in segs])
    bps = np.quantile(means, np.linspace(0, 1, alphabet + 1)[1:-1])
    symbols = np.digitize(means, bps, right=False)
    img = (symbols[:, None] == symbols[None, :]).astype(float)
    return cast(Array, img)


def random_projection_image(x: Array, size: int = 32, seed: int = 0) -> Array:
    """Random projection encoder producing a 2D image.

    Parameters
    ----------
    x:
        1D time series ``(N,)``.
    size:
        Width and height of the output image.
    seed:
        RNG seed controlling the projection matrix.

    Returns
    -------
    ndarray
        ``(size, size)`` image from projecting ``x`` with a random matrix.

    Raises
    ------
    ValueError
        If ``size`` is not positive.
    """

    x = _validate_series(x)
    if size <= 0:
        raise ValueError("size must be positive")
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((size * size, x.size))
    img = proj @ x
    return cast(Array, img.reshape(size, size))


def ensemble(x: Array, names: list[str] | None = None) -> Array:
    """Combine multiple encoders into a stacked representation."""

    if names is None:
        names = ["gaf", "rp"]
    x = _validate_series(x)
    imgs = [get_encoder(n)(x) for n in names]
    shape = imgs[0].shape
    if any(img.shape != shape for img in imgs[1:]):
        raise ValueError("encoders must return images of the same shape")
    return cast(Array, np.stack(imgs, axis=0))


# register built-in encoders
register_encoder("gaf", gaf)
register_encoder("gadf", lambda x: gaf(x, method="difference"))
register_encoder("rp", recurrence_plot)
register_encoder("spec", spectrogram)
register_encoder("cwt", cwt)
register_encoder("ph", persistence_image)
register_encoder("mtf", mtf)
register_encoder("msrp", multi_scale_rp)
register_encoder("dtw", dtw_matrix)
register_encoder("sax", sax)
register_encoder("randproj", random_projection_image)
register_encoder("ensemble", ensemble)


__all__ = [
    "gaf",
    "recurrence_plot",
    "spectrogram",
    "cwt",
    "persistence_image",
    "mtf",
    "multi_scale_rp",
    "dtw_matrix",
    "sax",
    "random_projection_image",
    "ensemble",
    "register_encoder",
    "get_encoder",
]


