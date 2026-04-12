from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray

try:  # optional
    import numba as _nb
except Exception:  # pragma: no cover - optional dependency
    _nb = cast(Any, None)
    _HAS_NUMBA = False
else:
    _HAS_NUMBA = True

try:  # optional Cython extension
    from ._encoders_cy import gaf_polar as _gaf_cy
    from ._encoders_cy import recurrence_dist as _rp_cy
    from ._encoders_cy import spectrogram_stft as _spec_cy
except Exception:  # pragma: no cover - extension not built
    _gaf_cy = cast(Any, None)
    _rp_cy = cast(Any, None)
    _spec_cy = cast(Any, None)
    _HAS_CYTHON = False
else:
    _HAS_CYTHON = True

try:  # optional wavelet backend
    import pywt as _pywt
except Exception:  # pragma: no cover - optional dependency
    _HAS_PYWT = False
else:
    _HAS_PYWT = True

Array = NDArray[np.float64]

NanPolicy = Literal["raise", "omit", "interpolate", "forward_fill"]

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


def _handle_nans(x: Array, policy: NanPolicy) -> Array:
    """Apply ``policy`` to non-finite values in ``x``.

    Parameters
    ----------
    x:
        1D float array (already cast).
    policy:
        ``"raise"`` raises on any non-finite value, ``"omit"`` drops them,
        ``"interpolate"`` fills gaps via linear interpolation, and
        ``"forward_fill"`` carries the last valid observation forward (then
        backward for leading gaps).

    Returns
    -------
    ndarray
        Cleaned 1D array.
    """

    mask = ~np.isfinite(x)
    if not np.any(mask):
        return x
    if policy == "raise":
        raise ValueError("x contains NaN or infinite values")
    if policy == "omit":
        x = x[~mask]
        if x.size == 0:
            raise ValueError("x is empty after removing non-finite values")
        return x
    if policy == "interpolate":
        good = np.where(~mask)[0]
        if good.size == 0:
            raise ValueError("x contains no finite values to interpolate")
        x = x.copy()
        x[mask] = np.interp(np.where(mask)[0], good, x[good])
        return x
    if policy == "forward_fill":
        x = x.copy()
        good = np.where(~mask)[0]
        if good.size == 0:
            raise ValueError("x contains no finite values to forward-fill")
        # forward fill
        for i in range(x.size):
            if mask[i]:
                if i > 0:
                    x[i] = x[i - 1]
        # backward fill remaining leading NaNs
        remaining = ~np.isfinite(x)
        if np.any(remaining):
            first_valid = np.argmax(~remaining)
            x[:first_valid] = x[first_valid]
        return x
    raise ValueError(
        f"nan_policy must be 'raise', 'omit', 'interpolate', or "
        f"'forward_fill', got {policy!r}"
    )


def _validate_series(
    x: Array,
    nan_policy: NanPolicy = "raise",
) -> Array:
    """Return ``x`` as a 1D ``float64`` array after validation.

    Parameters
    ----------
    x:
        Input array.
    nan_policy:
        How to handle NaN and infinite values.  ``"raise"`` (default)
        rejects them, ``"omit"`` removes them, ``"interpolate"`` fills
        gaps via linear interpolation, and ``"forward_fill"`` propagates
        the last valid observation.

    Raises
    ------
    ValueError
        If ``x`` is not 1D, empty, or contains non-finite values when
        ``nan_policy="raise"``.
    """

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    if x.size == 0:
        raise ValueError("x cannot be empty")
    x = _handle_nans(x, nan_policy)
    return x


def _minmax_scale(
    x: Array,
    nan_policy: NanPolicy = "raise",
) -> Array:
    """Scale a validated series to ``[-1, 1]``.

    Adds a small epsilon to avoid zero division.

    Parameters
    ----------
    x:
        1D time series.
    nan_policy:
        Forwarded to :func:`_validate_series`.

    Returns
    -------
    ndarray
        Scaled array in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If the dynamic range of ``x`` is not finite.
    """

    x = _validate_series(x, nan_policy=nan_policy)
    xmin, xmax = x.min(), x.max()
    span = xmax - xmin
    if not np.isfinite(span):
        raise ValueError("Input range is too large")
    if span == 0:
        return np.zeros_like(x)
    x01 = (x - xmin) / span
    return cast(Array, x01 * 2.0 - 1.0)


# ---------------------------------------------------------------------------
# Numba-accelerated helpers
# ---------------------------------------------------------------------------

if _HAS_NUMBA:
    @_nb.njit(cache=True)  # type: ignore[misc]
    def _gaf_numba(x: Array, summation: bool) -> Array:  # pragma: no cover - compiled
        n = x.shape[0]
        phi = np.arccos(np.clip(x, -1.0, 1.0))
        out = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if summation:
                    out[i, j] = math.cos(phi[i] + phi[j])
                else:
                    out[i, j] = math.sin(phi[i] - phi[j])
        return out

    @_nb.njit(cache=True)  # type: ignore[misc]
    def _recurrence_numba(x: Array, eps: float) -> Array:  # pragma: no cover
        n = x.shape[0]
        out = np.empty((n, n), dtype=np.float64)
        maxd = 0.0
        for i in range(n):
            xi = x[i]
            for j in range(n):
                d = abs(xi - x[j])
                out[i, j] = d
                if d > maxd:
                    maxd = d
        scale = 1.0 / (maxd + 1e-12)
        if eps >= 0.0:
            for i in range(n):
                for j in range(n):
                    out[i, j] = 1.0 if out[i, j] * scale <= eps else 0.0
        else:
            for i in range(n):
                for j in range(n):
                    out[i, j] = 1.0 - out[i, j] * scale
        return out

    @_nb.njit(cache=True)  # type: ignore[misc]
    def _spectrogram_frames(
        x: Array, win: int, hop: int, w: Array, n_frames: int
    ) -> Array:  # pragma: no cover
        out = np.empty((n_frames, win), dtype=np.float64)
        for n in range(n_frames):
            start = n * hop
            for i in range(win):
                out[n, i] = x[start + i] * w[i]
        return out
else:  # pragma: no cover - no numba
    _gaf_numba = None
    _recurrence_numba = None
    _spectrogram_frames = None


def _sliding_view(x: Array, win: int, hop: int) -> tuple[Array, int, Array]:
    """Return a memory-efficient sliding window view of ``x``.

    Parameters
    ----------
    x:
        Input series.
    win:
        Window length.
    hop:
        Step between windows.

    Returns
    -------
    view, n_frames, padded:
        Read-only view into the (possibly padded) series with shape
        ``(n_frames, win)``, the number of frames and the padded array used to
        create the view.
    """

    n_frames = 1 + math.ceil((len(x) - win) / hop)
    total_len = (n_frames - 1) * hop + win
    if len(x) < total_len:
        x = np.pad(x, (0, total_len - len(x)))
    view = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, win),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    )
    return view, n_frames, x


def gaf(
    x: Array,
    method: Literal["summation", "difference"] = "summation",
    *,
    nan_policy: NanPolicy = "raise",
    use_numba: bool = False,
    use_cython: bool = False,
    use_gpu: bool = False,
    gpu_device: int | None = None,
    gpu_mem_limit: int | None = None,
) -> Array:
    """Gramian Angular Field (GAF) encoding.

    Maps a 1D series to a Gramian image using a polar transform and the
    cosine rule. Setting ``use_numba`` or ``use_cython`` to ``True`` enables
    compiled implementations for speed when the optional dependencies are
    available.  The compiled paths are regression-tested to match the pure
    NumPy implementation within ``1e-6`` absolute and relative tolerance.

    Parameters
    ----------
    x:
        1D time series ``(N,)``.
    method:
        ``"summation"`` (GASF) or ``"difference"`` (GADF).
    use_numba:
        Whether to run a Numba JIT-compiled variant (if available).
    use_cython:
        Whether to run the Cython extension variant (if available).
    use_gpu:
        If ``True`` and CuPy is installed, compute on the GPU.
    gpu_device:
        Optional GPU device index when ``use_gpu`` is ``True``.
    gpu_mem_limit:
        If provided and ``use_gpu`` is ``True``, limit GPU memory usage in
        bytes when forming the Gramian matrix by chunking the computation.

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

    x = _minmax_scale(x, nan_policy=nan_policy)
    summation = method == "summation"
    if use_gpu:
        try:
            from .gpu.encoders import gaf as _gaf_gpu
        except Exception:  # pragma: no cover - optional path
            pass
        else:
            try:
                return _gaf_gpu(
                    x,
                    method=method,
                    device=gpu_device,
                    mem_limit=gpu_mem_limit,
                )
            except RuntimeError:
                pass
    if use_cython and _HAS_CYTHON:
        return cast(Array, _gaf_cy(x, summation))
    if use_numba and _HAS_NUMBA:
        return cast(Array, _gaf_numba(x, summation))
    phi = np.arccos(np.clip(x, -1.0, 1.0))
    phi_i = phi[:, None]
    phi_j = phi[None, :]
    if summation:
        return cast(Array, np.cos(phi_i + phi_j))
    if method == "difference":
        return cast(Array, np.sin(phi_i - phi_j))
    raise ValueError("method must be 'summation' or 'difference'")


def recurrence_plot(
    x: Array,
    metric: Literal["euclidean", "manhattan"] = "euclidean",
    eps: float | None = None,
    *,
    nan_policy: NanPolicy = "raise",
    use_numba: bool = False,
    use_cython: bool = False,
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
    use_numba:
        If ``True`` and `numba` is installed, compute distances via a
        compiled implementation.
    use_cython:
        If ``True`` and the Cython extension is present, it is preferred over
        the Numba and NumPy versions. Both compiled options are
        regression-tested to match the NumPy baseline within ``1e-6`` absolute
        and relative tolerance.

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

    x = _validate_series(x, nan_policy=nan_policy)
    x1 = x.reshape(-1)
    if metric not in {"euclidean", "manhattan"}:
        raise ValueError("metric must be 'euclidean' or 'manhattan'")
    if use_cython and _HAS_CYTHON:
        e = -1.0 if eps is None else eps
        return cast(Array, _rp_cy(x1, e))
    if use_numba and _HAS_NUMBA:
        e = -1.0 if eps is None else eps
        return cast(Array, _recurrence_numba(x1, e))
    diffs = x1[:, None] - x1[None, :]
    dist = np.abs(diffs)
    if not np.all(np.isfinite(dist)):
        raise ValueError("Input range is too large")
    dist = dist / (np.nanmax(dist) + 1e-12)
    if eps is None:
        return cast(Array, 1.0 - dist)
    if not np.isfinite(eps) or not 0.0 <= eps <= 1.0:
        raise ValueError("eps must be in [0, 1]")
    return cast(Array, (dist <= eps).astype(float))


def spectrogram(
    x: Array,
    win: int = 64,
    hop: int | None = None,
    window: Literal["hann", "rect"] = "hann",
    *,
    nan_policy: NanPolicy = "raise",
    use_numba: bool = False,
    use_cython: bool = False,
    use_gpu: bool = False,
    gpu_device: int | None = None,
) -> Array:
    """Very small STFT-based magnitude spectrogram.

    Pads the signal with zeros so all samples are covered by a window. Set
    ``use_numba`` or ``use_cython`` to ``True`` to enable compiled
    implementations of the STFT core when available.  The compiled backends
    are verified to agree with the NumPy reference within ``1e-6`` absolute
    and relative tolerance.

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
    use_numba:
        If ``True`` and `numba` is installed, compute the STFT using a
        compiled loop.
    use_cython:
        If ``True`` and the Cython extension is available, use it instead of
        the NumPy/Numba versions.
    use_gpu:
        If ``True`` and CuPy is installed, compute on the GPU.
    gpu_device:
        Optional GPU device index when ``use_gpu`` is ``True``.

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
    x = _validate_series(x, nan_policy=nan_policy)
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

    frames, n_frames, x_pad = _sliding_view(x, win, hop)

    mag: Array | None = None
    if use_gpu:
        try:
            from .gpu.encoders import spectrogram as _spec_gpu
        except Exception:  # pragma: no cover - optional path
            pass
        else:
            try:
                mag = _spec_gpu(x_pad, win, hop, window=window, device=gpu_device)
            except RuntimeError:
                mag = None
    if mag is None and use_cython and _HAS_CYTHON:
        mag = _spec_cy(x_pad, w, win, hop, n_frames)
    elif mag is None and use_numba and _HAS_NUMBA:
        frames = _spectrogram_frames(x_pad, win, hop, w, n_frames)
        fft = np.fft.rfft(frames, n=win, axis=1)
        mag = np.abs(fft).T
    elif mag is None:
        frames = frames * w[None, :]
        fft = np.fft.rfft(frames, n=win, axis=1)
        mag = np.abs(fft).T  # (F,T)
    if not np.all(np.isfinite(mag)):
        raise ValueError("Input range is too large")
    mag = mag / (np.max(mag) + 1e-12)
    return mag


def cwt(
    x: Array,
    scales: Array,
    wavelet: Literal["morlet", "mexh", "ricker"] = "morlet",
    *,
    nan_policy: NanPolicy = "raise",
) -> Array:
    """Continuous Wavelet Transform.

    Implements a CWT compatible with PyWavelets. The FFT-based Morlet
    implementation follows the formulation of Torrence and Compo (1998).

    Parameters
    ----------
    x:
        Input 1D series.
    scales:
        Positive scales at which to compute the transform.
    wavelet:
        Mother wavelet name. ``"morlet"`` uses a fast FFT implementation
        while other families require ``PyWavelets`` and fall back to its
        implementation.

    Returns
    -------
    Array
        ``(len(scales), len(x))`` transform magnitudes scaled to ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``scales`` are invalid or ``wavelet`` unsupported.
    """

    x = _validate_series(x, nan_policy=nan_policy)
    scales_arr = np.asarray(scales, dtype=float)
    if scales_arr.ndim != 1 or np.any(scales_arr <= 0):
        raise ValueError("scales must be a 1D array of positive values")
    if wavelet == "morlet" and not _HAS_PYWT:
        # custom FFT-based morlet for zero-dependency environments
        n = x.size
        fft_len = int(2 ** math.ceil(math.log2(n * 2)))
        fft_x = np.fft.fft(x, fft_len)
        freqs = np.fft.fftfreq(fft_len)
        out = np.empty((scales_arr.size, n), dtype=float)
        for i, s in enumerate(scales_arr):
            psi_hat = np.exp(-0.5 * (s * 2 * np.pi * freqs - 5.0) ** 2)
            conv = np.fft.ifft(fft_x * psi_hat)
            out[i] = np.abs(conv[:n])
    else:
        if not _HAS_PYWT:
            raise ValueError("PyWavelets required for chosen wavelet")
        coeffs, _ = _pywt.cwt(x, scales_arr, wavelet)
        out = np.abs(coeffs)
    out = out / (np.max(out) + 1e-12)
    return cast(Array, out)


def persistence_image(x: Array, bins: int = 32, *, nan_policy: NanPolicy = "raise") -> Array:
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

    x = _validate_series(x, nan_policy=nan_policy)
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


def mtf(
    x: Array,
    bins: int = 8,
    weighted: bool = False,
    *,
    nan_policy: NanPolicy = "raise",
) -> Array:
    """Markov Transition Field encoding.

    Quantises ``x`` into ``bins`` states and accumulates state transitions
    into a Markov matrix which is then expanded to an ``N×N`` field.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    bins:
        Number of quantisation bins ``>=2``.
    weighted:
        If ``True`` the transition counts are weighted by the absolute
        difference between successive samples, emphasising large jumps.

    Returns
    -------
    ndarray
        ``(N, N)`` matrix of transition probabilities.

    Raises
    ------
    ValueError
        If ``bins < 2`` or ``x`` is invalid.
    """

    x = _validate_series(x, nan_policy=nan_policy)
    if bins < 2:
        raise ValueError("bins must be >= 2")
    q = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    states = np.digitize(x, q, right=False)
    trans = np.zeros((bins, bins), dtype=float)
    for i in range(len(states) - 1):
        w = abs(x[i + 1] - x[i]) if weighted else 1.0
        trans[states[i], states[i + 1]] += w
    # normalise rows to probabilities
    trans /= np.maximum(trans.sum(axis=1, keepdims=True), 1e-12)
    img = trans[states[:, None], states[None, :]]
    return cast(Array, img)


def gdf(x: Array, *, nan_policy: NanPolicy = "raise") -> Array:
    """Gramian Difference Field.

    Constructs a matrix of pairwise differences on a min-max scaled series,
    highlighting relative changes between all time indices.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.

    Returns
    -------
    ndarray
        ``(N, N)`` matrix with values in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If ``x`` is invalid.
    """

    z = _minmax_scale(x, nan_policy=nan_policy)
    diff = z[None, :] - z[:, None]
    m = np.max(np.abs(diff)) + 1e-12
    return cast(Array, diff / m)


def multi_scale_rp(x: Array, scales: Array, *, nan_policy: NanPolicy = "raise") -> Array:
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

    x = _validate_series(x, nan_policy=nan_policy)
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


def dtw_matrix(x: Array, *, nan_policy: NanPolicy = "raise") -> Array:
    """Dynamic Time Warping cost matrix."""

    x = _validate_series(x, nan_policy=nan_policy)
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


def sax(
    x: Array,
    segments: int = 8,
    alphabet: int = 8,
    *,
    nan_policy: NanPolicy = "raise",
) -> Array:
    """Symbolic Aggregate approXimation image."""

    x = _validate_series(x, nan_policy=nan_policy)
    if segments <= 0 or alphabet < 2:
        raise ValueError("invalid segments or alphabet")
    segs = np.array_split(x, segments)
    means = np.array([seg.mean() for seg in segs])
    bps = np.quantile(means, np.linspace(0, 1, alphabet + 1)[1:-1])
    symbols = np.digitize(means, bps, right=False)
    img = (symbols[:, None] == symbols[None, :]).astype(float)
    return cast(Array, img)


def multi_scale_conv(
    x: Array,
    kernels: Sequence[int] = (3, 5, 7),
    *,
    nan_policy: NanPolicy = "raise",
) -> Array:
    """Multi-scale convolutional encoder.

    Applies simple average kernels of varying sizes and stacks the resulting
    responses, following the idea of multi-resolution temporal filtering.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    kernels:
        Iterable of odd kernel sizes ``>=1``.

    Returns
    -------
    ndarray
        ``(len(kernels), N)`` stack of filtered signals scaled to ``[0, 1]``.

    Raises
    ------
    ValueError
        If kernels are invalid or exceed ``len(x)``.
    """

    x = _validate_series(x, nan_policy=nan_policy)
    ks = np.asarray(kernels, dtype=int)
    if ks.ndim != 1 or np.any(ks <= 0) or np.any(ks > x.size):
        raise ValueError("kernels must be positive integers within len(x)")
    outs: list[Array] = []
    for k in ks:
        ker = np.ones(k, dtype=float) / k
        conv = np.convolve(x, ker, mode="same")
        outs.append(conv)
    arr = np.stack(outs, axis=0)
    arr = arr - arr.min()
    arr /= np.max(arr) + 1e-12
    return cast(Array, arr)


def tpa(x: Array, window: int = 8, *, nan_policy: NanPolicy = "raise") -> Array:
    """Temporal Pattern Attention encoder.

    Implements a simplified self-attention mechanism over sliding windows as
    described by Li *et al.* (2019), yielding an attention matrix between
    local temporal patterns.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    window:
        Length of each local pattern ``>=1`` and ``<= N``.

    Returns
    -------
    ndarray
        ``(N - window + 1, N - window + 1)`` attention matrix with rows
        normalised to sum to ``1``.

    Raises
    ------
    ValueError
        If ``window`` is invalid or ``x`` invalid.
    """

    x = _validate_series(x, nan_policy=nan_policy)
    if window < 1 or window > x.size:
        raise ValueError("window must be in [1, len(x)]")
    windows = np.lib.stride_tricks.sliding_window_view(x, window)
    scores = windows @ windows.T / math.sqrt(window)
    scores -= scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    attn = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    return cast(Array, attn)


def visibility_graph(x: Array, *, nan_policy: NanPolicy = "raise") -> Array:
    """Natural visibility graph adjacency matrix.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.

    Returns
    -------
    ndarray
        ``(N, N)`` binary adjacency matrix where ``1`` denotes visibility.

    Raises
    ------
    ValueError
        If ``x`` is invalid.

    Examples
    --------
    >>> x = np.array([0.0, 1.0, 0.5])
    >>> visibility_graph(x).shape
    (3, 3)
    """

    x = _validate_series(x, nan_policy=nan_policy)
    n = x.size
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            yi, yj = x[i], x[j]
            ks = slice(i + 1, j)
            if ks.start >= ks.stop:
                adj[i, j] = adj[j, i] = 1.0
                continue
            k_idx = np.arange(1, j - i)
            y_line = yi + (yj - yi) * (k_idx / (j - i))
            if np.all(x[ks] < y_line):
                adj[i, j] = adj[j, i] = 1.0
    return cast(Array, adj)


def shapelet_transform(
    x: Array,
    k: int = 3,
    length: int | None = None,
    seed: int | None = None,
    *,
    nan_policy: NanPolicy = "raise",
) -> Array:
    """Distance maps to randomly sampled shapelets.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    k:
        Number of shapelets to sample ``>=1``.
    length:
        Length of each shapelet; defaults to ``len(x) // k``.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    ndarray
        ``(k, N - length + 1)`` distance maps scaled to ``[0, 1]``.

    Raises
    ------
    ValueError
        If parameters are invalid or ``x`` is invalid.

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 1, 20))
    >>> shapelet_transform(x, k=2, length=5, seed=0).shape
    (2, 16)
    """

    x = _validate_series(x, nan_policy=nan_policy)
    n = x.size
    if k < 1:
        raise ValueError("k must be >= 1")
    if length is None:
        length = max(1, n // (k + 1))
    if length < 1 or length > n:
        raise ValueError("length must be in [1, len(x)]")
    windows = np.lib.stride_tricks.sliding_window_view(x, length)
    n_windows = windows.shape[0]
    if k > n_windows:
        raise ValueError("k cannot exceed number of subsequences")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_windows, size=k, replace=False)
    shapelets = windows[idx]
    dists = np.sqrt(((windows[None, :, :] - shapelets[:, None, :]) ** 2).sum(axis=2))
    dists = dists / (np.max(dists) + 1e-12)
    return cast(Array, dists)


def matrix_profile(x: Array, m: int, *, nan_policy: NanPolicy = "raise") -> Array:
    """Naive matrix profile for motif/discord discovery.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    m:
        Subsequence length ``>=2``.

    Returns
    -------
    ndarray
        Matrix profile ``(N - m + 1,)`` scaled to ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``m`` is not in ``[2, len(x)]`` or ``x`` invalid.

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 4 * np.pi, 32))
    >>> matrix_profile(x, m=4).shape
    (29,)
    """

    x = _validate_series(x, nan_policy=nan_policy)
    n = x.size
    if m < 2 or m > n:
        raise ValueError("m must be in [2, len(x)]")
    windows = np.lib.stride_tricks.sliding_window_view(x, m)
    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True)
    z = (windows - means) / (stds + 1e-12)
    n_sub = z.shape[0]
    profile = np.empty(n_sub, dtype=float)
    excl = m // 2
    for i in range(n_sub):
        dist = np.linalg.norm(z[i] - z, axis=1)
        lo = max(0, i - excl)
        hi = min(n_sub, i + excl + 1)
        dist[lo:hi] = np.inf
        profile[i] = dist.min()
    profile = profile / (np.max(profile) + 1e-12)
    return cast(Array, profile)


def random_projection_image(
    x: Array,
    size: int = 32,
    seed: int = 0,
    *,
    nan_policy: NanPolicy = "raise",
) -> Array:
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

    x = _validate_series(x, nan_policy=nan_policy)
    if size <= 0:
        raise ValueError("size must be positive")
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((size * size, x.size))
    img = proj @ x
    return cast(Array, img.reshape(size, size))


def ensemble(
    x: Array,
    names: Sequence[str] | None = None,
    *,
    nan_policy: NanPolicy = "raise",
    weights: Sequence[float] | None = None,
    aggregate: Literal["stack", "mean"] = "stack",
) -> Array:
    """Combine multiple encoders into a stacked or averaged representation."""

    if names is None:
        names = ["gaf", "rp"]
    x = _validate_series(x, nan_policy=nan_policy)
    imgs = [get_encoder(n)(x) for n in names]
    shape = imgs[0].shape
    if any(img.shape != shape for img in imgs[1:]):
        raise ValueError("encoders must return images of the same shape")
    arr = np.stack(imgs, axis=0)
    if aggregate == "stack":
        return arr
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != arr.shape[0]:
            raise ValueError("weights length must match encoders")
        w = w / np.sum(w)
        return cast(Array, np.tensordot(w, arr, axes=(0, 0)))
    return cast(Array, arr.mean(axis=0))


# register built-in encoders
register_encoder("gaf", gaf)
register_encoder("gadf", lambda x: gaf(x, method="difference"))
register_encoder("rp", recurrence_plot)
register_encoder("spec", spectrogram)
register_encoder("cwt", cwt)
register_encoder("ph", persistence_image)
register_encoder("mtf", mtf)
register_encoder("gdf", gdf)
register_encoder("msrp", multi_scale_rp)
register_encoder("dtw", dtw_matrix)
register_encoder("sax", sax)
register_encoder("msc", multi_scale_conv)
register_encoder("tpa", tpa)
register_encoder("vg", visibility_graph)
register_encoder("shapelet", shapelet_transform)
register_encoder("mp", matrix_profile)
register_encoder("randproj", random_projection_image)
register_encoder("ensemble", ensemble)

# aliases matching documentation
register_encoder("visibility_graph", visibility_graph)
register_encoder("matrix_profile", matrix_profile)
register_encoder("persistence_image", persistence_image)


__all__ = [
    "gaf",
    "recurrence_plot",
    "spectrogram",
    "cwt",
    "persistence_image",
    "mtf",
    "gdf",
    "multi_scale_rp",
    "multi_scale_conv",
    "tpa",
    "dtw_matrix",
    "sax",
    "visibility_graph",
    "shapelet_transform",
    "matrix_profile",
    "random_projection_image",
    "ensemble",
    "register_encoder",
    "get_encoder",
    "ENCODER_REGISTRY",
    "NanPolicy",
]


