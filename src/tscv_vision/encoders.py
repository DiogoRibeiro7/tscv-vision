from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from ._deprecation import deprecated_alias

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
    # numba.njit is untyped upstream, so every use needs the ignore; the code
    # is `untyped-decorator` whether or not numba is installed, because the
    # fallback binds `_nb` to Any.
    _njit: Any = _nb.njit(cache=True)

    @_njit  # type: ignore[untyped-decorator]
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

    @_njit  # type: ignore[untyped-decorator]
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

    @_njit  # type: ignore[untyped-decorator]
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
        except ImportError:  # pragma: no cover - optional path
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
        except ImportError:  # pragma: no cover - optional path
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
    # Map user-facing names to pywt equivalents (pywt >= 1.6 uses "morl")
    _PYWT_NAMES: dict[str, str] = {"morlet": "morl", "ricker": "mexh"}

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
        pywt_name = _PYWT_NAMES.get(wavelet, wavelet)
        coeffs, _ = _pywt.cwt(x, scales_arr, pywt_name)
        out = np.abs(coeffs)
    out = out / (np.max(out) + 1e-12)
    return cast(Array, out)


def persistence_diagram(
    x: Array,
    *,
    include_infinite: bool = False,
    nan_policy: NanPolicy = "raise",
) -> Array:
    """0-dimensional sublevel-set persistence diagram of a 1D series.

    Computes the exact degree-0 persistent homology of the lower-star
    filtration of ``x``: sweeping the threshold upwards, each local minimum
    creates a connected component and each local maximum merges two
    components. By the elder rule the younger component (the one with the
    larger birth value) dies at the merge. This is the standard sublevel-set
    filtration used for time-series topological data analysis and agrees with
    Ripser's ``H0`` on the corresponding lower-star complex.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    include_infinite:
        If ``True`` the essential class (global minimum, never dies) is
        appended with ``death = inf``. Default ``False``, matching what
        vectorisations such as :func:`persistence_image` can consume.
    nan_policy:
        How to treat NaNs, see :func:`_validate_series`.

    Returns
    -------
    ndarray
        ``(n_pairs, 2)`` array of ``(birth, death)`` values sorted by birth.
        May be empty (shape ``(0, 2)``) for a monotone series.

    Raises
    ------
    ValueError
        If ``x`` is invalid.

    Examples
    --------
    >>> persistence_diagram(np.array([0.0, 3.0, 1.0, 4.0]))
    array([[1., 3.]])
    """

    x = _validate_series(x, nan_policy=nan_policy)
    n = x.size
    parent = np.full(n, -1, dtype=np.int64)  # -1 marks "not yet added"
    birth = np.empty(n, dtype=float)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = int(parent[root])
        while parent[i] != root:  # path compression
            parent[i], i = root, int(parent[i])
        return root

    order = np.argsort(x, kind="stable")
    pairs: list[tuple[float, float]] = []
    for idx in order:
        i = int(idx)
        parent[i] = i
        birth[i] = x[i]
        for j in (i - 1, i + 1):
            if j < 0 or j >= n or parent[j] == -1:
                continue
            root_i, root_j = find(i), find(j)
            if root_i == root_j:
                continue
            # Elder rule: the component born later dies at the current value.
            if birth[root_i] <= birth[root_j]:
                older, younger = root_i, root_j
            else:
                older, younger = root_j, root_i
            if birth[younger] < x[i]:
                pairs.append((float(birth[younger]), float(x[i])))
            parent[younger] = older
    if include_infinite and n > 0:
        pairs.append((float(np.min(x)), float("inf")))
    if not pairs:
        return np.zeros((0, 2), dtype=float)
    diagram = np.asarray(pairs, dtype=float)
    return cast(Array, diagram[np.argsort(diagram[:, 0], kind="stable")])


#: Vectorised ``math.erf``; NumPy has no native erf and SciPy is not a
#: dependency of the core package.
_ERF = np.frompyfunc(math.erf, 1, 1)


def _erf_cdf(z: Array) -> Array:
    """Standard normal CDF of ``z``, evaluated elementwise."""

    vals: Array = np.asarray(_ERF(z / math.sqrt(2.0)), dtype=float)
    return 0.5 * (1.0 + vals)


def persistence_image(
    x: Array,
    bins: int = 32,
    *,
    sigma: float | None = None,
    weight: Literal["persistence", "ramp", "uniform"] = "persistence",
    birth_range: tuple[float, float] | None = None,
    pers_range: tuple[float, float] | None = None,
    nan_policy: NanPolicy = "raise",
) -> Array:
    """Persistence image of the 0D sublevel-set diagram of ``x``.

    Implements the stable vectorisation of Adams et al. (2017), *Persistence
    Images: A Stable Vector Representation of Persistent Homology*, JMLR
    18(8):1-35. The diagram from :func:`persistence_diagram` is mapped to
    birth--persistence coordinates, each point is replaced by a weighted
    isotropic Gaussian, and each pixel receives the **exact integral** of that
    surface over the pixel (computed from the normal CDF, not a point sample).

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    bins:
        Number of pixels per axis ``>= 1``.
    sigma:
        Standard deviation of the Gaussian kernel. Defaults to one pixel
        width of the persistence axis.
    weight:
        Weighting function :math:`w(b, p)` applied to each diagram point.
        ``"persistence"`` uses :math:`w = p` (the standard choice, and the
        default of :mod:`persim`), ``"ramp"`` uses Adams' piecewise-linear
        ramp :math:`\\min(p / p_{\\max}, 1)`, and ``"uniform"`` uses
        :math:`w = 1`. Any weight vanishing at ``p = 0`` keeps the map stable
        with respect to the bottleneck distance.
    birth_range, pers_range:
        Explicit ``(min, max)`` extents of the image. Default to the range of
        the diagram, which makes the image **series-relative**; pass fixed
        ranges when images from different series must be comparable.
    nan_policy:
        How to treat NaNs, see :func:`_validate_series`.

    Returns
    -------
    ndarray
        ``(bins, bins)`` image indexed as ``[persistence, birth]`` so that it
        displays with persistence on the vertical axis. Empty diagrams give an
        all-zero image.

    Raises
    ------
    ValueError
        If ``bins`` is not positive, ``sigma`` is not positive, ``weight`` is
        unknown, or ``x`` is invalid.

    Examples
    --------
    >>> img = persistence_image(np.sin(np.linspace(0, 8 * np.pi, 128)), bins=8)
    >>> img.shape
    (8, 8)

    See Also
    --------
    extrema_persistence_histogram :
        The cheaper extrema-pairing histogram previously exposed under this
        name.
    """

    if bins < 1:
        raise ValueError("bins must be >= 1")
    if sigma is not None and sigma <= 0:
        raise ValueError("sigma must be positive")
    if weight not in {"persistence", "ramp", "uniform"}:
        raise ValueError("weight must be 'persistence', 'ramp' or 'uniform'")

    diagram = persistence_diagram(x, nan_policy=nan_policy)
    if diagram.shape[0] == 0:
        return np.zeros((bins, bins), dtype=float)

    births = diagram[:, 0]
    pers = diagram[:, 1] - diagram[:, 0]

    b_lo, b_hi = birth_range if birth_range is not None else (births.min(), births.max())
    p_lo, p_hi = pers_range if pers_range is not None else (0.0, pers.max())
    if b_hi <= b_lo:
        b_hi = b_lo + 1.0
    if p_hi <= p_lo:
        p_hi = p_lo + 1.0

    if sigma is None:
        sigma = (p_hi - p_lo) / bins
    sigma = float(sigma)

    if weight == "persistence":
        w = pers
    elif weight == "ramp":
        w = np.minimum(pers / (pers.max() + 1e-300), 1.0)
    else:
        w = np.ones_like(pers)

    b_edges = np.linspace(b_lo, b_hi, bins + 1)
    p_edges = np.linspace(p_lo, p_hi, bins + 1)
    # Exact integral of an isotropic Gaussian over each pixel: the product of
    # the marginal CDF differences along each axis.
    cdf_b = _erf_cdf((b_edges[None, :] - births[:, None]) / sigma)
    cdf_p = _erf_cdf((p_edges[None, :] - pers[:, None]) / sigma)
    mass_b = np.diff(cdf_b, axis=1)  # (n_points, bins)
    mass_p = np.diff(cdf_p, axis=1)
    img = np.einsum("i,ip,ib->pb", w, mass_p, mass_b)
    return cast(Array, img)


def extrema_persistence_histogram(
    x: Array, bins: int = 32, *, nan_policy: NanPolicy = "raise"
) -> Array:
    """Histogram of consecutive-extrema ``(birth, persistence)`` pairs.

    .. note::
       This is **not** a persistence image and does not compute persistent
       homology. It pairs each local extremum with the next one and histograms
       the resulting value/amplitude pairs — a cheap heuristic descriptor of
       oscillation structure. It was exposed as ``persistence_image`` before
       0.2.0; the name now refers to the real construction. Use
       :func:`persistence_image` for the topological representation.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    bins:
        Number of histogram bins per axis ``>=1``.
    nan_policy:
        How to treat NaNs, see :func:`_validate_series`.

    Returns
    -------
    ndarray
        ``(bins, bins)`` histogram of counts.

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
    hist, _, _ = np.histogram2d(b_norm, p_norm, bins=bins, range=[[0, 1], [0, 1]])
    return cast(Array, hist)


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


def _gaussian_breakpoints(alphabet: int) -> Array:
    """Equiprobable breakpoints of the standard normal for ``alphabet`` symbols."""

    probs = np.arange(1, alphabet, dtype=float) / alphabet
    # Inverse standard normal CDF via bisection on erf; alphabet is small so
    # this costs nothing and avoids a SciPy dependency.
    lo = np.full(probs.shape, -10.0)
    hi = np.full(probs.shape, 10.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = _erf_cdf(mid)
        lo = np.where(cdf < probs, mid, lo)
        hi = np.where(cdf < probs, hi, mid)
    return cast(Array, 0.5 * (lo + hi))


def sax_symbols(
    x: Array,
    segments: int = 8,
    alphabet: int = 8,
    *,
    breakpoints: Literal["gaussian", "quantile"] = "gaussian",
    nan_policy: NanPolicy = "raise",
) -> NDArray[np.int64]:
    """Symbolic Aggregate approXimation (SAX) word for ``x``.

    Follows Lin et al. (2003/2007): the series is z-normalised, reduced by
    Piecewise Aggregate Approximation to ``segments`` means, and each mean is
    mapped to a symbol using equiprobable breakpoints of the standard normal.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    segments:
        Number of PAA segments; must be in ``[1, len(x)]``.
    alphabet:
        Alphabet size ``>= 2``.
    breakpoints:
        ``"gaussian"`` (default) uses the standard SAX breakpoints on the
        z-normalised series. ``"quantile"`` instead splits at empirical
        quantiles of the segment means, which is *not* standard SAX: it is
        data-adaptive, makes symbols incomparable across series, and always
        uses the full alphabet. It is kept because it was the behaviour
        before 0.2.0.
    nan_policy:
        How to treat NaNs, see :func:`_validate_series`.

    Returns
    -------
    ndarray
        ``(segments,)`` integer symbols in ``[0, alphabet)``.

    Raises
    ------
    ValueError
        If ``segments`` or ``alphabet`` is out of range, or ``x`` is invalid.

    Examples
    --------
    >>> sax_symbols(np.arange(8.0), segments=4, alphabet=4)
    array([0, 1, 2, 3])
    """

    x = _validate_series(x, nan_policy=nan_policy)
    if segments <= 0:
        raise ValueError("segments must be >= 1")
    if segments > x.size:
        raise ValueError(
            f"segments ({segments}) cannot exceed len(x) ({x.size}); each PAA "
            "segment must contain at least one sample"
        )
    if alphabet < 2:
        raise ValueError("alphabet must be >= 2")
    means = np.array([seg.mean() for seg in np.array_split(x, segments)], dtype=float)
    if breakpoints == "gaussian":
        std = float(x.std())
        z = (means - float(x.mean())) / std if std > 0 else np.zeros_like(means)
        bps = _gaussian_breakpoints(alphabet)
    elif breakpoints == "quantile":
        z = means
        bps = np.quantile(means, np.linspace(0.0, 1.0, alphabet + 1)[1:-1])
    else:
        raise ValueError("breakpoints must be 'gaussian' or 'quantile'")
    return np.digitize(z, bps, right=False).astype(np.int64)


def sax(
    x: Array,
    segments: int = 8,
    alphabet: int = 8,
    *,
    breakpoints: Literal["gaussian", "quantile"] = "gaussian",
    nan_policy: NanPolicy = "raise",
) -> Array:
    """Symbolic Aggregate approXimation image.

    Builds the SAX word with :func:`sax_symbols` and returns the binary
    symbol-equality matrix — a recurrence plot in symbol space.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    segments:
        Number of PAA segments; must be in ``[1, len(x)]``.
    alphabet:
        Alphabet size ``>= 2``.
    breakpoints:
        Breakpoint construction, see :func:`sax_symbols`. Defaults to the
        standard Gaussian breakpoints; before 0.2.0 the (non-standard)
        ``"quantile"`` variant was the only behaviour.
    nan_policy:
        How to treat NaNs, see :func:`_validate_series`.

    Returns
    -------
    ndarray
        ``(segments, segments)`` binary matrix.

    Raises
    ------
    ValueError
        If parameters are out of range or ``x`` is invalid.
    """

    symbols = sax_symbols(
        x,
        segments,
        alphabet,
        breakpoints=breakpoints,
        nan_policy=nan_policy,
    )
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


def window_attention(x: Array, window: int = 8, *, nan_policy: NanPolicy = "raise") -> Array:
    """Scaled dot-product self-attention between sliding windows.

    Each length-``window`` subsequence acts as its own query, key and value,
    giving a row-stochastic matrix of similarities between local temporal
    patterns. It is a parameter-free encoder in the style of Vaswani et al.
    (2017), not a learned model.

    .. note::
       This is **not** Temporal Pattern Attention. TPA (Shih, Sun & Lee,
       2019, *Temporal Pattern Attention for Multivariate Time Series
       Forecasting*, Machine Learning 108:1421-1441) runs CNN filters over
       the hidden states of a recurrent network and attends over the
       resulting row vectors with a learned scoring matrix. Nothing here is
       learned and there are no hidden states, so the two are unrelated
       beyond both using attention. The function was exposed as ``tpa`` with
       an incorrect attribution before 0.2.0.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    window:
        Length of each local pattern ``>=1`` and ``<= N``.
    nan_policy:
        How to treat NaNs, see :func:`_validate_series`.

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


tpa = deprecated_alias(
    window_attention,
    "tpa",
    reason=(
        "The implementation is plain window self-attention, not the Temporal "
        "Pattern Attention architecture of Shih, Sun & Lee (2019)."
    ),
)


#: Fourier-domain analytic wavelets available to the synchrosqueezed transform.
SSTWavelet = Literal["morlet", "bump"]


def _analytic_wavelet_hat(omega: Array, wavelet: SSTWavelet, mu: float) -> Array:
    """Fourier transform of an analytic mother wavelet, evaluated at ``omega``.

    Analytic (one-sided) wavelets are required for synchrosqueezing: the phase
    derivative of the transform only estimates instantaneous frequency when the
    negative-frequency half is suppressed.
    """

    hat = np.zeros_like(omega)
    positive = omega > 0
    if wavelet == "morlet":
        # Torrence & Compo normalisation, restricted to omega > 0.
        hat[positive] = np.pi**-0.25 * math.sqrt(2.0) * np.exp(
            -0.5 * (omega[positive] - mu) ** 2
        )
    elif wavelet == "bump":
        # Compactly supported bump wavelet (Daubechies et al., 2011).
        sigma = 1.0
        scaled = (omega - mu) / sigma
        inside = positive & (np.abs(scaled) < 1.0)
        hat[inside] = np.exp(1.0 - 1.0 / (1.0 - scaled[inside] ** 2))
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown wavelet {wavelet!r}")
    return hat


def _default_sst_scales(n: int, dt: float, voices: int) -> Array:
    """Dyadic log-spaced scales covering the resolvable band of an ``n``-sample series."""

    smallest = 2.0 * dt
    largest = n * dt / 4.0
    if largest <= smallest:
        return np.array([smallest], dtype=float)
    n_octaves = math.log2(largest / smallest)
    count = int(math.ceil(n_octaves * voices)) + 1
    grid: Array = smallest * 2.0 ** (np.arange(count, dtype=float) / voices)
    return grid


def synchrosqueezed_cwt(
    x: Array,
    *,
    fs: float = 1.0,
    scales: Array | None = None,
    wavelet: SSTWavelet = "morlet",
    frequencies: int = 64,
    voices: int = 32,
    threshold: float | None = None,
    magnitude: bool = True,
    log_scale: bool = False,
    nan_policy: NanPolicy = "raise",
) -> Array:
    r"""Synchrosqueezed continuous wavelet transform.

    A time-frequency image that is sharper than either a spectrogram or a
    plain CWT, because coefficient energy is *reassigned* along the frequency
    axis to the instantaneous frequency it actually represents rather than
    left smeared across the wavelet's bandwidth.

    This is not a renamed :func:`cwt`. The three steps are:

    1. an analytic (one-sided) complex CWT
       :math:`W(a, b) = \int x(t)\,\overline{\psi\!\left(\frac{t-b}{a}\right)}\,
       \frac{\mathrm{d}t}{a}`;
    2. a phase-derivative estimate of instantaneous frequency,
       :math:`\omega(a, b) = \frac{1}{2\pi}\,
       \mathrm{Im}\!\left(\frac{\partial_b W(a, b)}{W(a, b)}\right)`,
       computed exactly in the Fourier domain rather than by finite
       differences;
    3. synchrosqueezing, which sums each coefficient into the frequency bin
       containing :math:`\omega(a, b)`,
       :math:`T(\omega_l, b) = \sum_{a_k \in \Omega_l} W(a_k, b)\,
       a_k^{-1/2}\,\Delta(\log a)`,
       the reassignment measure for a log-spaced scale grid.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    fs:
        Sampling frequency in Hz; must be positive. The output frequency axis
        is in the same units.
    scales:
        Positive wavelet scales in seconds. Defaults to a dyadic log-spaced
        grid spanning the resolvable band, which is what the reassignment
        measure above assumes; pass your own only if it is also log-spaced.
    wavelet:
        ``"morlet"`` or ``"bump"``. Both are analytic, as synchrosqueezing
        requires.
    frequencies:
        Number of bins on the output frequency axis, ``>= 2``. The axis is
        linear from ``fs / N`` to the Nyquist frequency ``fs / 2``.
    voices:
        Scales per octave in the default grid, ``>= 1``. More voices give a
        finer reassignment at proportional cost.
    threshold:
        Coefficients with ``|W| <= threshold`` contribute nothing, because
        the phase derivative is numerically meaningless where there is no
        energy. Defaults to ``1e-8 * max|W|``. Pass ``0.0`` to disable, at the
        cost of noise in silent regions.
    magnitude:
        If ``True`` (default) return ``|T|`` normalised to ``[0, 1]``. If
        ``False`` return the complex ``T``, which is invertible but cannot be
        fed to the image feature extractors.
    log_scale:
        Apply ``log1p`` to the normalised magnitude and renormalise, which
        makes weak ridges visible. Ignored when ``magnitude`` is ``False``.
    nan_policy:
        How to treat NaNs, see :func:`_validate_series`.

    Returns
    -------
    ndarray
        ``(frequencies, N)`` image, indexed ``[frequency, time]``. Real and in
        ``[0, 1]`` when ``magnitude`` is ``True``, complex otherwise.

    Raises
    ------
    ValueError
        If ``fs`` is not positive, ``frequencies < 2``, ``voices < 1``,
        ``threshold`` is negative, ``scales`` are not positive, the wavelet is
        unknown, or ``x`` is invalid.

    Notes
    -----
    **Complexity** ``O(S · N log N)`` time for ``S`` scales, ``O(S · N)``
    memory. The default grid uses ``S ≈ voices · log2(N / 8)`` scales.

    **Invariances** Equivariant to time shift (the image shifts with the
    signal) and, because the magnitude is max-normalised, invariant to
    amplitude scaling. It is *not* invariant to resampling: the frequency axis
    is tied to ``fs``.

    **Information lost** The magnitude form discards phase, so the series
    cannot be recovered; use ``magnitude=False`` to keep it. Reassignment is
    quantised to the frequency grid, and components closer together than the
    wavelet bandwidth are not separated — synchrosqueezing sharpens ridges, it
    does not increase the underlying resolution.

    **Use cases** Signals with time-varying frequency content where a
    spectrogram is too blurred to read: chirps, machine run-ups, biomedical
    rhythms.

    References
    ----------
    Daubechies, Lu & Wu (2011), "Synchrosqueezed wavelet transforms: an
    empirical mode decomposition-like tool", Applied and Computational
    Harmonic Analysis 30(2):243-261.  Thakur, Brevdo, Fučkar & Wu (2013), "The
    synchrosqueezing algorithm for time-varying spectral analysis", Signal
    Processing 93(5):1079-1094.

    Examples
    --------
    >>> fs = 200.0
    >>> t = np.arange(1024) / fs
    >>> img = synchrosqueezed_cwt(np.sin(2 * np.pi * 20.0 * t), fs=fs, frequencies=128)
    >>> img.shape
    (128, 1024)
    """

    series = _validate_series(x, nan_policy=nan_policy)
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive, finite sampling frequency")
    if frequencies < 2:
        raise ValueError("frequencies must be >= 2")
    if voices < 1:
        raise ValueError("voices must be >= 1")
    if threshold is not None and (not math.isfinite(threshold) or threshold < 0):
        raise ValueError("threshold must be non-negative and finite")
    if wavelet not in {"morlet", "bump"}:
        raise ValueError("wavelet must be 'morlet' or 'bump'")

    n = series.size
    dt = 1.0 / fs
    if scales is None:
        scale_grid = _default_sst_scales(n, dt, voices)
    else:
        scale_grid = np.asarray(scales, dtype=float)
        if scale_grid.ndim != 1 or scale_grid.size == 0 or np.any(scale_grid <= 0):
            raise ValueError("scales must be a non-empty 1D array of positive values")
    dlog = math.log(2.0) / voices if scale_grid.size > 1 else 1.0

    mu = 6.0 if wavelet == "morlet" else 5.0
    spectrum = np.fft.fft(series)
    xi = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)  # angular frequency, rad/s

    transform = np.empty((scale_grid.size, n), dtype=complex)
    derivative = np.empty((scale_grid.size, n), dtype=complex)
    for k, scale in enumerate(scale_grid):
        psi_hat = _analytic_wavelet_hat(scale * xi, wavelet, mu)
        product = spectrum * psi_hat
        transform[k] = np.fft.ifft(product)
        # d/db of the transform is exact in the Fourier domain.
        derivative[k] = np.fft.ifft(1j * xi * product)

    peak = float(np.max(np.abs(transform)))
    cutoff = 1e-8 * peak if threshold is None else float(threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        inst_freq = np.imag(derivative / transform) / (2.0 * np.pi)
    usable = (np.abs(transform) > cutoff) & np.isfinite(inst_freq)

    f_min = fs / n
    f_max = fs / 2.0
    f_grid = np.linspace(f_min, f_max, frequencies)
    df = f_grid[1] - f_grid[0]

    squeezed = np.zeros((frequencies, n), dtype=complex)
    with np.errstate(invalid="ignore"):
        bins = np.round((inst_freq - f_min) / df)
    usable &= np.isfinite(bins) & (bins >= 0) & (bins <= frequencies - 1)
    if np.any(usable):
        weights = scale_grid**-0.5 * dlog
        rows, cols = np.nonzero(usable)
        np.add.at(
            squeezed,
            (bins[usable].astype(np.intp), cols),
            transform[usable] * weights[rows],
        )

    if not magnitude:
        return cast(Array, squeezed)
    image = np.abs(squeezed)
    image = image / (np.max(image) + 1e-12)
    if log_scale:
        image = np.log1p(image)
        image = image / (np.max(image) + 1e-12)
    return cast(Array, image)


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
    for i in range(n - 1):
        # Slopes from point i to every subsequent point
        slopes = (x[i + 1 :] - x[i]) / np.arange(1, n - i, dtype=float)
        # Adjacent point is always visible
        adj[i, i + 1] = adj[i + 1, i] = 1.0
        if slopes.size > 1:
            # Point j is visible iff slope(i,j) > max slope to any k in (i,j)
            cummax = np.maximum.accumulate(slopes)
            visible = slopes[1:] > cummax[:-1]
            idxs = np.nonzero(visible)[0] + (i + 2)
            adj[i, idxs] = 1.0
            adj[idxs, i] = 1.0
    return cast(Array, adj)


ChirpletAggregate = Literal["max", "mean", "energy", "none"]

#: Refuse to allocate a chirplet tensor larger than this, in bytes. The tensor
#: is ``chirp_rates x frequencies x frames`` and grows quickly.
MAX_CHIRPLET_BYTES = 512 * 1024**2


def chirplet_transform(
    x: Array,
    *,
    fs: float = 1.0,
    frequencies: Array | None = None,
    chirp_rates: Array | None = None,
    window_size: int = 64,
    hop_length: int | None = None,
    aggregate: ChirpletAggregate = "max",
    log_scale: bool = True,
    max_bytes: int = MAX_CHIRPLET_BYTES,
    nan_policy: NanPolicy = "raise",
) -> Array:
    r"""Chirplet transform: a time-frequency image that also resolves chirp rate.

    Correlates the signal against a dictionary of Gaussian-windowed *chirped*
    atoms

    .. math::

        g_{t_0, f, c}(t) = w(t - t_0)\,
        \exp\!\bigl(2\pi i \bigl[f (t - t_0) + \tfrac{c}{2}(t - t_0)^2\bigr]\bigr)

    each carrying a temporal centre :math:`t_0`, a centre frequency :math:`f`,
    a chirp rate :math:`c` in Hz/s, and complex phase. A short-time Fourier
    transform is the special case :math:`c = 0`; sweeping :math:`c` is what
    makes this a chirplet transform rather than an STFT with a different
    window.

    Computed by de-chirping each frame with :math:`\exp(-i\pi c\tau^2)` and
    taking one FFT per chirp rate, which evaluates every frequency at once.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    fs:
        Sampling frequency in Hz; positive. Chirp rates are in Hz/s.
    frequencies:
        Centre frequencies to evaluate. ``None`` (default) uses the FFT grid
        ``0 .. fs/2``, which is much faster; an explicit array is evaluated
        with a direct DFT and may be any set of frequencies.
    chirp_rates:
        Chirp rates in Hz/s. ``None`` uses 9 rates linearly spaced over
        ``+/- fs^2 / (4 * window_size)``, the largest sweep a single window can
        resolve without the instantaneous frequency leaving the band.
    window_size:
        Samples per frame, ``>= 4`` and ``<= N``.
    hop_length:
        Step between frames; defaults to ``window_size // 4``.
    aggregate:
        How to reduce the chirp-rate axis. ``"max"`` (default) keeps the
        best-matching rate per time-frequency cell, the conventional chirplet
        display. ``"mean"`` averages, ``"energy"`` takes the root sum of
        squares, and ``"none"`` returns the full three-dimensional tensor.
    log_scale:
        Apply ``log1p`` after normalising, which makes weak atoms visible.
    max_bytes:
        Refuse to allocate a tensor larger than this. The intermediate is
        ``len(chirp_rates) x len(frequencies) x n_frames`` float64.
    nan_policy:
        How to treat NaNs, see :func:`_validate_series`.

    Returns
    -------
    ndarray
        ``(n_frequencies, n_frames)`` image normalised to ``[0, 1]``, or
        ``(n_chirp_rates, n_frequencies, n_frames)`` when ``aggregate="none"``.

    Raises
    ------
    ValueError
        If any parameter is out of range, the requested tensor would exceed
        ``max_bytes``, or ``x`` is invalid.

    Notes
    -----
    **Complexity** ``O(C · F · W log W)`` for ``C`` chirp rates and ``F``
    frames on the FFT grid, or ``O(C · F · W · n_frequencies)`` with explicit
    frequencies. Memory is the tensor above, checked before allocation.

    **Invariances** Equivariant to time shift by whole hops. Invariant to
    amplitude scaling through the normalisation. Reversing the signal in time
    negates the recovered chirp rates.

    **Information lost** Phase, and any chirp rate outside the requested grid —
    a sweep faster than the grid covers is attributed to the nearest rate, so
    the grid is a modelling choice rather than a display setting. Aggregation
    additionally discards *which* rate matched; use ``aggregate="none"`` to
    keep it.

    **Use cases** Signals whose frequency sweeps within a single analysis
    window — radar and sonar chirps, machine run-ups, birdsong, gravitational
    wave templates — where an STFT smears the sweep across bins.

    References
    ----------
    Mann & Haykin (1995), "The chirplet transform: physical considerations",
    IEEE Transactions on Signal Processing 43(11):2745-2761.  Baraniuk & Jones
    (1996), "Wigner-based formulation of the chirplet transform", IEEE
    Transactions on Signal Processing 44(12):3129-3135.

    Examples
    --------
    >>> fs = 200.0
    >>> t = np.arange(1024) / fs
    >>> img = chirplet_transform(np.sin(2 * np.pi * (30 * t + 15 * t**2)), fs=fs)
    >>> img.shape
    (33, 61)
    """

    series = _validate_series(x, nan_policy=nan_policy)
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive, finite sampling frequency")
    if window_size < 4:
        raise ValueError("window_size must be >= 4")
    if window_size > series.size:
        raise ValueError(
            f"window_size={window_size} exceeds the series length {series.size}"
        )
    step = window_size // 4 if hop_length is None else int(hop_length)
    if step < 1:
        raise ValueError("hop_length must be >= 1")
    if aggregate not in {"max", "mean", "energy", "none"}:
        raise ValueError("aggregate must be 'max', 'mean', 'energy' or 'none'")

    if chirp_rates is None:
        limit = fs**2 / (4.0 * window_size)
        rates = np.linspace(-limit, limit, 9)
    else:
        rates = np.asarray(chirp_rates, dtype=float)
        if rates.ndim != 1 or rates.size == 0 or not np.all(np.isfinite(rates)):
            raise ValueError("chirp_rates must be a non-empty 1D array of finite values")

    use_fft_grid = frequencies is None
    if use_fft_grid:
        n_frequencies = window_size // 2 + 1
    else:
        freq_grid = np.asarray(frequencies, dtype=float)
        if freq_grid.ndim != 1 or freq_grid.size == 0 or not np.all(np.isfinite(freq_grid)):
            raise ValueError("frequencies must be a non-empty 1D array of finite values")
        n_frequencies = freq_grid.size

    n_frames = 1 + (series.size - window_size) // step
    needed = rates.size * n_frequencies * n_frames * 8
    if needed > max_bytes:
        raise ValueError(
            f"the chirplet tensor would need {needed / 1024**2:.1f} MiB "
            f"({rates.size} chirp rates x {n_frequencies} frequencies x "
            f"{n_frames} frames), above max_bytes="
            f"{max_bytes / 1024**2:.1f} MiB; reduce the grids, raise the hop, "
            "or raise max_bytes deliberately"
        )

    frames = np.lib.stride_tricks.as_strided(
        series,
        shape=(n_frames, window_size),
        strides=(series.strides[0] * step, series.strides[0]),
        writeable=False,
    )
    window = np.hanning(window_size)
    # Time relative to the centre of the atom, so the chirp is symmetric.
    tau = (np.arange(window_size) - (window_size - 1) / 2.0) / fs

    tensor = np.empty((rates.size, n_frequencies, n_frames), dtype=float)
    for index, rate in enumerate(rates):
        dechirped = frames * window * np.exp(-1j * np.pi * rate * tau**2)
        if use_fft_grid:
            # The de-chirped frame is complex, so its spectrum is not
            # conjugate-symmetric; take the full FFT and keep 0 .. fs/2.
            spectrum = np.fft.fft(dechirped, axis=1)[:, :n_frequencies]
        else:
            basis = np.exp(-2j * np.pi * np.outer(freq_grid, tau))
            spectrum = dechirped @ basis.T
        tensor[index] = np.abs(spectrum).T

    if aggregate == "none":
        return _normalise_image(tensor, log_scale)
    if aggregate == "max":
        reduced = tensor.max(axis=0)
    elif aggregate == "mean":
        reduced = tensor.mean(axis=0)
    else:
        reduced = np.sqrt(np.sum(tensor**2, axis=0))
    return _normalise_image(reduced, log_scale)


def _normalise_image(values: Array, log_scale: bool) -> Array:
    """Scale to ``[0, 1]``, optionally compressing the dynamic range first."""

    peak = float(values.max())
    if peak <= 0:
        empty: Array = np.zeros_like(values)
        return empty
    scaled: Array = values / peak
    if log_scale:
        scaled = np.log1p(scaled)
        scaled = scaled / (scaled.max() + 1e-300)
    return scaled


SpectralScaling = Literal["power", "log_power"]


def multitaper_spectrogram(
    x: Array,
    *,
    fs: float = 1.0,
    window_size: int = 128,
    hop_length: int | None = None,
    time_bandwidth: float = 3.5,
    n_tapers: int | None = None,
    n_fft: int | None = None,
    scaling: SpectralScaling = "log_power",
    dynamic_range: float = 80.0,
    nan_policy: NanPolicy = "raise",
) -> Array:
    r"""Multitaper spectrogram using DPSS (Slepian) tapers.

    A lower-variance alternative to :func:`spectrogram`. A single tapered
    periodogram is an inconsistent estimator: its variance does not shrink as
    the window grows. Thomson's method averages :math:`K` periodograms taken
    with orthogonal Slepian tapers, which are the sequences with maximal
    energy concentration in a bandwidth :math:`NW`, reducing variance by
    roughly :math:`1/K` while widening the effective resolution to
    :math:`2NW/W`.

    .. math::

        \hat{S}(f) = \frac{1}{K} \sum_{k=0}^{K-1}
        \left| \sum_{t} h_k[t]\, x[t]\, e^{-2\pi i f t} \right|^2

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    fs:
        Sampling frequency in Hz; positive. Used only to document the
        frequency axis, which spans ``0`` to ``fs / 2``.
    window_size:
        Samples per analysis window, ``>= 2`` and ``<= N``.
    hop_length:
        Step between windows; defaults to ``window_size // 4``.
    time_bandwidth:
        The product :math:`NW`, ``>= 1``. Larger values allow more tapers and
        so lower variance, at the cost of a wider main lobe.
    n_tapers:
        Number of tapers :math:`K`. Defaults to ``int(2 * NW) - 1``, the
        conventional choice — beyond it the tapers leak badly and adding them
        increases bias faster than it reduces variance. Must be in
        ``[1, window_size]``.
    n_fft:
        FFT length, defaults to ``window_size``. Larger values interpolate the
        frequency axis; they do not add resolution.
    scaling:
        ``"power"`` returns the averaged power spectrum. ``"log_power"``
        (default) returns decibels relative to the peak, floored at
        ``-dynamic_range``.
    dynamic_range:
        Decibels below the peak to retain before flooring, ``> 0``. Ignored
        for ``"power"``.
    nan_policy:
        How to treat NaNs, see :func:`_validate_series`.

    Returns
    -------
    ndarray
        ``(n_fft // 2 + 1, n_frames)`` image normalised to ``[0, 1]``, with
        frequency increasing down the rows.

    Raises
    ------
    ImportError
        If SciPy is not installed. DPSS tapers are the defining ingredient;
        approximating them with something easier would make this a different,
        worse estimator under the same name.
    ValueError
        If any argument is out of range, or ``x`` is invalid.

    Notes
    -----
    **Complexity** ``O(K · F · n_fft log n_fft)`` for ``F`` frames, plus
    ``O(K · window_size)`` for the tapers, which are computed once.

    **Invariances** Equivariant to time shift by whole hops. Invariant to
    amplitude scaling, because the output is normalised by its own peak.

    **Information lost** Phase, entirely. The dB form additionally discards
    everything more than ``dynamic_range`` below the peak, and the taper
    averaging deliberately trades frequency resolution for variance — two
    tones closer than ``2 NW / window_size`` in normalised frequency will not
    be separated.

    **Use cases** Spectral estimation on noisy or short records, where a
    single-taper spectrogram is too erratic to compare across windows.

    References
    ----------
    Thomson (1982), "Spectrum estimation and harmonic analysis", Proceedings
    of the IEEE 70(9):1055-1096.  Slepian (1978), "Prolate spheroidal wave
    functions, Fourier analysis, and uncertainty V", Bell System Technical
    Journal 57(5):1371-1430.

    Examples
    --------
    >>> t = np.arange(512) / 100.0
    >>> img = multitaper_spectrogram(np.sin(2 * np.pi * 10 * t), fs=100.0,
    ...                              window_size=128)
    >>> img.shape
    (65, 13)
    """

    series = _validate_series(x, nan_policy=nan_policy)
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive, finite sampling frequency")
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if window_size > series.size:
        raise ValueError(
            f"window_size={window_size} exceeds the series length {series.size}"
        )
    step = window_size // 4 if hop_length is None else int(hop_length)
    if step < 1:
        raise ValueError("hop_length must be >= 1")
    if not math.isfinite(time_bandwidth) or time_bandwidth < 1:
        raise ValueError("time_bandwidth (NW) must be finite and >= 1")
    tapers_wanted = int(2 * time_bandwidth) - 1 if n_tapers is None else int(n_tapers)
    if tapers_wanted < 1:
        raise ValueError(
            f"n_tapers must be >= 1; the default int(2 * NW) - 1 gives "
            f"{tapers_wanted} at time_bandwidth={time_bandwidth}, so raise NW "
            "or set n_tapers explicitly"
        )
    if tapers_wanted > window_size:
        raise ValueError("n_tapers cannot exceed window_size")
    length = window_size if n_fft is None else int(n_fft)
    if length < window_size:
        raise ValueError("n_fft must be at least window_size")
    if scaling not in {"power", "log_power"}:
        raise ValueError("scaling must be 'power' or 'log_power'")
    if scaling == "log_power" and (not math.isfinite(dynamic_range) or dynamic_range <= 0):
        raise ValueError("dynamic_range must be positive and finite")

    try:
        from scipy.signal.windows import dpss
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "SciPy is required for multitaper_spectrogram: the DPSS tapers are "
            "the method, and substituting an easier window would silently make "
            "this a different estimator. Install with "
            "`pip install 'tscv-vision[spectral]'`"
        ) from exc

    tapers = np.asarray(dpss(window_size, time_bandwidth, Kmax=tapers_wanted), dtype=float)

    n_frames = 1 + (series.size - window_size) // step
    frames = np.lib.stride_tricks.as_strided(
        series,
        shape=(n_frames, window_size),
        strides=(series.strides[0] * step, series.strides[0]),
        writeable=False,
    )

    # (K, frames, window) -> average |FFT|^2 over the taper axis.
    tapered = frames[None, :, :] * tapers[:, None, :]
    spectra = np.fft.rfft(tapered, n=length, axis=2)
    power = np.mean(np.abs(spectra) ** 2, axis=0).T  # (freq, frames)

    peak = float(power.max())
    if peak <= 0:
        return cast(Array, np.zeros_like(power))
    if scaling == "power":
        return cast(Array, power / peak)
    decibels = 10.0 * np.log10(power / peak + 1e-300)
    clipped = np.clip(decibels, -dynamic_range, 0.0)
    return cast(Array, (clipped + dynamic_range) / dynamic_range)


DensityMode = Literal["histogram", "gaussian"]


def delay_embedding_density(
    x: Array,
    *,
    delay: int = 1,
    dimension: int = 2,
    bins: int = 64,
    projection: tuple[int, int] = (0, 1),
    density: DensityMode = "histogram",
    sigma: float = 1.0,
    normalize: bool = True,
    nan_policy: NanPolicy = "raise",
) -> Array:
    r"""Delay-embedding occupancy density — a TSCV-Vision representation.

    .. note::
       This is a **TSCV-Vision representation**. It is *not* a recurrence
       plot: a recurrence plot is indexed by pairs of times and records which
       states are close to which, whereas this image is indexed by state-space
       coordinates and records how often the trajectory visits each region.
       The delay embedding itself is Takens' (1981); rendering its occupancy
       as an image is the part that is ours.

    Builds the delay-coordinate embedding

    .. math::

        X_t = \bigl(x_t,\; x_{t+\tau},\; \dots,\; x_{t+(m-1)\tau}\bigr)

    and histograms a chosen two-dimensional projection of the reconstructed
    state space.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    delay:
        Embedding delay :math:`\tau >= 1`. No automatic selection is provided;
        choosing it from the data is a separate decision that deserves its own
        documented method rather than a silent default.
    dimension:
        Embedding dimension :math:`m >= 2`.
    bins:
        Grid resolution per axis, ``>= 2``.
    projection:
        Which two embedding coordinates to plot, as ``(horizontal, vertical)``
        indices into ``0 .. dimension - 1``. They must differ.
    density:
        ``"histogram"`` counts occupancy per bin. ``"gaussian"`` additionally
        smooths those counts with a separable Gaussian, which turns the
        discrete counts into a kernel-density-style image.
    sigma:
        Gaussian width in bins, ``> 0``. Ignored for ``"histogram"``.
    normalize:
        Divide by the maximum so the image lies in ``[0, 1]``. Set ``False``
        to keep raw occupancy counts, whose total is the number of embedded
        points.
    nan_policy:
        How to treat NaNs, see :func:`_validate_series`.

    Returns
    -------
    ndarray
        ``(bins, bins)`` image indexed ``[vertical, horizontal]``, so that it
        displays with the second projected coordinate on the y-axis.

    Raises
    ------
    ValueError
        If ``delay < 1``, ``dimension < 2``, ``bins < 2``, the projection
        indices are equal or out of range, ``sigma <= 0``, the series is too
        short for even one embedded point, or ``x`` is invalid.

    Notes
    -----
    **Complexity** ``O(N)`` to embed and bin, plus ``O(bins^2)`` memory. The
    Gaussian mode adds ``O(bins^2 · k)`` for a separable kernel of radius
    ``k = ceil(3 sigma)``.

    **Invariances** Invariant to time shift and to any permutation of the
    trajectory's visits — only *where* the trajectory goes matters, not when.
    Equivariant to affine rescaling of the values in the sense that the image
    is unchanged, because the grid spans the observed range.

    **Information lost** Chronological order, completely. Two series visiting
    the same state-space cells in any order give the same image, so the
    direction of travel around an orbit, and therefore time reversal, is
    invisible. Occupancy within a bin is also lost — the image records how
    often, not where inside.

    **Use cases** Distinguishing periodic, quasi-periodic and chaotic
    dynamics, where a limit cycle appears as a thin closed curve and noise as
    diffuse cloud.

    References
    ----------
    Takens (1981), "Detecting strange attractors in turbulence", in Dynamical
    Systems and Turbulence, Lecture Notes in Mathematics 898:366-381.

    Examples
    --------
    >>> image = delay_embedding_density(np.sin(np.linspace(0, 40.0, 500)), delay=8)
    >>> image.shape
    (64, 64)
    """

    series = _validate_series(x, nan_policy=nan_policy)
    if delay < 1:
        raise ValueError("delay must be >= 1")
    if dimension < 2:
        raise ValueError("dimension must be >= 2")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    if len(projection) != 2:
        raise ValueError("projection must be a pair of coordinate indices")
    horizontal, vertical = int(projection[0]), int(projection[1])
    if horizontal == vertical:
        raise ValueError("projection indices must differ")
    for index in (horizontal, vertical):
        if index < 0 or index >= dimension:
            raise ValueError(
                f"projection index {index} is outside [0, {dimension - 1}]"
            )
    if density not in {"histogram", "gaussian"}:
        raise ValueError("density must be 'histogram' or 'gaussian'")
    if density == "gaussian" and (not math.isfinite(sigma) or sigma <= 0):
        raise ValueError("sigma must be positive and finite")

    span = (dimension - 1) * delay
    n_points = series.size - span
    if n_points < 1:
        raise ValueError(
            f"series of length {series.size} is too short to embed at "
            f"dimension={dimension}, delay={delay}; it needs at least "
            f"{span + 1} samples"
        )

    first = series[horizontal * delay : horizontal * delay + n_points]
    second = series[vertical * delay : vertical * delay + n_points]

    lo = float(series.min())
    hi = float(series.max())
    if hi == lo:
        # A constant series occupies a single point; place it in the centre
        # rather than dividing by a zero range.
        image = np.zeros((bins, bins), dtype=float)
        image[bins // 2, bins // 2] = float(n_points)
        return cast(Array, image / (image.max() + 1e-12) if normalize else image)

    counts, _, _ = np.histogram2d(
        second, first, bins=bins, range=[[lo, hi], [lo, hi]]
    )

    if density == "gaussian":
        radius = int(math.ceil(3.0 * sigma))
        offsets = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(counts, radius, mode="constant")
        # Separable: convolve rows then columns.
        smoothed = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="valid"), 1, padded
        )
        counts = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="valid"), 0, smoothed
        )

    if normalize:
        counts = counts / (counts.max() + 1e-12)
    density_image: Array = counts
    return density_image


#: Largest embedding order accepted by :func:`ordinal_transition_field`.
#: The state space is ``order!``, so the dense transition matrix grows as
#: ``(order!)^2``: order 7 already needs ~203 MB. Bandt & Pompe recommend
#: ``3 <= order <= 7`` on statistical grounds too — larger orders leave too few
#: windows to estimate transitions from.
MAX_ORDINAL_ORDER = 7

TiePolicy = Literal["stable", "raise", "jitter"]
OrdinalMode = Literal["transition_matrix", "field"]


def _ordinal_patterns(
    x: Array, order: int, delay: int, tie_policy: TiePolicy, seed: int
) -> NDArray[np.int64]:
    """Return the Lehmer code of each embedded window's ordinal pattern.

    The Lehmer code is a bijection from permutations of ``order`` elements onto
    ``0 .. order! - 1``, so patterns are labelled by exact integers rather than
    by hashing floats.
    """

    span = (order - 1) * delay
    n_windows = x.size - span
    embedded = np.empty((n_windows, order), dtype=float)
    for k in range(order):
        start = k * delay
        embedded[:, k] = x[start : start + n_windows]

    if tie_policy == "raise":
        if np.any(np.diff(np.sort(embedded, axis=1), axis=1) == 0):
            raise ValueError(
                "tied values inside an embedded window make the ordinal pattern "
                "ambiguous; use tie_policy='stable' or 'jitter'"
            )
    elif tie_policy == "jitter":
        # Deterministic given `seed`: ties are broken at random rather than
        # systematically in favour of earlier positions, which biases the
        # pattern distribution on quantised or repetitive data.
        rng = np.random.default_rng(seed)
        scale = float(np.max(np.abs(embedded))) or 1.0
        embedded = embedded + rng.uniform(-1e-9, 1e-9, size=embedded.shape) * scale

    # A stable argsort breaks remaining ties by position, which is the
    # conventional Bandt-Pompe choice.
    permutation = np.argsort(embedded, axis=1, kind="stable")

    factorials = np.array(
        [math.factorial(order - 1 - i) for i in range(order)], dtype=np.int64
    )
    codes = np.zeros(n_windows, dtype=np.int64)
    for i in range(order):
        later = permutation[:, i + 1 :]
        smaller = np.sum(later < permutation[:, i : i + 1], axis=1)
        codes += smaller.astype(np.int64) * factorials[i]
    return codes


def _block_mean(image: Array, size: int) -> Array:
    """Deterministically downsample a square image to ``size`` by block means."""

    n = image.shape[0]
    edges = np.linspace(0, n, size + 1).round().astype(int)
    out = np.empty((size, size), dtype=float)
    for i in range(size):
        rows = image[edges[i] : max(edges[i] + 1, edges[i + 1])]
        for j in range(size):
            out[i, j] = rows[:, edges[j] : max(edges[j] + 1, edges[j + 1])].mean()
    return out


def ordinal_transition_field(
    x: Array,
    *,
    order: int = 3,
    delay: int = 1,
    image_size: int | None = None,
    tie_policy: TiePolicy = "stable",
    mode: OrdinalMode = "field",
    seed: int = 0,
    nan_policy: NanPolicy = "raise",
) -> Array:
    r"""Ordinal Pattern Transition Field — a TSCV-Vision representation.

    .. note::
       This is a **TSCV-Vision representation**, not a previously published
       image algorithm. It composes two established ingredients — Bandt &
       Pompe ordinal patterns (2002) and ordinal-pattern transition networks
       (Small, 2013; McCullough et al., 2015) — into a field laid out like the
       Markov Transition Field of Wang & Oates (2015). The composition is ours;
       no source describes this exact construction.

    Three stages:

    1. **Ordinal encoding.** Each embedded window
       :math:`(x_t, x_{t+\tau}, \dots, x_{t+(m-1)\tau})` is replaced by the
       permutation that sorts it, labelled by its Lehmer code in
       :math:`0 \dots m!-1`.
    2. **Transition matrix.**
       :math:`P[a,b] = \frac{\#\{t : s_t = a,\, s_{t+1} = b\}}{\#\{t : s_t = a\}}`,
       so every observed row sums to 1. Rows for states that never occur are
       all zero — the conditional distribution is undefined there, and a
       uniform row would invent structure.
    3. **Field.** :math:`F[i,j] = P[s_i, s_j]`, one pixel per pair of time
       steps, in the manner of the MTF.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    order:
        Embedding dimension :math:`m`, in ``[2, 7]``; see
        :data:`MAX_ORDINAL_ORDER`.
    delay:
        Embedding delay :math:`\tau >= 1`.
    image_size:
        Downsample the field to ``(image_size, image_size)`` by block means.
        ``None`` keeps one pixel per window pair. Ignored when ``mode`` is
        ``"transition_matrix"``.
    tie_policy:
        What to do when a window contains equal values, which leaves the
        pattern ambiguous. ``"stable"`` (default) breaks ties by position, the
        conventional choice. ``"raise"`` refuses. ``"jitter"`` breaks them at
        random, deterministically given ``seed``, which avoids the systematic
        bias toward earlier positions that ``"stable"`` introduces on
        quantised data.
    mode:
        ``"field"`` returns the time-indexed field, ``"transition_matrix"``
        the raw ``(m!, m!)`` matrix.
    seed:
        Seed for ``tie_policy="jitter"``; ignored otherwise. Present so that
        jittered output is still reproducible.
    nan_policy:
        How to treat NaNs, see :func:`_validate_series`.

    Returns
    -------
    ndarray
        ``(W, W)`` field for ``W = N - (order - 1) * delay`` windows, or
        ``(image_size, image_size)`` if downsampled, or ``(m!, m!)`` in
        ``"transition_matrix"`` mode. All values are probabilities in
        ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``order`` is outside ``[2, 7]``, ``delay < 1``, ``image_size`` is
        not positive or exceeds the field, the series is too short for two
        windows, ``tie_policy="raise"`` and ties exist, or ``x`` is invalid.

    Notes
    -----
    **Complexity** ``O(N · m log m)`` to encode, ``O(N)`` to accumulate
    transitions, ``O(W^2)`` memory for the field. The *state space* is
    factorial in ``order``, so the dense transition matrix costs
    ``(m!)^2``: 36 entries at ``m=3``, 518 400 at ``m=6``, 25.4 million
    (~203 MB) at ``m=7``.

    **Invariances** Invariant under any strictly increasing transformation of
    the values, because only their order is used — the property tested
    explicitly below. Not invariant to time reversal or to reordering.

    **Information lost** All amplitude information: only the ranking within
    each window survives. Series with identical ordinal dynamics and wildly
    different magnitudes give the same field. Windows shorter than ``order``
    at the tail are dropped.

    **Use cases** Regime and dynamics discrimination where amplitude is
    unreliable or uncalibrated — the ordinal view is robust to monotonic
    sensor drift and to unknown gain.

    References
    ----------
    Bandt & Pompe (2002), "Permutation entropy: a natural complexity measure
    for time series", Physical Review Letters 88:174102.  Small (2013),
    "Complex networks from time series: capturing dynamics", ISCAS.
    McCullough, Small, Stemler & Iu (2015), "Time lagged ordinal partition
    networks for capturing dynamics of continuous dynamical systems", Chaos
    25:053101.

    Examples
    --------
    >>> field = ordinal_transition_field(np.sin(np.linspace(0, 12.0, 64)))
    >>> field.shape
    (62, 62)
    >>> bool(np.all((field >= 0) & (field <= 1)))
    True
    """

    series = _validate_series(x, nan_policy=nan_policy)
    if order < 2 or order > MAX_ORDINAL_ORDER:
        raise ValueError(
            f"order must be in [2, {MAX_ORDINAL_ORDER}]; the state space is "
            f"order! and the dense transition matrix (order!)^2, which is "
            f"already ~203 MB at order {MAX_ORDINAL_ORDER}"
        )
    if delay < 1:
        raise ValueError("delay must be >= 1")
    if tie_policy not in {"stable", "raise", "jitter"}:
        raise ValueError("tie_policy must be 'stable', 'raise' or 'jitter'")
    if mode not in {"transition_matrix", "field"}:
        raise ValueError("mode must be 'transition_matrix' or 'field'")

    span = (order - 1) * delay
    n_windows = series.size - span
    if n_windows < 2:
        raise ValueError(
            f"series of length {series.size} yields {max(n_windows, 0)} windows "
            f"at order={order}, delay={delay}; at least 2 are needed to observe "
            "a transition"
        )

    codes = _ordinal_patterns(series, order, delay, tie_policy, seed)
    n_states = math.factorial(order)

    # Accumulate transitions over the observed states only, so memory follows
    # the data rather than the factorial state space.
    observed, inverse = np.unique(codes, return_inverse=True)
    counts = np.zeros((observed.size, observed.size), dtype=float)
    np.add.at(counts, (inverse[:-1], inverse[1:]), 1.0)
    row_totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probabilities = np.where(row_totals > 0, counts / row_totals, 0.0)

    if mode == "transition_matrix":
        full = np.zeros((n_states, n_states), dtype=float)
        full[np.ix_(observed, observed)] = probabilities
        return cast(Array, full)

    field = probabilities[np.ix_(inverse, inverse)]
    if image_size is not None:
        if image_size < 1:
            raise ValueError("image_size must be positive")
        if image_size > n_windows:
            raise ValueError(
                f"image_size={image_size} exceeds the {n_windows}x{n_windows} "
                "field; upsampling would invent detail"
            )
        field = _block_mean(field, image_size)
    return cast(Array, field)


HVGWeight = Literal["binary", "amplitude", "distance"]


def _hvg_edges(x: Array) -> list[tuple[int, int]]:
    """Return the horizontal-visibility edges of ``x`` in ``O(N)`` time.

    A monotonic stack holds the indices that are still visible from the right.
    Each index is pushed once and popped once, so the total work is linear
    rather than the quadratic scan the definition suggests.
    """

    edges: list[tuple[int, int]] = []
    stack: list[int] = []
    for i in range(x.size):
        while stack and x[stack[-1]] < x[i]:
            edges.append((stack.pop(), i))
        if stack:
            edges.append((stack[-1], i))
            if x[stack[-1]] == x[i]:
                # Equal heights block everything behind them, so the older
                # index can never be seen again.
                stack.pop()
        stack.append(i)
    return edges


def horizontal_visibility_graph(
    x: Array,
    *,
    weighted: bool = False,
    weight: HVGWeight = "binary",
    nan_policy: NanPolicy = "raise",
) -> Array:
    r"""Horizontal visibility graph adjacency matrix.

    Two observations are *horizontally* visible when a horizontal line can be
    drawn between the tops of their bars without crossing an intermediate bar:
    for :math:`i < j`,

    .. math::

        x_k < \min(x_i, x_j) \quad \text{for all } i < k < j.

    This is a different graph from :func:`visibility_graph`, which uses the
    *natural* (line-of-sight) criterion. Every horizontal edge is also a
    natural edge, but not the reverse, so the HVG is a subgraph of the NVG.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    weighted:
        Must be ``True`` to use a non-binary ``weight``. Passing a weight
        without setting this raises, so a mis-specified call cannot silently
        produce a binary graph.
    weight:
        ``"binary"`` (the canonical definition) or one of two TSCV-Vision
        extensions, which are **not** part of the published HVG:
        ``"amplitude"`` weights an edge by ``|x_i - x_j|`` normalised by its
        maximum, and ``"distance"`` by ``1 / (1 + |i - j|)``.
    nan_policy:
        How to treat NaNs, see :func:`_validate_series`.

    Returns
    -------
    ndarray
        ``(N, N)`` symmetric adjacency matrix with a zero diagonal. Binary
        ``{0, 1}`` unless a weighting is requested.

    Raises
    ------
    ValueError
        If ``weight`` is unknown, a non-binary weight is requested without
        ``weighted=True``, or ``x`` is invalid.

    Notes
    -----
    **Complexity** ``O(N)`` time to find the edges via a monotonic stack, then
    ``O(N^2)`` memory to materialise the dense matrix — the matrix, not the
    algorithm, is the quadratic part.

    **Invariances** Invariant under any strictly increasing transformation of
    the values, since the criterion involves only their order. This is the
    sharpest practical difference from the natural visibility graph, whose
    slope-based criterion is not order-invariant. Also invariant to time
    reversal.

    **Information lost** Everything except the ordinal structure: amplitudes,
    sampling interval and absolute time all disappear. Two series with the same
    ranking of values give the same binary graph.

    **Use cases** Distinguishing stochastic from chaotic dynamics, where the
    HVG degree distribution has known analytic forms — uncorrelated noise gives
    :math:`P(k) = \frac{1}{3}\left(\frac{2}{3}\right)^{k-2}`.

    References
    ----------
    Luque, Lacasa, Ballesteros & Luque (2009), "Horizontal visibility graphs:
    exact results for random time series", Physical Review E 80:046103.

    Examples
    --------
    >>> horizontal_visibility_graph(np.array([1.0, 3.0, 2.0, 4.0]))
    array([[0., 1., 0., 0.],
           [1., 0., 1., 1.],
           [0., 1., 0., 1.],
           [0., 1., 1., 0.]])
    """

    series = _validate_series(x, nan_policy=nan_policy)
    if weight not in {"binary", "amplitude", "distance"}:
        raise ValueError("weight must be 'binary', 'amplitude' or 'distance'")
    if weight != "binary" and not weighted:
        raise ValueError(
            f"weight={weight!r} requires weighted=True; refusing to silently "
            "return a binary graph"
        )

    n = series.size
    adjacency = np.zeros((n, n), dtype=float)
    edges = _hvg_edges(series)
    if not edges:
        return cast(Array, adjacency)

    rows = np.fromiter((i for i, _ in edges), dtype=np.intp, count=len(edges))
    cols = np.fromiter((j for _, j in edges), dtype=np.intp, count=len(edges))

    if not weighted or weight == "binary":
        values = np.ones(len(edges), dtype=float)
    elif weight == "amplitude":
        values = np.abs(series[rows] - series[cols])
        values = values / (values.max() + 1e-12)
    else:  # distance
        values = 1.0 / (1.0 + np.abs(rows - cols).astype(float))

    adjacency[rows, cols] = values
    adjacency[cols, rows] = values
    return cast(Array, adjacency)


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


def matrix_profile(
    x: Array,
    m: int,
    *,
    exclusion: int | None = None,
    normalize: bool = True,
    nan_policy: NanPolicy = "raise",
) -> Array:
    """Naive z-normalised matrix profile for motif/discord discovery.

    For every length-``m`` subsequence the profile stores the Euclidean
    distance to its closest non-trivial match, where matches within the
    exclusion zone around the diagonal are ignored.

    Parameters
    ----------
    x:
        Input 1D series ``(N,)``.
    m:
        Subsequence length ``>= 2``.
    exclusion:
        Half-width of the trivial-match exclusion zone. Defaults to ``m // 2``,
        the usual convention.
    normalize:
        If ``True`` (default) the profile is divided by its maximum so that it
        lies in ``[0, 1]``; set to ``False`` to keep raw distances.
    nan_policy:
        How to treat NaNs, see :func:`_validate_series`.

    Returns
    -------
    ndarray
        Matrix profile of shape ``(N - m + 1,)``.

    Raises
    ------
    ValueError
        If ``m`` is not in ``[2, len(x)]``, ``exclusion`` is negative, ``x`` is
        invalid, or the series is too short for any subsequence to have a
        non-trivial match. The last condition needs
        ``N - m + 1 >= exclusion + 2``; with the default exclusion that means
        ``N >= m + m // 2 + 1``. Previously such inputs silently returned
        ``nan`` (every candidate was excluded, leaving an infinite profile
        that was then divided by infinity).

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
    excl = m // 2 if exclusion is None else int(exclusion)
    if excl < 0:
        raise ValueError("exclusion must be non-negative")
    n_sub = n - m + 1
    if n_sub < excl + 2:
        raise ValueError(
            f"series is too short: with m={m} and exclusion={excl} there are "
            f"{n_sub} subsequences but at least {excl + 2} are needed for every "
            "subsequence to have a non-trivial match; use a smaller m, a smaller "
            "exclusion, or a longer series"
        )
    windows = np.lib.stride_tricks.sliding_window_view(x, m)
    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True)
    z = (windows - means) / (stds + 1e-12)
    # Vectorised pairwise distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    sq_norms = np.sum(z**2, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (z @ z.T)
    np.maximum(dist_sq, 0.0, out=dist_sq)
    dist_mat = np.sqrt(dist_sq)
    # Mask the exclusion zone (trivial matches near the diagonal)
    idx = np.arange(n_sub)
    dist_mat[np.abs(idx[:, None] - idx[None, :]) <= excl] = np.inf
    profile = dist_mat.min(axis=1)
    if normalize:
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
register_encoder("mtspec", multitaper_spectrogram)
register_encoder("chirplet", chirplet_transform)
register_encoder("cwt", cwt)
register_encoder("sst", synchrosqueezed_cwt)
register_encoder("ph", persistence_image)
register_encoder("eph", extrema_persistence_histogram)
register_encoder("mtf", mtf)
register_encoder("otf", ordinal_transition_field)
register_encoder("gdf", gdf)
register_encoder("msrp", multi_scale_rp)
register_encoder("ded", delay_embedding_density)
register_encoder("dtw", dtw_matrix)
register_encoder("sax", sax)
register_encoder("msc", multi_scale_conv)
register_encoder("attn", window_attention)
register_encoder("vg", visibility_graph)
register_encoder("hvg", horizontal_visibility_graph)
register_encoder("shapelet", shapelet_transform)
register_encoder("mp", matrix_profile)
register_encoder("randproj", random_projection_image)
register_encoder("ensemble", ensemble)

# aliases matching documentation
register_encoder("visibility_graph", visibility_graph)
register_encoder("matrix_profile", matrix_profile)
register_encoder("persistence_image", persistence_image)
register_encoder("window_attention", window_attention)
def _scattering_encoder(x: Array, **kwargs: Any) -> Array:
    """Registry entry point for the scattering encoder.

    The import happens on call, not at module import: `scattering` imports
    this module for its validators, so importing it back at module scope makes
    the two modules circular whenever `scattering` is imported first.
    """

    from .scattering import _default_scattering

    return _default_scattering(x, **kwargs)


register_encoder("scat", _scattering_encoder)

# Kept for backwards compatibility: "tpa" resolves to `window_attention`, which
# is not the Temporal Pattern Attention architecture. See its docstring.
register_encoder("tpa", window_attention)

#: Names registered by this module, snapshotted before any user or plugin
#: registration. Use it to tell built-in encoders — which carry provenance
#: metadata in :mod:`tscv_vision.representations.metadata` — apart from
#: encoders added at runtime via :func:`register_encoder`.
BUILTIN_ENCODERS: frozenset[str] = frozenset(ENCODER_REGISTRY)


__all__ = [
    "gaf",
    "recurrence_plot",
    "spectrogram",
    "cwt",
    "multitaper_spectrogram",
    "SpectralScaling",
    "chirplet_transform",
    "ChirpletAggregate",
    "MAX_CHIRPLET_BYTES",
    "synchrosqueezed_cwt",
    "SSTWavelet",
    "persistence_diagram",
    "persistence_image",
    "extrema_persistence_histogram",
    "mtf",
    "ordinal_transition_field",
    "MAX_ORDINAL_ORDER",
    "TiePolicy",
    "OrdinalMode",
    "gdf",
    "multi_scale_rp",
    "delay_embedding_density",
    "DensityMode",
    "multi_scale_conv",
    "window_attention",
    "tpa",  # deprecated alias
    "dtw_matrix",
    "sax",
    "sax_symbols",
    "visibility_graph",
    "horizontal_visibility_graph",
    "HVGWeight",
    "shapelet_transform",
    "matrix_profile",
    "random_projection_image",
    "ensemble",
    "register_encoder",
    "get_encoder",
    "ENCODER_REGISTRY",
    "BUILTIN_ENCODERS",
    "NanPolicy",
]


