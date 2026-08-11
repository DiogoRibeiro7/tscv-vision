"""Encoders that consume more than one series.

:mod:`tscv_vision.encoders` validates a single 1D series, which is the right
contract for most of the package but the wrong one here: cross and joint
recurrence compare *different* trajectories, and their outputs may be
rectangular. Those encoders live here instead of being forced through a
univariate validator.

The shared machinery — delay embedding, pairwise distances, threshold
selection — is defined once and used by every encoder in this module.

.. note::
   :func:`tscv_vision.encoders.recurrence_plot` deliberately keeps its own
   implementation. It has Numba and Cython backends whose agreement with the
   NumPy path is separately regression-tested, so rerouting it through these
   utilities would risk that for no behavioural gain.
   ``test_cross_recurrence.py`` asserts the two paths agree exactly, which is
   the property the shared code was meant to guarantee.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from .encoders import NanPolicy, _validate_series

Array = NDArray[np.float64]

Metric = Literal["euclidean", "manhattan", "chebyshev"]

__all__ = [
    "Metric",
    "Combination",
    "WaveletCoherenceResult",
    "wavelet_coherence",
    "delay_embed",
    "cross_recurrence_plot",
    "joint_recurrence_plot",
]


def delay_embed(x: Array, dimension: int = 1, delay: int = 1) -> Array:
    r"""Return the delay-coordinate embedding of ``x``.

    .. math::

        X_i = \bigl(x_i,\; x_{i+\tau},\; \dots,\; x_{i+(m-1)\tau}\bigr)

    Parameters
    ----------
    x:
        Validated 1D series ``(N,)``.
    dimension:
        Embedding dimension :math:`m >= 1`. ``1`` returns the series as a
        column, which is the degenerate but frequently wanted case.
    delay:
        Embedding delay :math:`\tau >= 1`.

    Returns
    -------
    ndarray
        ``(N - (m - 1) * tau, m)`` matrix of state vectors.

    Raises
    ------
    ValueError
        If ``dimension`` or ``delay`` is below 1, or the series is too short.

    Examples
    --------
    >>> delay_embed(np.arange(5.0), dimension=2, delay=1)
    array([[0., 1.],
           [1., 2.],
           [2., 3.],
           [3., 4.]])
    """

    series = np.asarray(x, dtype=float)
    if series.ndim != 1:
        raise ValueError("x must be 1D")
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    if delay < 1:
        raise ValueError("delay must be >= 1")
    span = (dimension - 1) * delay
    count = series.size - span
    if count < 1:
        raise ValueError(
            f"series of length {series.size} is too short for dimension="
            f"{dimension}, delay={delay}; it needs at least {span + 1} samples"
        )
    embedded = np.empty((count, dimension), dtype=float)
    for k in range(dimension):
        start = k * delay
        embedded[:, k] = series[start : start + count]
    return embedded


def _pairwise_distance(left: Array, right: Array, metric: Metric) -> Array:
    """Distance between every pair of state vectors, ``(n_left, n_right)``."""

    difference = left[:, None, :] - right[None, :, :]
    if metric == "euclidean":
        return cast(Array, np.sqrt(np.sum(difference**2, axis=2)))
    if metric == "manhattan":
        return cast(Array, np.sum(np.abs(difference), axis=2))
    if metric == "chebyshev":
        return cast(Array, np.max(np.abs(difference), axis=2))
    raise ValueError("metric must be 'euclidean', 'manhattan' or 'chebyshev'")


def _recurrence_threshold(
    distances: Array, epsilon: float | None, recurrence_rate: float
) -> float:
    """Resolve the recurrence threshold.

    An explicit ``epsilon`` is returned unchanged. Otherwise the threshold is
    the ``recurrence_rate`` quantile of the observed distances, so that the
    resulting plot has approximately that recurrence rate by construction.
    Targeting a rate is the standard recommendation (Marwan et al., 2007,
    §3.2.1) precisely because a fixed distance means different things on
    differently-scaled data.
    """

    if epsilon is not None:
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be non-negative and finite")
        return float(epsilon)
    if not 0.0 < recurrence_rate < 1.0:
        raise ValueError("recurrence_rate must be in (0, 1)")
    return float(np.quantile(distances, recurrence_rate))


def cross_recurrence_plot(
    x: Array,
    y: Array,
    *,
    dimension: int = 1,
    delay: int = 1,
    epsilon: float | None = None,
    recurrence_rate: float = 0.1,
    metric: Metric = "euclidean",
    binary: bool = True,
    nan_policy: NanPolicy = "raise",
) -> Array:
    r"""Cross recurrence plot between two series.

    Where a recurrence plot asks when one trajectory revisits its own states,
    a cross recurrence plot asks when one trajectory visits states of
    *another*:

    .. math::

        CR[i, j] = \Theta\bigl(\varepsilon - \lVert X_i - Y_j \rVert\bigr)

    for delay embeddings :math:`X` of ``x`` and :math:`Y` of ``y``.

    Parameters
    ----------
    x, y:
        The two 1D series. **They need not have the same length**; the output
        is then rectangular, one row per embedded state of ``x`` and one
        column per embedded state of ``y``.
    dimension:
        Embedding dimension, shared by both series so their state vectors are
        comparable.
    delay:
        Embedding delay, likewise shared.
    epsilon:
        Recurrence threshold in the units of the chosen metric. ``None``
        (default) selects it automatically as the ``recurrence_rate`` quantile
        of the observed distances.
    recurrence_rate:
        Target recurrence rate in ``(0, 1)`` used when ``epsilon`` is ``None``.
        The rule is stated rather than hidden: the threshold is the quantile
        of the cross-distance distribution at this level, so roughly this
        fraction of the plot is filled.
    metric:
        ``"euclidean"``, ``"manhattan"`` or ``"chebyshev"``.
    binary:
        ``True`` thresholds at ``epsilon``. ``False`` returns
        ``1 - distance / max(distance)``, the same continuous convention as
        :func:`tscv_vision.encoders.recurrence_plot`, where 1 means identical.
    nan_policy:
        How to treat NaNs, applied to each series independently.

    Returns
    -------
    ndarray
        ``(N_x - (m-1)tau, N_y - (m-1)tau)`` image. Square only when the two
        series have the same length.

    Raises
    ------
    ValueError
        If either series is invalid or too short for the embedding, or any
        parameter is out of range.

    Notes
    -----
    **Complexity** ``O(N_x · N_y · m)`` time and ``O(N_x · N_y)`` memory. The
    distance tensor is materialised, so very long pairs need chunking.

    **Invariances** Equivariant to swapping the arguments: ``CR(y, x)`` is the
    transpose of ``CR(x, y)``. With an automatic threshold it is invariant to
    a common rescaling of both series, because the quantile rescales with the
    distances; with an explicit ``epsilon`` it is not, which is the point of
    preferring the rate.

    **Information lost** The binary form keeps only whether states are within
    ``epsilon``, discarding how far apart they are. Diagonal structure encodes
    shared dynamics but the plot alone cannot say which series leads.

    **Use cases** Detecting shared or lagged dynamics between two recordings;
    the offset of the main diagonal line measures their relative lag.

    References
    ----------
    Marwan, Romano, Thiel & Kurths (2007), "Recurrence plots for the analysis
    of complex systems", Physics Reports 438(5-6):237-329, §3.3.  Marwan &
    Kurths (2002), "Nonlinear analysis of bivariate data with cross recurrence
    plots", Physics Letters A 302(5-6):299-307.

    Examples
    --------
    >>> t = np.linspace(0, 6.0, 64)
    >>> cross_recurrence_plot(np.sin(t), np.sin(t)).shape
    (64, 64)
    >>> cross_recurrence_plot(np.sin(t), np.sin(t)[:32]).shape
    (64, 32)
    """

    first = _validate_series(x, nan_policy=nan_policy)
    second = _validate_series(y, nan_policy=nan_policy)
    left = delay_embed(first, dimension, delay)
    right = delay_embed(second, dimension, delay)

    distances = _pairwise_distance(left, right, metric)
    if not binary:
        peak = float(distances.max())
        if peak <= 0:
            identical: Array = np.ones_like(distances)
            return identical
        similarity: Array = 1.0 - distances / peak
        return similarity

    threshold = _recurrence_threshold(distances, epsilon, recurrence_rate)
    return cast(Array, (distances <= threshold).astype(float))


Combination = Literal["and", "product", "mean"]


def _validate_channels(X: Array, nan_policy: NanPolicy) -> Array:
    """Return ``X`` as a validated ``(n_samples, n_channels)`` float matrix."""

    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("X must be 2D with shape (n_samples, n_channels)")
    if matrix.shape[1] < 1:
        raise ValueError("X must have at least one channel")
    columns = [
        _validate_series(matrix[:, c], nan_policy=nan_policy)
        for c in range(matrix.shape[1])
    ]
    lengths = {column.size for column in columns}
    if len(lengths) > 1:
        raise ValueError(
            "channels have different lengths after applying nan_policy "
            f"({sorted(lengths)}); nan_policy='omit' can do this, so use "
            "'interpolate' or clean the data first"
        )
    return cast(Array, np.column_stack(columns))


def joint_recurrence_plot(
    X: Array,
    *,
    dimension: int = 1,
    delay: int = 1,
    epsilon: float | Array | None = None,
    recurrence_rate: float = 0.1,
    metric: Metric = "euclidean",
    combination: Combination = "and",
    nan_policy: NanPolicy = "raise",
) -> Array:
    r"""Joint recurrence plot of a multi-channel series.

    Each channel gets its own recurrence plot, with its **own** threshold, and
    the results are combined. The canonical definition is the logical AND:

    .. math::

        JR[i, j] = \prod_{c=1}^{C}
        \Theta\bigl(\varepsilon_c - \lVert X^{(c)}_i - X^{(c)}_j \rVert\bigr)
Vertigr)

    so a joint recurrence occurs only where *every* channel recurs
    simultaneously. Thresholding per channel is what makes this meaningful on
    channels with different units — a single global epsilon would let the
    largest-amplitude channel decide the result.

    Parameters
    ----------
    X:
        ``(n_samples, n_channels)`` matrix. Channels share the sample axis.
    dimension, delay:
        Embedding parameters, applied identically to every channel.
    epsilon:
        Threshold. A scalar applies to every channel; an array of length
        ``n_channels`` gives each its own; ``None`` (default) targets
        ``recurrence_rate`` per channel, as in :func:`cross_recurrence_plot`.
    recurrence_rate:
        Per-channel target rate used when ``epsilon`` is ``None``. The *joint*
        rate is lower — for ``C`` independent channels roughly
        ``recurrence_rate ** C``, which is the point of the AND.
    metric:
        Distance metric, shared by all channels.
    combination:
        ``"and"`` is the canonical binary definition. ``"product"`` and
        ``"mean"`` are **TSCV-Vision extensions** that combine the continuous
        per-channel similarities instead of thresholded indicators, giving a
        graded image; they are not part of the published joint recurrence
        plot.
    nan_policy:
        How to treat NaNs, applied per channel.

    Returns
    -------
    ndarray
        ``(W, W)`` symmetric matrix for ``W = n_samples - (m - 1) * tau``.
        Binary for ``"and"``, continuous in ``[0, 1]`` otherwise.

    Raises
    ------
    ValueError
        If ``X`` is not 2D, has no channels, the channels end up with different
        lengths, ``epsilon`` has the wrong shape, or a parameter is invalid.

    Notes
    -----
    **Complexity** ``O(C * W^2 * m)`` time and ``O(W^2)`` memory; per-channel
    plots are accumulated rather than all held at once.

    **Invariances** Invariant to channel order for all three combination
    rules, since AND, product and mean are commutative. Symmetric with a
    filled diagonal. With per-channel automatic thresholds it is invariant to
    rescaling **any individual channel**, which a shared epsilon would not be.

    **Information lost** The binary form records only simultaneous recurrence,
    so it cannot say which channels contributed. Channels recurring at
    different times cancel to zero even when each is individually structured.

    **Use cases** Detecting shared dynamical states across simultaneously
    recorded channels, and phase synchronisation analysis.

    References
    ----------
    Romano, Thiel, Kurths & von Bloh (2004), "Multivariate recurrence plots",
    Physics Letters A 330(3-4):214-223.  Marwan, Romano, Thiel & Kurths (2007),
    Physics Reports 438(5-6):237-329, section 3.4.

    Examples
    --------
    >>> t = np.linspace(0, 12.0, 64)
    >>> joint_recurrence_plot(np.column_stack([np.sin(t), np.cos(t)])).shape
    (64, 64)
    """

    matrix = _validate_channels(X, nan_policy)
    n_channels = matrix.shape[1]
    if combination not in {"and", "product", "mean"}:
        raise ValueError("combination must be 'and', 'product' or 'mean'")

    if epsilon is None:
        thresholds: list[float | None] = [None] * n_channels
    else:
        values = np.atleast_1d(np.asarray(epsilon, dtype=float))
        if values.size == 1:
            thresholds = [float(values[0])] * n_channels
        elif values.size == n_channels:
            thresholds = [float(v) for v in values]
        else:
            raise ValueError(
                "epsilon must be a scalar or have one entry per channel "
                f"({n_channels}), got {values.size}"
            )

    joint: Array | None = None
    for channel in range(n_channels):
        embedded = delay_embed(matrix[:, channel], dimension, delay)
        distances = _pairwise_distance(embedded, embedded, metric)
        if combination == "and":
            threshold = _recurrence_threshold(
                distances, thresholds[channel], recurrence_rate
            )
            plot = (distances <= threshold).astype(float)
        else:
            peak = float(distances.max())
            plot = np.ones_like(distances) if peak <= 0 else 1.0 - distances / peak
        if joint is None:
            joint = plot
        elif combination == "mean":
            joint = joint + plot
        else:
            # "and" multiplies indicators, "product" multiplies similarities.
            joint = joint * plot

    if joint is None:  # pragma: no cover - n_channels >= 1 is validated above
        raise ValueError("X must have at least one channel")
    if combination == "mean":
        joint = joint / n_channels
    result: Array = joint
    return result


class WaveletCoherenceResult(NamedTuple):
    """Coherence together with the axes and phase needed to interpret it."""

    coherence: Array
    """``(n_scales, N)`` squared wavelet coherence in ``[0, 1]``."""

    phase: Array
    """``(n_scales, N)`` phase of the cross-wavelet spectrum, in radians.

    Positive values mean ``x`` leads ``y`` at that time and scale.
    """

    scales: Array
    """``(n_scales,)`` wavelet scales in seconds."""

    frequencies: Array
    """``(n_scales,)`` Fourier frequencies the scales correspond to, in Hz."""


def _morlet_cwt(x: Array, scales: Array, dt: float, mu: float = 6.0) -> Array:
    """Analytic Morlet CWT, ``(n_scales, N)`` complex."""

    from .encoders import _analytic_wavelet_hat

    n = x.size
    spectrum = np.fft.fft(x)
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)
    out = np.empty((scales.size, n), dtype=complex)
    for index, scale in enumerate(scales):
        psi = _analytic_wavelet_hat(scale * omega, "morlet", mu)
        out[index] = np.fft.ifft(spectrum * psi)
    return out


def _smooth(power: Array, scales: Array, dt: float, dj: float) -> Array:
    """Smooth in time and then in scale, as wavelet coherence requires.

    Time smoothing uses a Gaussian whose width follows the scale; scale
    smoothing a boxcar of 0.6 octaves. Both follow Torrence & Webster (1999).
    Without them the coherence is identically one -- see
    :func:`wavelet_coherence`.
    """

    smoothed = np.empty_like(power)
    for index, scale in enumerate(scales):
        width = max(1, int(round(scale / dt)))
        offsets = np.arange(-2 * width, 2 * width + 1, dtype=float)
        kernel = np.exp(-0.5 * (offsets / width) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(power[index], offsets.size // 2, mode="reflect")
        smoothed[index] = np.convolve(padded, kernel, mode="valid")

    span = max(1, int(round(0.6 / dj)))
    if span > 1 and scales.size > 1:
        kernel = np.ones(span) / span
        padded = np.pad(smoothed, ((span, span), (0, 0)), mode="reflect")
        stacked = np.apply_along_axis(
            lambda column: np.convolve(column, kernel, mode="same"), 0, padded
        )
        smoothed = stacked[span : span + scales.size]
    result: Array = smoothed
    return result


def wavelet_coherence(
    x: Array,
    y: Array,
    *,
    fs: float = 1.0,
    scales: Array | None = None,
    voices: int = 12,
    wavelet: Literal["morlet"] = "morlet",
    smoothing: bool = True,
    return_phase: bool = False,
    nan_policy: NanPolicy = "raise",
) -> Array | WaveletCoherenceResult:
    r"""Squared wavelet coherence between two series.

    Localised, frequency-resolved coupling:

    .. math::

        R^2(t, s) = \frac{\bigl|S\bigl(s^{-1} W_{xy}(t, s)\bigr)\bigr|^2}
        {S\bigl(s^{-1}|W_x(t,s)|^2\bigr)\; S\bigl(s^{-1}|W_y(t,s)|^2\bigr)}

    where :math:`W_{xy} = W_x \overline{W_y}` is the cross-wavelet spectrum and
    :math:`S` smooths in time and scale.

    .. warning::
       The smoothing is not cosmetic. Without it the expression collapses to
       :math:`|W_x \overline{W_y}|^2 / (|W_x|^2 |W_y|^2) \equiv 1` at every
       point, for *any* pair of signals — a well-known degeneracy. Passing
       ``smoothing=False`` therefore returns an all-ones image that measures
       nothing; it exists to make the degeneracy inspectable, and a test
       asserts it.

    Parameters
    ----------
    x, y:
        Two 1D series of the **same length**; coherence is defined pointwise
        in time, so unequal lengths have no meaning here.
    fs:
        Sampling frequency in Hz.
    scales:
        Wavelet scales in seconds. ``None`` uses a dyadic grid with ``voices``
        per octave.
    voices:
        Scales per octave in the default grid, ``>= 1``.
    wavelet:
        Only ``"morlet"`` is supported: coherence needs an analytic wavelet,
        and the smoothing widths above are derived for Morlet specifically.
    smoothing:
        Apply the time and scale smoothing. See the warning.
    return_phase:
        Return a :class:`WaveletCoherenceResult` carrying the phase and axes
        instead of the bare image, so the return type follows the argument
        rather than the data.
    nan_policy:
        How to treat NaNs, applied to each series.

    Returns
    -------
    ndarray or WaveletCoherenceResult
        ``(n_scales, N)`` coherence in ``[0, 1]``, or the structured result.

    Raises
    ------
    ValueError
        If the series differ in length, ``fs`` or ``voices`` is invalid, the
        wavelet is unsupported, or a series is invalid.

    Notes
    -----
    **Complexity** ``O(S * N log N)`` for ``S`` scales, plus ``O(S * N * w)``
    for the time smoothing whose kernel width ``w`` follows the scale.

    **Invariances** Invariant to independently rescaling either series, since
    the normalisation divides by both auto-spectra. Equivariant to a common
    time shift. Swapping the arguments preserves the coherence and negates the
    phase.

    **Information lost** Coherence is a normalised quantity: it says how
    consistently the two series are related, not how strong either is. Two
    barely-present signals can be perfectly coherent. Amplitude lives in the
    individual wavelet spectra, not here.

    **Use cases** Finding intermittent, band-limited coupling between two
    recordings — the classic application being climate teleconnections.

    References
    ----------
    Torrence & Compo (1998), "A practical guide to wavelet analysis", BAMS
    79(1):61-78.  Torrence & Webster (1999), "Interdecadal changes in the
    ENSO-monsoon system", Journal of Climate 12(8):2679-2690.  Grinsted, Moore
    & Jevrejeva (2004), "Application of the cross wavelet transform and wavelet
    coherence to geophysical time series", Nonlinear Processes in Geophysics
    11(5-6):561-566.

    Examples
    --------
    >>> t = np.linspace(0, 10.0, 512)
    >>> coherence = wavelet_coherence(np.sin(6 * t), np.sin(6 * t + 0.5), fs=51.2)
    >>> bool(np.all((coherence >= 0) & (coherence <= 1)))
    True
    """

    first = _validate_series(x, nan_policy=nan_policy)
    second = _validate_series(y, nan_policy=nan_policy)
    if first.size != second.size:
        raise ValueError(
            f"x and y must have the same length for coherence, got "
            f"{first.size} and {second.size}; coherence is defined pointwise "
            "in time, so there is no meaningful pairing otherwise"
        )
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive, finite sampling frequency")
    if voices < 1:
        raise ValueError("voices must be >= 1")
    if wavelet != "morlet":
        raise ValueError(
            "only the Morlet wavelet is supported: the smoothing widths that "
            "make coherence meaningful are derived for it specifically"
        )

    dt = 1.0 / fs
    if scales is None:
        from .encoders import _default_sst_scales

        scale_grid = _default_sst_scales(first.size, dt, voices)
    else:
        scale_grid = np.asarray(scales, dtype=float)
        if scale_grid.ndim != 1 or scale_grid.size == 0 or np.any(scale_grid <= 0):
            raise ValueError("scales must be a non-empty 1D array of positive values")
    dj = math.log(2.0) / voices

    wx = _morlet_cwt(first, scale_grid, dt)
    wy = _morlet_cwt(second, scale_grid, dt)
    cross = wx * np.conjugate(wy)

    # The 1/s weighting makes the smoothed quantities comparable across scales.
    weight = (1.0 / scale_grid)[:, None]
    if smoothing:
        real = _smooth((weight * cross).real, scale_grid, dt, dj)
        imag = _smooth((weight * cross).imag, scale_grid, dt, dj)
        numerator = real**2 + imag**2
        denominator = _smooth(weight * np.abs(wx) ** 2, scale_grid, dt, dj) * _smooth(
            weight * np.abs(wy) ** 2, scale_grid, dt, dj
        )
    else:
        numerator = np.abs(weight * cross) ** 2
        denominator = (weight * np.abs(wx) ** 2) * (weight * np.abs(wy) ** 2)

    with np.errstate(invalid="ignore", divide="ignore"):
        coherence = np.where(denominator > 0, numerator / denominator, 0.0)
    coherence = np.clip(coherence, 0.0, 1.0)

    if not return_phase:
        image: Array = coherence
        return image
    # Morlet scale-to-frequency conversion, Torrence & Compo table 1.
    mu = 6.0
    factor = (4.0 * np.pi) / (mu + math.sqrt(2.0 + mu**2))
    return WaveletCoherenceResult(
        coherence=coherence,
        phase=np.angle(cross),
        scales=scale_grid,
        frequencies=1.0 / (factor * scale_grid),
    )
