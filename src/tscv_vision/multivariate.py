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
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from .encoders import NanPolicy, _validate_series

Array = NDArray[np.float64]

Metric = Literal["euclidean", "manhattan", "chebyshev"]

__all__ = [
    "Metric",
    "Combination",
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
