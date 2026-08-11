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
    "delay_embed",
    "cross_recurrence_plot",
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
