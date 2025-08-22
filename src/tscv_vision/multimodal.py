"""Multi-modal and cross-domain utilities.

This module provides lightweight helpers for fusing multiple time series,
combining heterogeneous modalities and performing simple domain adaptation
or graph-based processing.  The implementations favour NumPy-only
approaches so that they can run in constrained environments without heavy
machine-learning dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def fuse_series(
    series: Array,
    method: Literal["weighted", "pca"] = "weighted",
    weights: Array | None = None,
    n_components: int = 1,
) -> Array:
    """Fuse multi-variate time series into a lower-dimensional representation.

    Parameters
    ----------
    series:
        Array of shape ``(C, T)`` containing ``C`` channels.
    method:
        ``"weighted"`` for a weighted average or ``"pca"`` for a principal
        component projection.
    weights:
        Optional weights for the ``"weighted"`` method. Defaults to uniform
        weighting.
    n_components:
        Number of components to keep when ``method="pca"``.

    Returns
    -------
    Array
        Fused series with shape ``(n_components, T)``.

    Raises
    ------
    ValueError
        If input validation fails.
    """

    if series.ndim != 2:
        raise ValueError("series must have shape (C, T)")
    channels, _ = series.shape
    if method == "weighted":
        w = np.asarray(weights, dtype=float) if weights is not None else np.ones(channels)
        if w.shape[0] != channels:
            raise ValueError("weights must match number of channels")
        w = w / np.sum(w)
        return np.tensordot(w, series, axes=(0, 0))[None, :]
    if method == "pca":
        if n_components < 1 or n_components > channels:
            raise ValueError("invalid number of components")
        mean = series.mean(axis=1, keepdims=True)
        centered = series - mean
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        proj = (u[:, :n_components].T @ centered)
        return cast(Array, proj)
    raise ValueError("Unsupported fusion method")


def cross_modal_concat(series: Array, metadata: Array) -> Array:
    """Combine time-series data with metadata features via concatenation.

    Both arrays are flattened and concatenated to form a joint representation.
    The function assumes metadata remains constant over time.
    """

    if series.ndim != 1:
        raise ValueError("series must be 1D")
    meta = np.asarray(metadata, dtype=float).ravel()
    return np.concatenate([series.ravel(), meta])


def coral_align(source: Array, target: Array) -> Array:
    """Align ``source`` features to match the covariance of ``target``.

    Implements CORAL (CORrelation ALignment) domain adaptation using a
    whitening and re-colouring transform.
    """

    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be 2D feature arrays")
    cov_s = np.cov(source, rowvar=False) + np.eye(source.shape[1]) * 1e-10
    cov_t = np.cov(target, rowvar=False) + np.eye(target.shape[1]) * 1e-10
    u_s, s_s, _ = np.linalg.svd(cov_s)
    u_t, s_t, _ = np.linalg.svd(cov_t)
    whiten = u_s @ np.diag(s_s ** -0.5) @ u_s.T
    color = u_t @ np.diag(s_t ** 0.5) @ u_t.T
    return cast(Array, (source - source.mean(axis=0)) @ whiten @ color + target.mean(axis=0))


def temporal_graph_propagate(
    series: Array,
    adjacency: Array,
    steps: int = 1,
) -> Array:
    """Propagate information across related series using a graph.

    Parameters
    ----------
    series:
        Array with shape ``(N, T)`` for ``N`` nodes over time.
    adjacency:
        Square ``(N, N)`` adjacency matrix with non-negative weights.
    steps:
        Number of message-passing steps to perform.
    """

    if series.ndim != 2:
        raise ValueError("series must be (N, T)")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if adjacency.shape[0] != series.shape[0]:
        raise ValueError("adjacency size must match number of nodes")
    prop = series.astype(float)
    norm = adjacency / np.maximum(adjacency.sum(axis=1, keepdims=True), 1.0)
    for _ in range(max(steps, 0)):
        prop = norm @ prop
    return prop


def granger_causality(x: Array, y: Array, maxlag: int = 1) -> float:
    """Compute a simple Granger-causality score from ``x`` to ``y``.

    The implementation fits linear autoregressive models via least squares and
    returns the log ratio of residual variances. Positive values suggest that
    ``x`` helps predict ``y``.
    """

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D")
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    n = len(x) - maxlag
    if n <= maxlag:
        raise ValueError("Time series too short for requested lag")
    X = np.column_stack([y[maxlag - i : len(y) - i] for i in range(1, maxlag + 1)])
    Y = y[maxlag:]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid_y = Y - X @ beta
    lagged_x = np.column_stack([x[maxlag - i : len(x) - i] for i in range(1, maxlag + 1)])
    X2 = np.column_stack([X, lagged_x])
    beta2, *_ = np.linalg.lstsq(X2, Y, rcond=None)
    resid_xy = Y - X2 @ beta2
    var_y = np.var(resid_y)
    var_xy = np.var(resid_xy)
    return float(np.log(var_y / var_xy))


def federated_average(parameters: Sequence[Array], weights: Sequence[float] | None = None) -> Array:
    """Compute a weighted average of model parameters for federated learning.

    Parameters must all share the same shape.
    """

    if len(parameters) == 0:
        raise ValueError("parameters must not be empty")
    arrs = [np.asarray(p, dtype=float) for p in parameters]
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError("All parameter arrays must share the same shape")
    if weights is None:
        w = np.ones(len(arrs)) / len(arrs)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != len(arrs):
            raise ValueError("weights length must match number of parameter arrays")
        w = w / np.sum(w)
    stacked = np.stack(arrs, axis=0)
    return np.tensordot(w, stacked, axes=(0, 0))


__all__ = [
    "fuse_series",
    "cross_modal_concat",
    "coral_align",
    "temporal_graph_propagate",
    "granger_causality",
    "federated_average",
]
