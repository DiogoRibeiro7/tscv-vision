"""Temporal aggregation utilities for feature sequences."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _skew(x: Array) -> Array:
    m = np.mean(x, axis=0)
    s = np.std(x, axis=0)
    return cast(Array, np.mean(((x - m) / (s + 1e-12)) ** 3, axis=0))


def _kurtosis(x: Array) -> Array:
    m = np.mean(x, axis=0)
    s = np.std(x, axis=0)
    return cast(Array, np.mean(((x - m) / (s + 1e-12)) ** 4, axis=0) - 3.0)


AGGREGATORS: dict[str, Callable[[Array], Array]] = {
    "mean": lambda x: cast(Array, np.mean(x, axis=0)),
    "median": lambda x: cast(Array, np.median(x, axis=0)),
    "var": lambda x: cast(Array, np.var(x, axis=0)),
    "min": lambda x: cast(Array, np.min(x, axis=0)),
    "max": lambda x: cast(Array, np.max(x, axis=0)),
    "skew": _skew,
    "kurt": _kurtosis,
}


def aggregate(features: Array, funcs: Sequence[str]) -> Array:
    """Aggregate feature sequences using one or more functions.

    Parameters
    ----------
    features:
        Feature matrix ``(N, D)`` where ``N`` is the number of windows.
    funcs:
        Sequence of aggregator names from :data:`AGGREGATORS`.

    Returns
    -------
    Array
        Concatenated aggregated features with shape ``(D * len(funcs),)``.
    """

    feats = np.asarray(features, dtype=float)
    if feats.ndim != 2:
        raise ValueError("features must be 2D")
    outs = []
    for name in funcs:
        func = AGGREGATORS.get(name)
        if func is None:
            raise ValueError(f"Unknown aggregator '{name}'")
        outs.append(func(feats))
    return np.concatenate(outs, axis=0)


__all__ = ["aggregate", "AGGREGATORS"]
