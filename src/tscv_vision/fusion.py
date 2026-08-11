"""Fusion utilities for combining encoder feature outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def fuse(
    features_list: Sequence[Array],
    mode: Literal["concat", "mean", "median", "weighted"] = "concat",
    weights: Sequence[float] | None = None,
) -> Array:
    """Fuse multiple feature arrays into a single representation.

    Parameters
    ----------
    features_list:
        Sequence of arrays with identical shape. Both feature matrices
        ``(N, D)`` and single feature vectors ``(D,)`` are accepted;
        concatenation always happens along the last axis.
    mode:
        Fusion strategy: ``"concat"`` stacks feature dimensions, while
        ``"mean"``, ``"median"``, and ``"weighted"`` aggregate across encoder
        outputs.
    weights:
        Optional weights for ``"weighted"`` mode. Must match the number of
        arrays provided.

    Returns
    -------
    Array
        Fused feature array. ``"concat"`` widens the last axis by a factor
        of ``M``, the number of inputs; the other modes preserve the input
        shape.
    """

    if len(features_list) == 0:
        raise ValueError("features_list must not be empty")
    arrs = [np.asarray(a, dtype=float) for a in features_list]
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError("All feature arrays must share the same shape")

    if mode == "concat":
        return cast(Array, np.concatenate(arrs, axis=-1))

    stack = np.stack(arrs, axis=0)
    if mode == "mean":
        return cast(Array, np.mean(stack, axis=0))
    if mode == "median":
        return cast(Array, np.median(stack, axis=0))
    if mode == "weighted":
        if weights is None:
            raise ValueError("weights required for weighted fusion")
        if len(weights) != len(arrs):
            raise ValueError("weights length must equal number of arrays")
        w = np.asarray(weights, dtype=float)
        w = w / np.sum(w)
        return cast(Array, np.tensordot(w, stack, axes=(0, 0)))
    raise ValueError("Unsupported fusion mode")


__all__ = ["fuse"]
