"""Feature selection and importance utilities."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]


def variance_threshold(features: Array, threshold: float) -> tuple[Array, BoolArray]:
    """Select features with variance above ``threshold``."""

    feats = np.asarray(features, dtype=float)
    if feats.ndim != 2:
        raise ValueError("features must be 2D")
    var = feats.var(axis=0)
    mask = var >= threshold
    return feats[:, mask], mask


def topk_variance(features: Array, k: int) -> tuple[Array, IntArray]:
    """Select the ``k`` features with highest variance."""

    feats = np.asarray(features, dtype=float)
    if feats.ndim != 2:
        raise ValueError("features must be 2D")
    if k <= 0 or k > feats.shape[1]:
        raise ValueError("k must be in 1..D")
    var = feats.var(axis=0)
    idx = np.argsort(var)[-k:]
    return feats[:, idx], idx.astype(int)


def feature_importance_corr(features: Array, target: Array) -> Array:
    """Compute absolute correlation of each feature with ``target``."""

    feats = np.asarray(features, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if feats.ndim != 2:
        raise ValueError("features must be 2D")
    if tgt.ndim != 1 or tgt.shape[0] != feats.shape[0]:
        raise ValueError("target must be 1D and match number of samples")
    feats_center = feats - feats.mean(axis=0)
    tgt_center = tgt - tgt.mean()
    num = feats_center.T @ tgt_center
    denom = np.sqrt(np.sum(feats_center**2, axis=0) * np.sum(tgt_center**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.abs(num / denom)
    corr[~np.isfinite(corr)] = 0.0
    return cast(Array, corr.astype(float))


__all__ = ["variance_threshold", "topk_variance", "feature_importance_corr"]

