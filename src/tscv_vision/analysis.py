"""Feature selection, importance, and representation analysis utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def _representation_matrix(values: Array, *, name: str) -> Array:
    """Return ``values`` as a finite ``(n_samples, n_features)`` matrix."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be array-like")
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return cast(Array, matrix)


def _centered_gram(values: Array) -> Array:
    """Return the centered linear Gram matrix for ``values``."""

    centered = values - values.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    row_mean = gram.mean(axis=1, keepdims=True)
    col_mean = gram.mean(axis=0, keepdims=True)
    total_mean = float(gram.mean())
    return cast(Array, gram - row_mean - col_mean + total_mean)


def representation_alignment(
    left: Array,
    right: Array,
    *,
    method: str = "linear_cka",
) -> float:
    """Measure alignment between two representation matrices.

    Parameters
    ----------
    left, right:
        Representation matrices. Inputs may be 1D, 2D, or higher-dimensional;
        the first axis is interpreted as samples and all remaining axes are
        flattened into features.
    method:
        Currently only ``"linear_cka"`` is implemented. Linear centered kernel
        alignment is invariant to isotropic rescaling and orthogonal rotations
        of either representation.

    Returns
    -------
    float
        Similarity in ``[0, 1]`` for non-degenerate inputs. A degenerate
        constant representation has zero alignment with everything.

    Raises
    ------
    ValueError
        If the sample counts differ, inputs are invalid, or ``method`` is not
        supported.
    """

    if method != "linear_cka":
        raise ValueError("method must be 'linear_cka'")
    x = _representation_matrix(left, name="left")
    y = _representation_matrix(right, name="right")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"left and right must have the same number of samples, got "
            f"{x.shape[0]} and {y.shape[0]}"
        )
    k = _centered_gram(x)
    other = _centered_gram(y)
    denom = np.linalg.norm(k, ord="fro") * np.linalg.norm(other, ord="fro")
    if denom <= 0.0:
        return 0.0
    score = float(np.sum(k * other) / denom)
    return float(np.clip(score, 0.0, 1.0))


def representation_similarity(
    representations: Mapping[str, Array] | Sequence[Array],
    *,
    method: str = "linear_cka",
) -> Array:
    """Return a pairwise representation-similarity matrix.

    Parameters
    ----------
    representations:
        Mapping or sequence of representation matrices. The first axis of
        every matrix must contain the same samples in the same order.
    method:
        Similarity metric. Currently ``"linear_cka"``.

    Returns
    -------
    ndarray
        Symmetric ``(n_representations, n_representations)`` matrix with a
        unit diagonal for non-degenerate representations.
    """

    if isinstance(representations, Mapping):
        values = list(representations.values())
    else:
        values = list(representations)
    if not values:
        raise ValueError("representations cannot be empty")
    n = len(values)
    out = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = representation_alignment(
                values[i],
                values[j],
                method=method,
            )
    return out


def representation_redundancy(
    representations: Mapping[str, Array] | Sequence[Array],
    *,
    method: str = "linear_cka",
) -> float:
    """Return the mean off-diagonal similarity among representations."""

    matrix = representation_similarity(representations, method=method)
    if matrix.shape[0] == 1:
        return 0.0
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return float(np.mean(upper))


def representation_complementarity(
    individual_scores: Mapping[str, float],
    fused_scores: Mapping[tuple[str, str], float],
) -> list[dict[str, float | str]]:
    """Quantify pairwise fusion gains over the best constituent score.

    Parameters
    ----------
    individual_scores:
        Downstream score per single representation.
    fused_scores:
        Downstream score per pair. Pair order is ignored.

    Returns
    -------
    list of dict
        Rows with ``left``, ``right``, ``fused_score``,
        ``best_individual_score`` and ``improvement``.
    """

    rows: list[dict[str, float | str]] = []
    for (left, right), fused in fused_scores.items():
        if left not in individual_scores or right not in individual_scores:
            raise ValueError(f"missing individual score for pair {(left, right)!r}")
        best = max(float(individual_scores[left]), float(individual_scores[right]))
        rows.append(
            {
                "left": left,
                "right": right,
                "fused_score": float(fused),
                "best_individual_score": best,
                "improvement": float(fused) - best,
            }
        )
    return rows


def representation_effective_rank(values: Array, *, eps: float = 1e-12) -> float:
    """Return the entropy-based effective rank of a representation.

    The representation is centered, singular values are normalised into a
    probability vector, and the rank is ``exp(entropy(p))``. A constant matrix
    has effective rank zero.
    """

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    matrix = _representation_matrix(values, name="values")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    total = float(singular.sum())
    if total <= eps:
        return 0.0
    probabilities = singular / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + eps)))
    return float(np.exp(entropy))


__all__ = [
    "variance_threshold",
    "topk_variance",
    "feature_importance_corr",
    "representation_alignment",
    "representation_similarity",
    "representation_redundancy",
    "representation_complementarity",
    "representation_effective_rank",
]

