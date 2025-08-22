"""AutoML utilities for tscv-vision.

This module implements lightweight helpers that analyse time series and
recommend encoders or feature subsets. The implementation favours simple
heuristics and NumPy-only dependencies so it can run in minimal
environments. The utilities here are optional and are designed to keep
the core API stable while enabling automatic configuration for new
datasets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def suggest_encoders(series: Array, max_encoders: int = 2) -> list[str]:
    """Suggest encoder names based on the characteristics of ``series``.

    Parameters
    ----------
    series:
        1D time series to inspect.
    max_encoders:
        Maximum number of encoder names to return.

    Returns
    -------
    list[str]
        Encoder names ordered by preference.

    Notes
    -----
    This routine uses simple heuristics based on series length, variance
    and dominant frequency. It does not rely on any heavy statistical
    tests and is meant only as a starting point.
    """
    if series.ndim != 1 or series.size == 0:
        raise ValueError("series must be a non-empty 1D array")

    length = series.size
    var = float(np.var(series))
    # rough frequency estimate
    fft_mag = np.abs(np.fft.rfft(series))
    dom_freq = float(np.argmax(fft_mag[1:]) + 1) / length

    recommendations: list[str] = []

    if length >= 128 and dom_freq > 0.1:
        recommendations.append("spec")
    if var < 1e-8:
        recommendations.append("rp")
    else:
        recommendations.append("gaf")

    if length >= 256:
        recommendations.append("cwt")

    return recommendations[:max_encoders]


def rank_feature_importance(
    X: Array, y: Array, *, method: str = "corr", bins: int = 16
) -> NDArray[np.int_]:
    """Rank features according to their relevance to ``y``.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(n_samples, n_features)``.
    y:
        Target vector of shape ``(n_samples,)``.
    method:
        ``"corr"`` for Pearson correlation magnitude, ``"mi"`` for a
        simple discrete mutual information estimate.
    bins:
        Number of bins to use for the mutual information estimate.

    Returns
    -------
    np.ndarray
        Indices of features sorted from most to least important.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be 1D and match X")

    if method == "corr":
        y_c = y - y.mean()
        X_c = X - X.mean(axis=0)
        num = np.abs(X_c.T @ y_c)
        den = np.linalg.norm(X_c, axis=0) * float(np.linalg.norm(y_c))
        scores = np.divide(num, den, out=np.zeros_like(num), where=den != 0.0)
    elif method == "mi":
        y_d = np.digitize(y, np.linspace(y.min(), y.max(), bins + 1))
        scores = np.empty(X.shape[1])
        for i in range(X.shape[1]):
            x_d = np.digitize(
                X[:, i], np.linspace(X[:, i].min(), X[:, i].max(), bins + 1)
            )
            joint = np.histogram2d(x_d, y_d, bins=bins)[0]
            px = joint.sum(axis=1, keepdims=True)
            py = joint.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                mi = joint * (np.log(joint + 1e-12) - np.log(px) - np.log(py))
            scores[i] = float(mi.sum())
    else:
        raise ValueError("unknown method")

    return np.argsort(-scores)


@dataclass
class MetaLearner:
    """Very small meta-learning container.

    The meta-learner stores dataset statistics and preferred
    configurations. Suggestions are made by finding the stored entry with
    the closest ``length``.
    """

    store: list[tuple[float, Mapping[str, Any]]] = field(default_factory=list)

    def update(self, length: int, config: Mapping[str, Any]) -> None:
        """Record the best ``config`` for a series of ``length``."""
        self.store.append((float(length), dict(config)))

    def suggest(self, length: int) -> Mapping[str, Any] | None:
        """Return the config for the closest known length, if any."""
        if not self.store:
            return None
        distances = [abs(stored_len - length) for stored_len, _ in self.store]
        idx = int(np.argmin(distances))
        return self.store[idx][1]


def active_window_selection(
    series: Array, candidates: Sequence[int], *, hop_ratio: float = 0.5
) -> tuple[int, int]:
    """Choose a window size and hop based on variance criteria.

    Parameters
    ----------
    series:
        Input 1D time series.
    candidates:
        Candidate window lengths.
    hop_ratio:
        Hop size as a fraction of the chosen window length.

    Returns
    -------
    tuple[int, int]
        Selected window length and hop size.
    """
    if series.ndim != 1:
        raise ValueError("series must be 1D")
    if not candidates:
        raise ValueError("candidates cannot be empty")

    best_len = candidates[0]
    best_score = -np.inf
    for win in candidates:
        if win > series.size:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(series, win)
        score = float(np.mean(np.var(windows, axis=1)))
        if score > best_score:
            best_score = score
            best_len = win
    hop = max(1, int(best_len * hop_ratio))
    return best_len, hop


def evolve_hyperparams(
    objective: Callable[[Mapping[str, Any]], float],
    search_space: Mapping[str, Sequence[Any]],
    *,
    generations: int = 5,
    population: int = 10,
) -> Mapping[str, Any]:
    """Basic evolutionary search over ``search_space``.

    Parameters
    ----------
    objective:
        Function returning a score to maximise.
    search_space:
        Mapping from parameter name to list of candidate values.
    generations:
        Number of evolutionary generations.
    population:
        Number of individuals per generation.

    Returns
    -------
    Mapping[str, Any]
        Best found parameter combination.
    """
    keys = list(search_space)
    rng = np.random.default_rng(0)

    def random_individual() -> list[Any]:
        return [rng.choice(search_space[k]) for k in keys]

    population_data = [random_individual() for _ in range(population)]
    scores = np.empty(population)

    for _ in range(generations):
        for i, individual in enumerate(population_data):
            params = dict(zip(keys, individual, strict=True))
            scores[i] = objective(params)
        idx = np.argsort(-scores)[: population // 2]
        parents = [population_data[i] for i in idx]
        # produce offspring with simple mutation
        offspring: list[list[Any]] = []
        while len(parents) + len(offspring) < population:
            p = parents[rng.integers(len(parents))].copy()
            j = rng.integers(len(keys))
            p[j] = rng.choice(search_space[keys[j]])
            offspring.append(p)
        population_data = parents + offspring

    best_idx = int(np.argmax(scores))
    return dict(zip(keys, population_data[best_idx], strict=True))


def select_feature_subset(
    X: Array,
    y: Array,
    *,
    max_features: int,
    objective: Callable[[Array, Array], float],
) -> NDArray[np.int_]:
    """Greedy feature subset selection.

    Parameters
    ----------
    X, y:
        Dataset.
    max_features:
        Maximum number of features to retain.
    objective:
        Callable returning a score given ``(X_subset, y)`` to maximise.

    Returns
    -------
    np.ndarray
        Indices of the selected features.
    """
    remaining = list(range(X.shape[1]))
    selected: list[int] = []
    for _ in range(max_features):
        best_score = -np.inf
        best_idx = None
        for idx in remaining:
            trial = selected + [idx]
            score = objective(X[:, trial], y)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return np.array(selected, dtype=int)
