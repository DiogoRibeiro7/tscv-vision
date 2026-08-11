"""AutoML utilities for tscv-vision.

This module implements lightweight helpers that analyse time series and
recommend encoders or feature subsets. The implementation favours simple
heuristics and NumPy-only dependencies so it can run in minimal
environments. The utilities here are optional and are designed to keep
the core API stable while enabling automatic configuration for new
datasets.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .encoders import ENCODER_REGISTRY
from .features import extract_feature_vector

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
        Best evaluated parameter combination.

    Raises
    ------
    ValueError
        If ``search_space`` is empty or the sizing arguments are not positive.

    Notes
    -----
    The best individual is tracked as evaluations happen. Selecting it from
    the final population by ``argmax`` of the previous generation's scores —
    as this function did before 0.2.0 — can return an unevaluated offspring
    that has nothing to do with the reported score.
    """

    keys = list(search_space)
    if not keys:
        raise ValueError("search_space cannot be empty")
    if generations < 1 or population < 2:
        raise ValueError("generations must be >= 1 and population >= 2")
    rng = np.random.default_rng(0)

    def random_individual() -> list[Any]:
        return [rng.choice(search_space[k]) for k in keys]

    population_data = [random_individual() for _ in range(population)]
    best_individual: list[Any] = population_data[0]
    best_score = -np.inf

    for _ in range(generations):
        scores = np.empty(len(population_data))
        for i, individual in enumerate(population_data):
            params = dict(zip(keys, individual, strict=True))
            scores[i] = objective(params)
            if scores[i] > best_score:
                best_score = float(scores[i])
                best_individual = list(individual)
        idx = np.argsort(-scores)[: max(1, len(population_data) // 2)]
        parents = [population_data[i] for i in idx]
        # produce offspring with simple mutation
        offspring: list[list[Any]] = []
        while len(parents) + len(offspring) < population:
            p = parents[rng.integers(len(parents))].copy()
            j = rng.integers(len(keys))
            p[j] = rng.choice(search_space[keys[j]])
            offspring.append(p)
        population_data = parents + offspring

    return dict(zip(keys, best_individual, strict=True))


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


class _BaseModel:
    """Minimal estimator interface used by :class:`AutoTSCV`.

    The implementations mimic the ``fit``/``predict`` API from scikit-learn but
    avoid any heavy dependencies. Only the small subset required for the tests
    is implemented.
    """

    def fit(self, X: Array, y: Array) -> _BaseModel:
        raise NotImplementedError

    def predict(self, X: Array) -> Array:
        raise NotImplementedError

    def get_params(self) -> Mapping[str, Any]:  # pragma: no cover - interface
        return {}


class _KNNClassifier(_BaseModel):
    def __init__(self, k: int = 1) -> None:
        self.k = k
        self._X: Array | None = None
        self._y: Array | None = None

    def fit(self, X: Array, y: Array) -> _KNNClassifier:
        self._X = X
        self._y = y.astype(int)
        return self

    def predict(self, X: Array) -> Array:
        if self._X is None or self._y is None:
            raise RuntimeError("model not fitted")
        X_train = self._X
        y_train = self._y
        dists = np.linalg.norm(X_train[None, :, :] - X[:, None, :], axis=2)
        idx = np.argpartition(dists, self.k - 1, axis=1)[:, : self.k]
        votes = y_train[idx]
        if self.k == 1:
            return votes[:, 0]
        return np.array([np.bincount(v).argmax() for v in votes])

    def get_params(self) -> Mapping[str, Any]:
        return {"k": self.k}


class _KNNRegressor(_BaseModel):
    def __init__(self, k: int = 1) -> None:
        self.k = k
        self._X: Array | None = None
        self._y: Array | None = None

    def fit(self, X: Array, y: Array) -> _KNNRegressor:
        self._X = X
        self._y = y
        return self

    def predict(self, X: Array) -> Array:
        if self._X is None or self._y is None:
            raise RuntimeError("model not fitted")
        X_train = self._X
        y_train = self._y
        dists = np.linalg.norm(X_train[None, :, :] - X[:, None, :], axis=2)
        idx = np.argpartition(dists, self.k - 1, axis=1)[:, : self.k]
        return cast(Array, y_train[idx].mean(axis=1))

    def get_params(self) -> Mapping[str, Any]:
        return {"k": self.k}


class _ModeClassifier(_BaseModel):
    def __init__(self) -> None:
        self._mode: int | None = None

    def fit(self, X: Array, y: Array) -> _ModeClassifier:
        counts = np.bincount(y.astype(int))
        self._mode = int(counts.argmax())
        return self

    def predict(self, X: Array) -> Array:
        if self._mode is None:
            raise RuntimeError("model not fitted")
        return np.full(X.shape[0], self._mode, dtype=int)


class _MeanRegressor(_BaseModel):
    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, X: Array, y: Array) -> _MeanRegressor:
        self._mean = float(np.mean(y))
        return self

    def predict(self, X: Array) -> Array:
        if self._mean is None:
            raise RuntimeError("model not fitted")
        return np.full(X.shape[0], self._mean, dtype=float)


def _param_grid(space: Mapping[str, Sequence[Any]]) -> Iterator[Mapping[str, Any]]:
    keys = list(space)
    if not keys:
        yield {}
        return
    for values in product(*(space[k] for k in keys)):
        yield dict(zip(keys, values, strict=True))


def _ts_splits(n: int, n_splits: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    fold = max(1, n // (n_splits + 1))
    for i in range(n_splits):
        train_end = fold * (i + 1)
        test_end = min(n, fold * (i + 2))
        if train_end >= n:
            break
        train = np.arange(train_end)
        test = np.arange(train_end, test_end)
        if test.size == 0:
            break
        yield train, test


@dataclass
class AutoTSCV:
    """Automated machine-learning pipeline for time series.

    The class provides a very small AutoML system that profiles the input
    data, selects encoders and simple estimators, and performs a grid search to
    maximise a multi-objective score balancing predictive accuracy and
    computational cost. The implementation avoids heavy dependencies and is
    intended as a demonstration; it can be extended to integrate libraries such
    as scikit-learn, Optuna, or Ray when available.

    Parameters
    ----------
    task:
        ``"classification"`` or ``"regression"``.
    tradeoff:
        Weight applied to the computational cost (feature dimensionality) when
        scoring models.
    random_state:
        Optional seed for internal randomness.
    """

    task: Literal["classification", "regression"] = "classification"
    tradeoff: float = 0.0
    random_state: int | None = None
    cv: int = 3

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_state)
        self.profile_: Mapping[str, float] | None = None
        self.encoder_: str | None = None
        self.encoder_params_: Mapping[str, Any] = {}
        self.model_: _BaseModel | None = None
        self.feature_importances_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Profiling
    def _profile(self, X: Sequence[Array]) -> Mapping[str, float]:
        lengths = np.array([x.size for x in X], dtype=float)
        means = np.array([np.mean(x) for x in X], dtype=float)
        variances = np.array([np.var(x) for x in X], dtype=float)
        diff_var = np.array([np.var(np.diff(x)) for x in X], dtype=float)
        periodicity: list[float] = []
        for x in X:
            fft_mag = np.abs(np.fft.rfft(x - np.mean(x)))
            if fft_mag.size > 1:
                periodicity.append(float(np.argmax(fft_mag[1:]) + 1) / x.size)
        noise_vals = [float(np.std(x - np.mean(x))) for x in X]
        miss_vals = [float(np.mean(np.isnan(x))) for x in X]
        profile = {
            "length": float(np.mean(lengths)),
            "mean": float(np.mean(means)),
            "variance": float(np.mean(variances)),
            "stationarity": float(np.mean(diff_var / (variances + 1e-12))),
            "periodicity": float(np.mean(periodicity) if periodicity else 0.0),
            "noise": float(np.mean(noise_vals)),
            "missing": float(np.mean(miss_vals)),
        }
        return profile

    def profile(self, X: Sequence[Array]) -> Mapping[str, float]:
        self.profile_ = self._profile(X)
        return self.profile_

    # ------------------------------------------------------------------
    # Fitting
    def _candidate_models(
        self,
    ) -> Mapping[str, tuple[type[_BaseModel], Mapping[str, Sequence[Any]]]]:
        if self.task == "classification":
            return {
                "knn": (_KNNClassifier, {"k": [1]}),
                "majority": (_ModeClassifier, {}),
            }
        return {
            "knn": (_KNNRegressor, {"k": [1]}),
            "mean": (_MeanRegressor, {}),
        }

    def _encode(self, X: Sequence[Array], name: str, params: Mapping[str, Any]) -> Array:
        func = ENCODER_REGISTRY[name]
        images = [func(x, **params) for x in X]
        feats = np.stack([extract_feature_vector(im) for im in images])
        return feats

    def _cross_val(
        self, X: Array, y: Array, model_cls: type[_BaseModel], params: Mapping[str, Any]
    ) -> float:
        scores: list[float] = []
        for train_idx, test_idx in _ts_splits(X.shape[0], self.cv):
            model = model_cls(**params)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            if self.task == "classification":
                scores.append(float(np.mean(pred == y[test_idx])))
            else:
                scores.append(-float(np.mean((pred - y[test_idx]) ** 2)))
        return float(np.mean(scores)) if scores else -np.inf

    @staticmethod
    def _fingerprint(X: Sequence[Array], y: Array) -> str:
        """Cheap content hash used to detect train/validation reuse."""

        digest = hashlib.sha256()
        for x in X:
            digest.update(np.ascontiguousarray(x, dtype=float).tobytes())
        digest.update(np.ascontiguousarray(y).tobytes())
        return digest.hexdigest()

    def fit(self, X: Sequence[Array], y: Array) -> AutoTSCV:
        """Profile the data, then search encoders and estimators by CV."""
        X = list(X)
        if len(X) != y.shape[0]:
            raise ValueError("X and y length mismatch")
        self._train_fingerprint = self._fingerprint(X, y)
        self.profile(X)
        candidates = suggest_encoders(X[0])
        best_score = -np.inf
        best_feats: Array | None = None
        for enc in candidates:
            params_list: list[dict[str, Any]] = [{}]
            for params in params_list:
                feats = self._encode(X, enc, params)
                for _, (model_cls, grid) in self._candidate_models().items():
                    for m_params in _param_grid(grid):
                        score = self._cross_val(feats, y, model_cls, m_params)
                        cost = feats.shape[1]
                        objective = score - self.tradeoff * cost
                        if objective > best_score:
                            best_score = objective
                            self.encoder_ = enc
                            self.encoder_params_ = params
                            self.model_ = model_cls(**m_params).fit(feats, y)
                            best_feats = feats
        if best_feats is None or self.model_ is None or self.encoder_ is None:
            raise RuntimeError("AutoTSCV could not fit a model")
        self.feature_importances_ = rank_feature_importance(best_feats, y)
        return self

    # ------------------------------------------------------------------
    # Prediction and validation
    def predict(self, X: Sequence[Array]) -> Array:
        """Predict targets for ``X`` with the selected encoder and model."""
        if self.model_ is None or self.encoder_ is None:
            raise RuntimeError("AutoTSCV not fitted")
        feats = self._encode(X, self.encoder_, self.encoder_params_)
        return self.model_.predict(feats)

    def validate(self, X: Sequence[Array], y: Array) -> float:
        """Score the *already selected* pipeline on ``X``/``y``.

        .. warning::
           This is only a generalisation estimate when ``X``/``y`` were **not**
           seen by :meth:`fit`. The encoder and estimator were chosen by
           cross-validating on the training data, so re-scoring that same data
           is optimistically biased; a :class:`UserWarning` is emitted if the
           inputs are byte-identical to the training set. For an unbiased
           number without a held-out split, use :meth:`nested_score`.

        Parameters
        ----------
        X, y:
            Preferably a held-out set that :meth:`fit` never saw.

        Returns
        -------
        float
            Mean score over the time-series splits of ``X``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """

        if self.model_ is None or self.encoder_ is None:
            raise RuntimeError("AutoTSCV not fitted")
        X = list(X)
        if getattr(self, "_train_fingerprint", None) == self._fingerprint(X, y):
            warnings.warn(
                "validate() was called on the data used by fit(); encoders and "
                "models were selected on this data, so the result is an "
                "optimistically biased model-selection score, not a "
                "generalisation estimate. Pass a held-out set or use "
                "nested_score().",
                UserWarning,
                stacklevel=2,
            )
        feats = self._encode(X, self.encoder_, self.encoder_params_)
        return self._cross_val(feats, y, type(self.model_), self.model_.get_params())

    def nested_score(self, X: Sequence[Array], y: Array, *, outer_splits: int = 3) -> float:
        """Unbiased score via nested (forward-chaining) cross-validation.

        The entire :meth:`fit` procedure — profiling, encoder selection and
        model selection — is repeated on each outer training split and scored
        on the following, unseen block. No choice is informed by the data it
        is scored on.

        Parameters
        ----------
        X, y:
            Full dataset; splits respect temporal order.
        outer_splits:
            Number of outer forward-chaining splits.

        Returns
        -------
        float
            Mean outer-fold score, or ``-inf`` if no split was usable.

        Raises
        ------
        ValueError
            If ``X`` and ``y`` lengths disagree.
        """

        X = list(X)
        if len(X) != y.shape[0]:
            raise ValueError("X and y length mismatch")
        scores: list[float] = []
        for train_idx, test_idx in _ts_splits(len(X), outer_splits):
            inner = AutoTSCV(
                task=self.task,
                tradeoff=self.tradeoff,
                random_state=self.random_state,
                cv=self.cv,
            )
            try:
                inner.fit([X[i] for i in train_idx], y[train_idx])
            except (RuntimeError, ValueError):  # pragma: no cover - degenerate split
                continue
            pred = inner.predict([X[i] for i in test_idx])
            truth = y[test_idx]
            if self.task == "classification":
                scores.append(float(np.mean(pred == truth)))
            else:
                scores.append(-float(np.mean((pred - truth) ** 2)))
        return float(np.mean(scores)) if scores else -np.inf

    # ------------------------------------------------------------------
    # Drift detection
    def detect_drift(self, X_new: Sequence[Array], threshold: float = 0.1) -> bool:
        if self.profile_ is None:
            raise RuntimeError("profile not available")
        new_profile = self._profile(X_new)
        diffs = [
            abs(new_profile[k] - self.profile_[k]) / (abs(self.profile_[k]) + 1e-12)
            for k in self.profile_
        ]
        return bool(float(np.mean(diffs)) > threshold)
