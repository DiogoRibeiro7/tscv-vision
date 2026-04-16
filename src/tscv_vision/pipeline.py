"""Adaptive feature engineering pipelines."""


from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

try:  # optional dependency
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_val_score
except Exception:  # pragma: no cover - optional dependency
    mutual_info_classif = cast(Any, None)
    GaussianProcessRegressor = cast(Any, None)
    Matern = cast(Any, None)
    LogisticRegression = cast(Any, None)
    KFold = cast(Any, None)
    cross_val_score = cast(Any, None)

from .encoders import ENCODER_REGISTRY, get_encoder
from .features import extract_feature_vector

Array = NDArray[np.float64]


def _validate_dataset(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, series_len)")
    if X.size == 0:
        raise ValueError("X cannot be empty")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values")
    return X


def _validate_target(y: Array, n: int) -> Array:
    y = np.asarray(y)
    if y.ndim != 1 or y.size != n:
        raise ValueError("y must be 1D with length matching X")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values")
    return y


def select_features(
    X: Array,
    y: Array,
    *,
    method: str = "mutual_info",
    k: int = 10,
    cv: int = 3,
    random_state: int | None = None,
) -> NDArray[np.int64]:
    """Return indices of the top ``k`` features according to ``method``.

    Parameters
    ----------
    X:
        Feature matrix ``(n_samples, n_features)``.
    y:
        Target array ``(n_samples,)``.
    method:
        ``"mutual_info"``, ``"correlation"`` or ``"stability"``.
    k:
        Number of features to select.
    cv:
        Number of folds for stability selection.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    ndarray
        Indices of selected features sorted in ascending order.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    X = _validate_dataset(X)
    y = _validate_target(y, X.shape[0])
    if X.shape[1] <= k:
        return np.arange(X.shape[1])

    rng = np.random.default_rng(random_state)

    if method == "mutual_info":
        if mutual_info_classif is None:
            raise ImportError("scikit-learn required for mutual_info method")
        scores = mutual_info_classif(X, y, random_state=random_state)
    elif method == "correlation":
        corr = np.corrcoef(X, y, rowvar=False)
        scores = np.abs(corr[:-1, -1])
    elif method == "stability":
        if mutual_info_classif is None or KFold is None:
            raise ImportError("scikit-learn required for stability selection")
        counts = np.zeros(X.shape[1], dtype=float)
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        for train, _ in kf.split(X):
            mi = mutual_info_classif(
                X[train], y[train], random_state=rng.integers(1_000_000_000)
            )
            top = np.argsort(mi)[-k:]
            counts[top] += 1
        scores = counts
    else:
        raise ValueError("unknown method")
    idx = np.argsort(scores)[-k:]
    return np.sort(idx)


@dataclass
class AdaptivePipeline:
    """Select encoders and features based on data characteristics.

    Examples
    --------
    >>> from tscv_vision.pipeline import AdaptivePipeline
    >>> import numpy as np
    >>> X = np.random.rand(10, 32)
    >>> y = (X.mean(axis=1) > 0.5).astype(int)
    >>> pipe = AdaptivePipeline(encoders=["gaf", "recurrence_plot"], random_state=0)
    >>> feats = pipe.fit_transform(X, y)
    >>> feats.shape[0]
    10
    """

    encoders: list[str] = field(default_factory=list)
    feature_select: str = "mutual_info"
    k: int = 10
    cv: int = 3
    random_state: int | None = None

    selected_encoder_: str | None = field(init=False, default=None)
    selected_features_: NDArray[np.int64] | None = field(init=False, default=None)
    model_: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.encoders:
            self.encoders = list(ENCODER_REGISTRY)
        if not self.encoders:
            raise ValueError("No encoders available")

    def _extract_all(self, X: Array, name: str) -> Array:
        func = get_encoder(name)
        imgs = [func(x) for x in X]
        vecs = [extract_feature_vector(img) for img in imgs]
        return np.vstack(vecs)

    def fit(self, X: Array, y: Array) -> AdaptivePipeline:
        X = _validate_dataset(X)
        y = _validate_target(y, X.shape[0])
        if LogisticRegression is None or cross_val_score is None:
            raise ImportError("scikit-learn required for AdaptivePipeline")
        best_score = -np.inf
        best_name = self.encoders[0]
        for name in self.encoders:
            feats = self._extract_all(X, name)
            lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
            score = cross_val_score(lr, feats, y, cv=self.cv).mean()
            if score > best_score:
                best_score = score
                best_name = name
        self.selected_encoder_ = best_name
        feats = self._extract_all(X, best_name)
        self.selected_features_ = select_features(
            feats,
            y,
            method=self.feature_select,
            k=min(self.k, feats.shape[1]),
            cv=self.cv,
            random_state=self.random_state,
        )
        self.model_ = LogisticRegression(max_iter=1000, random_state=self.random_state).fit(
            feats[:, self.selected_features_], y
        )
        return self

    def transform(self, X: Array) -> Array:
        if self.selected_encoder_ is None or self.selected_features_ is None:
            raise ValueError("Pipeline must be fitted first")
        X = _validate_dataset(X)
        feats = self._extract_all(X, self.selected_encoder_)
        return feats[:, self.selected_features_]

    def fit_transform(self, X: Array, y: Array) -> Array:
        self.fit(X, y)
        return self.transform(X)

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> AdaptivePipeline:
        """Load a pipeline previously written by :meth:`save`.

        .. warning::
            This uses :mod:`pickle` and must only be called on files from a
            trusted source. Unpickling attacker-controlled data can execute
            arbitrary code.
        """

        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, AdaptivePipeline):
            raise TypeError("Invalid pipeline file")
        return obj

    def optimize(self, X: Array, y: Array, *, n_iter: int = 10) -> tuple[str, int, float]:
        """Return best encoder and feature count via Bayesian optimization.

        Parameters
        ----------
        X, y:
            Dataset and labels used for evaluation.
        n_iter:
            Number of optimization iterations.

        Returns
        -------
        tuple
            ``(best_encoder, best_k, score)`` where ``score`` is the best
            cross-validation accuracy.
        """

        X = _validate_dataset(X)
        y = _validate_target(y, X.shape[0])
        sample = self._extract_all(X[:1], self.encoders[0])
        n_features = sample.shape[1]
        ks = np.arange(5, min(50, n_features) + 1, 5)
        configs: list[list[float]] = []
        scores: list[float] = []
        if (
            GaussianProcessRegressor is None
            or Matern is None
            or LogisticRegression is None
            or cross_val_score is None
        ):
            raise ImportError("scikit-learn required for optimization")
        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), random_state=self.random_state)
        rng = np.random.default_rng(self.random_state)
        for _ in range(max(2, n_iter)):
            if len(configs) >= 2:
                gp.fit(np.array(configs), np.array(scores))
                grid = np.array(
                    [[float(i), float(k)] for i in range(len(self.encoders)) for k in ks],
                    dtype=np.float64,
                )
                mean, std = gp.predict(grid, return_std=True)
                idx = int(np.argmax(mean + 1.96 * std))
                enc_idx, k = grid[idx]
            else:
                enc_idx = float(rng.integers(len(self.encoders)))
                k = float(rng.choice(ks))
            name = self.encoders[int(enc_idx)]
            feats = self._extract_all(X, name)
            sel = select_features(
                feats,
                y,
                method=self.feature_select,
                k=min(int(k), feats.shape[1]),
                cv=self.cv,
                random_state=self.random_state,
            )
            lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
            score = cross_val_score(lr, feats[:, sel], y, cv=self.cv).mean()
            configs.append([enc_idx, float(int(k))])
            scores.append(score)
        best = int(np.argmax(scores))
        best_enc = self.encoders[int(configs[best][0])]
        best_k = int(configs[best][1])
        return best_enc, best_k, float(scores[best])


@dataclass
class FeatureEnsemble:
    """Combine multiple encoders using learned cross-validated weights.

    Parameters
    ----------
    encoders:
        Sequence of encoder names to aggregate.
    cv:
        Number of cross-validation folds used to derive weights.
    random_state:
        Seed for reproducible scoring.

    Examples
    --------
    >>> from tscv_vision.pipeline import FeatureEnsemble
    >>> import numpy as np
    >>> X = np.random.rand(5, 16)
    >>> y = (X.mean(axis=1) > 0.5).astype(int)
    >>> ens = FeatureEnsemble(["gaf", "recurrence_plot"], random_state=0)
    >>> feats = ens.fit_transform(X, y)
    >>> feats.shape[0]
    5
    """

    encoders: Sequence[str]
    cv: int = 3
    random_state: int | None = None

    weights_: NDArray[np.float64] | None = field(init=False, default=None)

    def _extract_all(self, X: Array, name: str) -> Array:
        func = get_encoder(name)
        imgs = [func(x) for x in X]
        vecs = [extract_feature_vector(img) for img in imgs]
        return np.vstack(vecs)

    def fit(self, X: Array, y: Array) -> FeatureEnsemble:
        X = _validate_dataset(X)
        y = _validate_target(y, X.shape[0])
        if LogisticRegression is None or cross_val_score is None:
            raise ImportError("scikit-learn required for FeatureEnsemble")
        scores: list[float] = []
        for name in self.encoders:
            feats = self._extract_all(X, name)
            lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
            score = cross_val_score(lr, feats, y, cv=self.cv).mean()
            scores.append(score)
        weights = np.array(scores, dtype=float)
        total = float(weights.sum())
        if total == 0:
            weights[:] = 1.0 / len(self.encoders)
        else:
            weights /= total
        self.weights_ = weights
        return self

    def transform(self, X: Array) -> Array:
        weights = self.weights_
        if weights is None:
            raise ValueError("Ensemble must be fitted first")
        X = _validate_dataset(X)
        feats = []
        for w, name in zip(weights, self.encoders, strict=True):
            f = self._extract_all(X, name)
            feats.append(w * f)
        return np.hstack(feats)

    def fit_transform(self, X: Array, y: Array) -> Array:
        self.fit(X, y)
        return self.transform(X)


__all__ = ["AdaptivePipeline", "FeatureEnsemble", "select_features"]
