"""Adaptive feature engineering pipelines."""


from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

try:  # optional dependency
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
except Exception:  # pragma: no cover - optional dependency
    BaseEstimator = cast(Any, object)
    TransformerMixin = cast(Any, object)
    mutual_info_classif = cast(Any, None)
    GaussianProcessRegressor = cast(Any, None)
    Matern = cast(Any, None)
    LogisticRegression = cast(Any, None)
    KFold = cast(Any, None)
    StratifiedKFold = cast(Any, None)
    cross_val_score = cast(Any, None)
    Pipeline = cast(Any, None)

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


class FeatureSelector(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """scikit-learn transformer wrapping :func:`select_features`.

    Exists so that supervised feature selection can be placed **inside** a
    :class:`~sklearn.pipeline.Pipeline` and therefore re-fitted on each
    training fold. Selecting features on the full matrix and then calling
    :func:`~sklearn.model_selection.cross_val_score` leaks the validation
    folds into the selection step and inflates the reported score.

    Parameters
    ----------
    method:
        Selection criterion, see :func:`select_features`.
    k:
        Number of features to keep.
    cv:
        Inner folds used by the ``"stability"`` method.
    random_state:
        Seed for reproducibility.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    """

    def __init__(
        self,
        *,
        method: str = "mutual_info",
        k: int = 10,
        cv: int = 3,
        random_state: int | None = None,
    ) -> None:
        if cross_val_score is None:  # pragma: no cover - optional dependency
            raise ImportError("scikit-learn required for FeatureSelector")
        self.method = method
        self.k = k
        self.cv = cv
        self.random_state = random_state

    def fit(self, X: Array, y: Array) -> FeatureSelector:
        """Fit the selector on ``X``/``y`` only."""
        X = _validate_dataset(X)
        y = _validate_target(y, X.shape[0])
        self.support_ = select_features(
            X,
            y,
            method=self.method,
            k=min(self.k, X.shape[1]),
            cv=self.cv,
            random_state=self.random_state,
        )
        self.n_features_in_ = int(X.shape[1])
        return self

    def transform(self, X: Array) -> Array:
        """Restrict ``X`` to the selected columns."""
        support = getattr(self, "support_", None)
        if support is None:
            raise ValueError("FeatureSelector must be fitted first")
        X = _validate_dataset(X)
        selected: Array = X[:, support]
        return selected


def _make_estimator(
    *,
    method: str,
    k: int,
    cv: int,
    random_state: int | None,
) -> Any:
    """Build a leakage-safe ``selection -> classifier`` pipeline."""

    return Pipeline(
        [
            (
                "select",
                FeatureSelector(method=method, k=k, cv=cv, random_state=random_state),
            ),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )


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

    Notes
    -----
    :meth:`fit` chooses the encoder by cross-validating on the data it is
    given, so any score computed on that same data afterwards is optimistic.
    Use :meth:`nested_score` for a generalisation estimate.
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

        Feature selection runs **inside** each cross-validation fold (through
        :class:`FeatureSelector` in an sklearn :class:`~sklearn.pipeline.Pipeline`),
        so the reported per-configuration scores are not contaminated by the
        validation folds. Before 0.2.0 selection was performed once on the
        whole matrix, which leaked every fold into the selection step.

        .. warning::
           The returned score is the **maximum over the searched
           configurations** and is therefore still optimistically biased as an
           estimate of generalisation ("winner's curse"): the search itself saw
           all of ``X``. Use :meth:`nested_score` for an unbiased estimate, or
           evaluate the chosen configuration on a held-out test set.

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
            cross-validation accuracy observed during the search.

        Raises
        ------
        ImportError
            If scikit-learn is not installed.
        """

        X = _validate_dataset(X)
        y = _validate_target(y, X.shape[0])
        if (
            GaussianProcessRegressor is None
            or Matern is None
            or LogisticRegression is None
            or cross_val_score is None
        ):
            raise ImportError("scikit-learn required for optimization")
        best_enc, best_k, best_score, _ = self._search(X, y, n_iter=n_iter)
        return best_enc, best_k, best_score

    def _search(
        self, X: Array, y: Array, *, n_iter: int
    ) -> tuple[str, int, float, list[float]]:
        """Run the Bayesian search and return the best config plus all scores."""

        sample = self._extract_all(X[:1], self.encoders[0])
        n_features = sample.shape[1]
        ks = np.arange(5, min(50, n_features) + 1, 5)
        if ks.size == 0:
            ks = np.array([n_features], dtype=int)
        configs: list[list[float]] = []
        scores: list[float] = []
        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), random_state=self.random_state)
        rng = np.random.default_rng(self.random_state)
        cache: dict[str, Array] = {}
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
            if name not in cache:
                cache[name] = self._extract_all(X, name)
            feats = cache[name]
            est = _make_estimator(
                method=self.feature_select,
                k=min(int(k), feats.shape[1]),
                cv=self.cv,
                random_state=self.random_state,
            )
            score = float(cross_val_score(est, feats, y, cv=self.cv).mean())
            configs.append([enc_idx, float(int(k))])
            scores.append(score)
        best = int(np.argmax(scores))
        return (
            self.encoders[int(configs[best][0])],
            int(configs[best][1]),
            float(scores[best]),
            scores,
        )

    def nested_score(
        self, X: Array, y: Array, *, n_iter: int = 10, outer_cv: int = 5
    ) -> float:
        """Unbiased accuracy estimate via nested cross-validation.

        The complete selection procedure — encoder search, feature-count
        search and feature selection — is re-run from scratch on each outer
        training fold and scored on the corresponding held-out fold, which
        never influences any choice. This is the number to report in a paper;
        the score returned by :meth:`optimize` is a model-selection score, not
        a generalisation estimate.

        Parameters
        ----------
        X, y:
            Dataset and labels.
        n_iter:
            Inner search iterations per outer fold.
        outer_cv:
            Number of outer folds.

        Returns
        -------
        float
            Mean accuracy over the outer folds.

        Raises
        ------
        ImportError
            If scikit-learn is not installed.
        """

        X = _validate_dataset(X)
        y = _validate_target(y, X.shape[0])
        if StratifiedKFold is None or cross_val_score is None:
            raise ImportError("scikit-learn required for nested_score")
        splitter = StratifiedKFold(
            n_splits=outer_cv, shuffle=True, random_state=self.random_state
        )
        outer_scores: list[float] = []
        for train_idx, test_idx in splitter.split(X, y):
            inner = AdaptivePipeline(
                encoders=list(self.encoders),
                feature_select=self.feature_select,
                k=self.k,
                cv=self.cv,
                random_state=self.random_state,
            )
            enc, best_k, _, _ = inner._search(X[train_idx], y[train_idx], n_iter=n_iter)
            train_feats = inner._extract_all(X[train_idx], enc)
            test_feats = inner._extract_all(X[test_idx], enc)
            est = _make_estimator(
                method=self.feature_select,
                k=min(best_k, train_feats.shape[1]),
                cv=self.cv,
                random_state=self.random_state,
            )
            est.fit(train_feats, y[train_idx])
            outer_scores.append(float(np.mean(est.predict(test_feats) == y[test_idx])))
        return float(np.mean(outer_scores))


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


__all__ = ["AdaptivePipeline", "FeatureEnsemble", "FeatureSelector", "select_features"]
