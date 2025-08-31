from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import numpy as np
from numpy.typing import NDArray

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
except Exception as exc:  # pragma: no cover - handled at runtime
    raise ImportError("scikit-learn is required for domain adaptation") from exc

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]
Extractor = Callable[[Array], Array]


class DomainAdapter:
    """Fine-tune a pre-trained feature extractor for a specific domain.

    Parameters
    ----------
    extractor:
        Callable mapping a raw series to a feature vector.
    base_estimator:
        Optional pre-trained estimator. Defaults to ``LogisticRegression``.

    Examples
    --------
    >>> from tscv_vision.domains.finance import microstructure_features, generate_price_series
    >>> series = [generate_price_series(50, drift=-0.01), generate_price_series(50, drift=0.01)]
    >>> labels = np.array([0, 1])
    >>> adapter = DomainAdapter(microstructure_features)
    >>> adapter.fit(series, labels)
    DomainAdapter(...)
    """

    def __init__(
        self,
        extractor: Extractor,
        base_estimator: LogisticRegression | None = None,
    ) -> None:
        self.extractor = extractor
        self.model = base_estimator or LogisticRegression(max_iter=100, warm_start=True)

    def _transform(self, series: Sequence[Array]) -> Array:
        features = [self.extractor(np.asarray(s, dtype=float)) for s in series]
        return np.asarray(np.vstack(features), dtype=float)

    def fit(self, series: Sequence[Array], labels: Array) -> DomainAdapter:
        X = self._transform(series)
        self.model.fit(X, labels)
        return self

    def predict(self, series: Sequence[Array]) -> IntArray:
        X = self._transform(series)
        return np.asarray(self.model.predict(X), dtype=int)

    def predict_proba(self, series: Sequence[Array]) -> Array:
        X = self._transform(series)
        return np.asarray(self.model.predict_proba(X), dtype=float)


def classification_metrics(y_true: Array, y_pred: Array) -> dict[str, float]:
    """Compute accuracy, precision, recall and F1 metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro")),
        "recall": float(recall_score(y_true, y_pred, average="macro")),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }


class PrototypicalClassifier:
    """Few-shot classifier using class prototypes.

    Examples
    --------
    >>> clf = PrototypicalClassifier().fit([[0, 0], [1, 1]], [0, 1])
    >>> clf.predict([[0.1, 0.2]])
    array([0])
    """

    def fit(self, X: Array, y: Array) -> PrototypicalClassifier:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
            raise ValueError("invalid shapes for X and y")
        self.classes_ = np.unique(y)
        self.prototypes_ = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X: Array) -> IntArray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        dists = np.sum((X[:, None, :] - self.prototypes_) ** 2, axis=2)
        idx = np.argmin(dists, axis=1)
        return np.asarray(self.classes_[idx], dtype=int)


def uncertainty_sampling(
    model: DomainAdapter,
    series: Sequence[Array],
    n_samples: int,
) -> IntArray:
    """Select indices of the most uncertain samples using margin sampling."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    probs = model.predict_proba(series)
    if probs.ndim != 2:
        raise ValueError("model must output 2D probability array")
    if probs.shape[1] == 2:
        margin = np.abs(probs[:, 0] - 0.5)
    else:
        margin = probs.max(axis=1)
    order = np.argsort(margin)
    return np.asarray(order[:n_samples], dtype=int)
