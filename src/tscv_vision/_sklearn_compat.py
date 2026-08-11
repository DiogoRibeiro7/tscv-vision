"""scikit-learn base classes, with usable fallbacks when it is absent.

Several modules expose scikit-learn-compatible transformers while keeping
scikit-learn optional. They all need the same thing: the real base classes when
available, and stubs that still provide the estimator protocol when not. That
lives here once, because getting it wrong is easy and invisible on a machine
with scikit-learn installed — two earlier attempts shipped broken:

* importing the bases only under ``TYPE_CHECKING``, so at runtime the
  transformer always inherited empty stubs and never received
  ``fit_transform``;
* aliasing both names to ``object``, making ``class T(TransformerMixin,
  BaseEstimator)`` a ``TypeError: duplicate base class object`` at import time.

``tests/test_optional_dependency_isolation.py`` covers both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["BaseEstimator", "TransformerMixin", "HAS_SKLEARN"]

if TYPE_CHECKING:  # pragma: no cover - import for type check only
    from sklearn.base import BaseEstimator, TransformerMixin

    HAS_SKLEARN = True
else:
    try:  # prefer the real scikit-learn base classes when available
        from sklearn.base import BaseEstimator, TransformerMixin

        HAS_SKLEARN = True
    except Exception:  # pragma: no cover - runtime fallback without scikit-learn
        HAS_SKLEARN = False

        class BaseEstimator:  # type: ignore[no-redef]
            """Stub base class used when scikit-learn is not installed.

            Reproduces just enough of the estimator protocol
            (``get_params``/``set_params``) for the package's transformers to be
            usable standalone. It is *not* a drop-in replacement: install the
            ``ml`` extra for real interoperability.
            """

            def get_params(self, deep: bool = True) -> dict[str, Any]:
                """Return the constructor parameters of this estimator."""
                import inspect

                names = [
                    p.name
                    for p in inspect.signature(type(self).__init__).parameters.values()
                    if p.name != "self" and p.kind is not p.VAR_KEYWORD
                ]
                return {name: getattr(self, name) for name in names}

            def set_params(self, **params: Any) -> BaseEstimator:
                """Set constructor parameters on this estimator."""
                for key, value in params.items():
                    if key not in self.get_params():
                        raise ValueError(f"Invalid parameter {key!r}")
                    setattr(self, key, value)
                return self

        class TransformerMixin:  # type: ignore[no-redef]
            """Stub mixin providing ``fit_transform`` without scikit-learn."""

            def fit_transform(self, X: Any, y: Any = None, **kwargs: Any) -> Any:
                """Fit to ``X`` then transform it."""
                if y is None:
                    return self.fit(X, **kwargs).transform(X)  # type: ignore[attr-defined]
                return self.fit(X, y, **kwargs).transform(X)  # type: ignore[attr-defined]
