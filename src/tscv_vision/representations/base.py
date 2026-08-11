"""Abstract interfaces shared by every representation.

Three families exist, and the distinction is deliberate — conflating them is
how test data leaks into model selection:

:class:`Representation`
    A pure function of one input. Nothing is learned, so there is nothing that
    could see the test set.
:class:`FittedRepresentation`
    Has parameters estimated from data. ``fit`` must only ever see training
    data; the class refuses to ``transform`` before ``fit``.
:class:`PretrainedRepresentation`
    Wraps externally trained weights. Nothing is fitted here, but the weights
    were fitted elsewhere, which is a provenance fact users need to record.

None of these inherit from scikit-learn. Use :func:`as_sklearn` when a
scikit-learn estimator is required, so that the dependency stays optional.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metadata import RepresentationInfo

FloatArray = NDArray[np.floating[Any]]

__all__ = [
    "FloatArray",
    "Representation",
    "FittedRepresentation",
    "PretrainedRepresentation",
    "NotFittedError",
    "as_sklearn",
]


class NotFittedError(RuntimeError):
    """Raised when a fitted representation is used before ``fit``."""


class Representation(ABC):
    """A deterministic mapping from one input to one representation.

    Subclasses implement :meth:`transform` and :attr:`info`. Everything else —
    batching, repr, equality of configuration — is provided here.
    """

    @abstractmethod
    def transform(self, x: FloatArray) -> FloatArray:
        """Map a single input to its representation.

        Parameters
        ----------
        x:
            One series (or image, depending on ``info.input_kind``).

        Returns
        -------
        ndarray
            The representation, with the shape documented by
            ``info.output_kind``.
        """

    @property
    @abstractmethod
    def info(self) -> RepresentationInfo:
        """Provenance and validation metadata for this representation."""

    def transform_many(self, X: Iterable[FloatArray]) -> list[FloatArray]:
        """Transform each element of ``X``.

        A list rather than an array, because encoders whose output size
        depends on the input length (spectrograms, matrix profiles) cannot be
        stacked when the inputs differ in length. Use :meth:`transform_stack`
        when a rectangular result is required.
        """

        return [self.transform(x) for x in X]

    def transform_stack(self, X: Iterable[FloatArray]) -> FloatArray:
        """Transform each element of ``X`` and stack the results.

        Raises
        ------
        ValueError
            If ``X`` is empty or the outputs do not share a shape.
        """

        out = self.transform_many(X)
        if not out:
            raise ValueError("cannot stack an empty collection")
        shapes = {arr.shape for arr in out}
        if len(shapes) > 1:
            raise ValueError(
                f"{self.info.name} produced outputs of differing shapes "
                f"{sorted(shapes)}; the inputs likely differ in length. Use "
                "transform_many() and handle them individually."
            )
        return np.stack(out)

    def iter_transform(self, X: Iterable[FloatArray]) -> Iterator[FloatArray]:
        """Lazily transform ``X``, one representation at a time."""

        for x in X:
            yield self.transform(x)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the configuration of this representation.

        The default reads the public attributes set by ``__init__``; override
        when that is not faithful. ``deep`` exists only so that
        :func:`sklearn.base.clone` can traverse a representation held as a
        parameter of a wrapper; nested configuration is not expanded.
        """

        return {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_") and not key.endswith("_")
        }

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in sorted(self.get_params().items()))
        return f"{type(self).__name__}({params})"


class FittedRepresentation(Representation):
    """A representation whose parameters are estimated from data.

    Subclasses implement :meth:`_fit` and :meth:`transform`, and must set
    nothing on ``self`` outside ``_fit`` that ``transform`` depends on.

    .. warning::
       ``fit`` must only see training data. When a fitted representation is
       used inside cross-validation, fit it within each training fold — see
       :class:`tscv_vision.pipeline.FeatureSelector` for the pattern, or wrap
       it with :func:`as_sklearn` and put it in a
       :class:`~sklearn.pipeline.Pipeline`.
    """

    _is_fitted: bool = False

    @abstractmethod
    def _fit(self, X: Sequence[FloatArray], y: Any | None = None) -> None:
        """Estimate parameters from ``X`` (and optionally ``y``)."""

    def fit(self, X: Sequence[FloatArray], y: Any | None = None) -> FittedRepresentation:
        """Estimate parameters from training data and return ``self``."""

        self._fit(list(X), y)
        self._is_fitted = True
        return self

    def fit_transform(
        self, X: Sequence[FloatArray], y: Any | None = None
    ) -> list[FloatArray]:
        """Fit on ``X`` then transform it."""

        return self.fit(X, y).transform_many(X)

    def check_fitted(self) -> None:
        """Raise :class:`NotFittedError` unless :meth:`fit` has been called."""

        if not self._is_fitted:
            raise NotFittedError(
                f"{type(self).__name__} must be fitted before transforming; "
                "call fit(X_train) on training data only"
            )


class PretrainedRepresentation(Representation):
    """A representation backed by externally pretrained weights.

    Subclasses implement :meth:`encode`, which receives a batch, because
    pretrained backbones are far more efficient batched. :meth:`transform`
    delegates to it.

    .. note::
       Pretrained weights were fitted on some corpus. If that corpus overlaps
       your evaluation data, results are contaminated in a way no
       cross-validation scheme can repair. Record the checkpoint in
       ``info.reference`` and say so in the write-up.
    """

    @abstractmethod
    def encode(self, X: Sequence[FloatArray]) -> FloatArray:
        """Encode a batch, returning ``(len(X), ...)``."""

    def transform(self, x: FloatArray) -> FloatArray:
        """Encode a single input via :meth:`encode`."""

        out: FloatArray = self.encode([x])[0]
        return out

    def transform_many(self, X: Iterable[FloatArray]) -> list[FloatArray]:
        """Encode ``X`` in one batched call."""

        return list(self.encode(list(X)))


def as_sklearn(representation: Representation, *, stack: bool = True) -> Any:
    """Wrap ``representation`` as a scikit-learn transformer.

    Kept out of the classes themselves so that scikit-learn stays optional:
    representations work without it, and only this function needs it.

    Parameters
    ----------
    representation:
        Any representation. A :class:`FittedRepresentation` is fitted inside
        the transformer's ``fit``, which makes it safe to place in a
        :class:`~sklearn.pipeline.Pipeline` and cross-validate.
    stack:
        Return a stacked ``(n_samples, ...)`` array. Set ``False`` to get a
        list, for representations whose output size varies with input length.

    Returns
    -------
    sklearn.base.TransformerMixin
        A transformer delegating to ``representation``.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline           # doctest: +SKIP
    >>> from tscv_vision.representations import get_representation
    >>> tr = as_sklearn(get_representation("gaf"))      # doctest: +SKIP
    """

    try:
        from sklearn.base import BaseEstimator, TransformerMixin
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("scikit-learn is required for as_sklearn") from exc

    class _RepresentationTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
        """scikit-learn transformer delegating to a Representation."""

        def __init__(
            self, representation: Representation = representation, stack: bool = stack
        ) -> None:
            self.representation = representation
            self.stack = stack

        def fit(self, X: Any, y: Any | None = None) -> Any:
            """Fit the wrapped representation when it needs fitting."""
            if isinstance(self.representation, FittedRepresentation):
                self.representation.fit(list(X), y)
            return self

        def transform(self, X: Any) -> Any:
            """Transform every row of ``X``."""
            rows = list(X)
            if self.stack:
                return self.representation.transform_stack(rows)
            return self.representation.transform_many(rows)

    return _RepresentationTransformer()
