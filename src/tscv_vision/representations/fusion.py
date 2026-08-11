"""Combining several representations of the same series.

Only the deterministic case lives here: concatenating or averaging views whose
outputs are already commensurate. Learned or weighted fusion is a fitted
representation — the weights are parameters, and choosing them on anything but
training data leaks — so it belongs on :class:`LearnedFusion`, which is left
abstract for now.

The feature-vector reduction reuses :func:`tscv_vision.fusion.fuse` rather
than reimplementing it.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from ..fusion import fuse as _fuse
from .base import FittedRepresentation, FloatArray, Representation
from .metadata import RepresentationInfo, ValidationLevel

__all__ = ["FusionRepresentation", "ConcatFusion", "LearnedFusion"]


class FusionRepresentation(Representation):
    """Base for representations that combine several views of one input.

    Parameters
    ----------
    views:
        The representations to combine; at least one.

    Raises
    ------
    ValueError
        If ``views`` is empty.
    """

    def __init__(self, views: Sequence[Representation]) -> None:
        if not views:
            raise ValueError("at least one view is required")
        self.views = list(views)

    @property
    def view_names(self) -> list[str]:
        """Names of the combined views, in order."""

        return [view.info.name for view in self.views]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor configuration."""

        return {"views": self.views}


class ConcatFusion(FusionRepresentation):
    """Flatten each view and combine the results.

    Parameters
    ----------
    views:
        Representations to combine.
    mode:
        Reduction applied by :func:`tscv_vision.fusion.fuse`: ``"concat"``
        keeps every view, the others require the views to produce equal-length
        vectors.
    weights:
        Per-view weights for ``mode="weighted"``.

    Raises
    ------
    ValueError
        If ``views`` is empty, or the modes and weights disagree.

    Examples
    --------
    >>> from tscv_vision.representations import get_representation
    >>> fusion = ConcatFusion(
    ...     [get_representation("gaf", image_size=8), get_representation("mtf", image_size=8)]
    ... )
    >>> fusion.transform(np.sin(np.linspace(0, 6.0, 64))).shape
    (128,)
    """

    def __init__(
        self,
        views: Sequence[Representation],
        *,
        mode: Literal["concat", "mean", "median", "weighted"] = "concat",
        weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__(views)
        if mode == "weighted" and weights is None:
            raise ValueError("mode='weighted' requires weights")
        if weights is not None and len(weights) != len(self.views):
            raise ValueError("weights must have one entry per view")
        self.mode = mode
        self.weights = list(weights) if weights is not None else None

    def transform(self, x: FloatArray) -> FloatArray:
        """Transform ``x`` with every view and reduce the flattened outputs."""

        parts = [
            np.asarray(np.ravel(view.transform(x)), dtype=np.float64)
            for view in self.views
        ]
        out: FloatArray = _fuse(parts, mode=self.mode, weights=self.weights)
        return out

    @property
    def info(self) -> RepresentationInfo:
        """Combined metadata; canonical only if every view is."""

        infos = [view.info for view in self.views]
        return RepresentationInfo(
            name=f"concat({'+'.join(self.view_names)})",
            family="fusion",
            input_kind=infos[0].input_kind,
            output_kind="embedding",
            trainable=any(i.trainable for i in infos),
            pretrained=any(i.pretrained for i in infos),
            deterministic=all(i.deterministic for i in infos),
            differentiable=all(i.differentiable for i in infos),
            dimension=None,
            canonical_method=False,
            reference=None,
            complexity="sum of the constituent views",
            validation_level=min(
                (i.validation_level for i in infos), default=ValidationLevel.SMOKE
            ),
            validated_by=("tests/test_representations.py::test_concat_fusion_shapes",),
            notes=(
                "Deterministic concatenation of "
                f"{', '.join(self.view_names)}. Validation level is the weakest "
                "of the combined views."
            ),
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor configuration."""

        return {"views": self.views, "mode": self.mode, "weights": self.weights}


class LearnedFusion(FittedRepresentation, FusionRepresentation):
    """Base for fusion whose combination weights are learned.

    Left abstract deliberately. Fusion weights are model parameters: selecting
    them on validation or test data leaks exactly as any other supervised
    choice would, so an implementation must estimate them inside ``fit`` and
    be usable inside a cross-validation fold.

    Subclasses implement :meth:`_fit`, :meth:`transform` and :attr:`info`.
    """

    def __init__(self, views: Sequence[Representation]) -> None:
        FusionRepresentation.__init__(self, views)

    @abstractmethod
    def _fit(self, X: Sequence[FloatArray], y: Any | None = None) -> None:
        """Estimate the fusion weights from training data only."""
