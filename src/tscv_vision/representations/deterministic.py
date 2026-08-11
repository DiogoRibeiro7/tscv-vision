"""Representation adapters over the existing deterministic encoders.

These are thin wrappers: the mathematics stays in
:mod:`tscv_vision.encoders`, and this module only adds the uniform
:class:`~tscv_vision.representations.base.Representation` interface, argument
validation, and the optional ``image_size`` resampling that lets an encoder
produce a fixed-size image from series of any length.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..encoders import ENCODER_REGISTRY, NanPolicy, get_encoder
from ..encoders import _validate_series as _validate
from .base import FloatArray, Representation
from .metadata import RepresentationInfo, ValidationLevel, get_encoder_metadata

__all__ = [
    "paa",
    "DeterministicRepresentation",
    "GAFRepresentation",
    "RecurrencePlotRepresentation",
    "SpectrogramRepresentation",
    "MTFRepresentation",
    "PersistenceImageRepresentation",
    "SAXRepresentation",
]


def paa(x: FloatArray, size: int) -> NDArray[np.float64]:
    """Piecewise Aggregate Approximation of ``x`` to ``size`` points.

    Splits the series into ``size`` contiguous segments and takes the mean of
    each (Keogh et al., 2001, *Dimensionality Reduction for Fast Similarity
    Search in Large Time Series Databases*, KAIS 3:263-286). This is how a
    fixed-size image is obtained from a variable-length series.

    Parameters
    ----------
    x:
        1D series.
    size:
        Number of output points; must be in ``[1, len(x)]``.

    Returns
    -------
    ndarray
        ``(size,)`` array of segment means. Returns ``x`` unchanged when
        ``size == len(x)``.

    Raises
    ------
    ValueError
        If ``size`` is outside ``[1, len(x)]``.

    Examples
    --------
    >>> paa(np.arange(8.0), 4)
    array([0.5, 2.5, 4.5, 6.5])
    """

    arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("x must be 1D")
    if size < 1 or size > arr.size:
        raise ValueError(f"size must be in [1, {arr.size}], got {size}")
    if size == arr.size:
        return arr
    return np.asarray(
        [segment.mean() for segment in np.array_split(arr, size)], dtype=np.float64
    )


class DeterministicRepresentation(Representation):
    """Adapter exposing any registered encoder as a :class:`Representation`.

    Parameters
    ----------
    name:
        Key in :data:`tscv_vision.encoders.ENCODER_REGISTRY`.
    image_size:
        Resample the series to this many points with :func:`paa` before
        encoding, so that square-image encoders produce a fixed
        ``(image_size, image_size)`` output regardless of input length.
        ``None`` (default) encodes the series as-is. Series shorter than
        ``image_size`` are rejected rather than up-sampled, because
        interpolating detail that is not there would silently fabricate
        structure the encoder then reports.
    nan_policy:
        Forwarded to the encoder when it accepts one.
    params:
        Extra keyword arguments forwarded to the encoder, e.g. ``bins=8``.

    Raises
    ------
    KeyError
        If ``name`` is not a registered encoder.
    ValueError
        If ``image_size`` is not positive.

    Examples
    --------
    >>> rep = DeterministicRepresentation("gaf", image_size=8)
    >>> rep.transform(np.sin(np.linspace(0, 6.0, 64))).shape
    (8, 8)
    >>> rep.info.canonical_method
    True
    """

    def __init__(
        self,
        name: str,
        *,
        image_size: int | None = None,
        nan_policy: NanPolicy = "raise",
        **params: Any,
    ) -> None:
        if name not in ENCODER_REGISTRY:
            raise KeyError(
                f"unknown encoder {name!r}; available: {sorted(ENCODER_REGISTRY)}"
            )
        if image_size is not None and image_size < 1:
            raise ValueError("image_size must be positive")
        self.name = name
        self.image_size = image_size
        self.nan_policy = nan_policy
        self.params = dict(params)

    # -- Representation ---------------------------------------------------

    def transform(self, x: FloatArray) -> FloatArray:
        """Encode ``x``, resampling first when ``image_size`` is set."""

        series = _validate(np.asarray(x, dtype=float), nan_policy=self.nan_policy)
        if self.image_size is not None:
            if series.size < self.image_size:
                raise ValueError(
                    f"series of length {series.size} is shorter than "
                    f"image_size={self.image_size}; up-sampling would invent "
                    "detail the series does not contain"
                )
            series = paa(series, self.image_size)
        encoder = get_encoder(self.name)
        kwargs = dict(self.params)
        if _accepts_nan_policy(self.name):
            kwargs.setdefault("nan_policy", "raise")  # already handled above
        out: FloatArray = encoder(series, **kwargs)
        return out

    @property
    def info(self) -> RepresentationInfo:
        """Metadata of the wrapped encoder, with the resolved output size."""

        base = get_encoder_metadata(self.name)
        if self.image_size is None or base.output_kind != "square_image":
            return base
        return base.replace(dimension=(self.image_size, self.image_size))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor configuration."""

        params: dict[str, Any] = {
            "name": self.name,
            "image_size": self.image_size,
            "nan_policy": self.nan_policy,
        }
        params.update(self.params)
        return params


def _accepts_nan_policy(name: str) -> bool:
    """Whether the registered encoder takes a ``nan_policy`` keyword."""

    import inspect

    try:
        signature = inspect.signature(get_encoder(name))
    except (TypeError, ValueError):  # pragma: no cover - builtins/partials
        return False
    return "nan_policy" in signature.parameters


class GAFRepresentation(DeterministicRepresentation):
    """Gramian Angular Field (Wang & Oates, 2015).

    Parameters
    ----------
    method:
        ``"summation"`` (GASF) or ``"difference"`` (GADF).
    image_size:
        See :class:`DeterministicRepresentation`.
    nan_policy:
        NaN handling policy.

    Examples
    --------
    >>> GAFRepresentation(method="difference", image_size=16).transform(
    ...     np.linspace(0.0, 1.0, 64)
    ... ).shape
    (16, 16)
    """

    def __init__(
        self,
        *,
        method: Literal["summation", "difference"] = "summation",
        image_size: int | None = None,
        nan_policy: NanPolicy = "raise",
    ) -> None:
        if method not in {"summation", "difference"}:
            raise ValueError("method must be 'summation' or 'difference'")
        super().__init__(
            "gaf" if method == "summation" else "gadf",
            image_size=image_size,
            nan_policy=nan_policy,
        )
        self.method = method

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor configuration."""
        return {
            "method": self.method,
            "image_size": self.image_size,
            "nan_policy": self.nan_policy,
        }


class RecurrencePlotRepresentation(DeterministicRepresentation):
    """Recurrence plot (Eckmann et al., 1987).

    Parameters
    ----------
    eps:
        Threshold in ``[0, 1]`` on min-max normalised distances; ``None``
        returns ``1 - normalised distance`` instead of a binary plot.
    metric:
        ``"euclidean"`` or ``"manhattan"`` (identical for univariate input).
    image_size, nan_policy:
        See :class:`DeterministicRepresentation`.
    """

    def __init__(
        self,
        *,
        eps: float | None = None,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        image_size: int | None = None,
        nan_policy: NanPolicy = "raise",
    ) -> None:
        super().__init__(
            "rp", image_size=image_size, nan_policy=nan_policy, eps=eps, metric=metric
        )
        self.eps = eps
        self.metric = metric

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor configuration."""
        return {
            "eps": self.eps,
            "metric": self.metric,
            "image_size": self.image_size,
            "nan_policy": self.nan_policy,
        }


class SpectrogramRepresentation(DeterministicRepresentation):
    """STFT magnitude spectrogram.

    ``image_size`` is not offered: resampling the series before an STFT would
    change the effective sample rate and therefore the meaning of every
    frequency bin. Control the output shape with ``win`` and ``hop`` instead.

    Parameters
    ----------
    win:
        Window length, ``>= 8``.
    hop:
        Hop length; defaults to ``win // 4``.
    window:
        ``"hann"`` or ``"rect"``.
    nan_policy:
        NaN handling policy.
    """

    def __init__(
        self,
        *,
        win: int = 64,
        hop: int | None = None,
        window: Literal["hann", "rect"] = "hann",
        nan_policy: NanPolicy = "raise",
    ) -> None:
        super().__init__(
            "spec", image_size=None, nan_policy=nan_policy, win=win, hop=hop, window=window
        )
        self.win = win
        self.hop = hop
        self.window = window

    @property
    def info(self) -> RepresentationInfo:
        """Metadata with the fixed number of frequency bins filled in."""
        return get_encoder_metadata("spec").replace(dimension=(self.win // 2 + 1, -1))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor configuration."""
        return {
            "win": self.win,
            "hop": self.hop,
            "window": self.window,
            "nan_policy": self.nan_policy,
        }


class MTFRepresentation(DeterministicRepresentation):
    """Markov Transition Field (Wang & Oates, 2015).

    Parameters
    ----------
    bins:
        Number of quantile states, ``>= 2``.
    weighted:
        Weight transitions by the absolute jump size.
    image_size, nan_policy:
        See :class:`DeterministicRepresentation`.
    """

    def __init__(
        self,
        *,
        bins: int = 8,
        weighted: bool = False,
        image_size: int | None = None,
        nan_policy: NanPolicy = "raise",
    ) -> None:
        super().__init__(
            "mtf", image_size=image_size, nan_policy=nan_policy, bins=bins, weighted=weighted
        )
        self.bins = bins
        self.weighted = weighted

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor configuration."""
        return {
            "bins": self.bins,
            "weighted": self.weighted,
            "image_size": self.image_size,
            "nan_policy": self.nan_policy,
        }


class PersistenceImageRepresentation(DeterministicRepresentation):
    """Persistence image of the 0D sublevel-set diagram (Adams et al., 2017).

    Parameters
    ----------
    bins:
        Pixels per axis.
    sigma:
        Gaussian bandwidth; defaults to one pixel of the persistence axis.
    weight:
        ``"persistence"``, ``"ramp"`` or ``"uniform"``.
    birth_range, pers_range:
        Fix the image extent so that images from different series are
        comparable. Left ``None`` the extent follows each series, which makes
        the representation series-relative — usable within one series,
        misleading across a dataset.
    nan_policy:
        NaN handling policy.
    """

    def __init__(
        self,
        *,
        bins: int = 32,
        sigma: float | None = None,
        weight: Literal["persistence", "ramp", "uniform"] = "persistence",
        birth_range: tuple[float, float] | None = None,
        pers_range: tuple[float, float] | None = None,
        nan_policy: NanPolicy = "raise",
    ) -> None:
        super().__init__(
            "ph",
            image_size=None,
            nan_policy=nan_policy,
            bins=bins,
            sigma=sigma,
            weight=weight,
            birth_range=birth_range,
            pers_range=pers_range,
        )
        self.bins = bins
        self.sigma = sigma
        self.weight = weight
        self.birth_range = birth_range
        self.pers_range = pers_range

    @property
    def info(self) -> RepresentationInfo:
        """Metadata with the fixed image size and comparability caveat."""
        base = get_encoder_metadata("ph").replace(dimension=(self.bins, self.bins))
        if self.birth_range is not None and self.pers_range is not None:
            return base.replace(
                notes="Explicit birth/persistence ranges: images are comparable across series."
            )
        return base

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor configuration."""
        return {
            "bins": self.bins,
            "sigma": self.sigma,
            "weight": self.weight,
            "birth_range": self.birth_range,
            "pers_range": self.pers_range,
            "nan_policy": self.nan_policy,
        }


class SAXRepresentation(DeterministicRepresentation):
    """SAX symbol-equality image (Lin et al., 2007, for the symbolisation).

    Parameters
    ----------
    segments:
        PAA segments, ``<= len(x)``.
    alphabet:
        Alphabet size, ``>= 2``.
    breakpoints:
        ``"gaussian"`` for standard SAX, ``"quantile"`` for the
        data-adaptive variant used before 0.2.0.
    nan_policy:
        NaN handling policy.
    """

    def __init__(
        self,
        *,
        segments: int = 8,
        alphabet: int = 8,
        breakpoints: Literal["gaussian", "quantile"] = "gaussian",
        nan_policy: NanPolicy = "raise",
    ) -> None:
        super().__init__(
            "sax",
            image_size=None,
            nan_policy=nan_policy,
            segments=segments,
            alphabet=alphabet,
            breakpoints=breakpoints,
        )
        self.segments = segments
        self.alphabet = alphabet
        self.breakpoints = breakpoints

    @property
    def info(self) -> RepresentationInfo:
        """Metadata with the fixed image size filled in."""
        base = get_encoder_metadata("sax").replace(dimension=(self.segments, self.segments))
        if self.breakpoints == "quantile":
            return base.replace(
                validation_level=ValidationLevel.INVARIANT,
                validated_by=(
                    "tests/test_encoder_definitions.py"
                    "::test_sax_quantile_variant_is_not_the_standard_one",
                ),
                notes=(
                    "Non-standard quantile breakpoints: symbols are rescaled to "
                    "the segment means of each series and are not comparable "
                    "across series."
                ),
            )
        return base

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor configuration."""
        return {
            "segments": self.segments,
            "alphabet": self.alphabet,
            "breakpoints": self.breakpoints,
            "nan_policy": self.nan_policy,
        }


#: Convenience classes keyed by the encoder name they wrap. Encoders absent
#: from this mapping are still reachable through
#: :class:`DeterministicRepresentation`.
NAMED_REPRESENTATIONS: dict[str, type[DeterministicRepresentation]] = {
    "gaf": GAFRepresentation,
    "rp": RecurrencePlotRepresentation,
    "spec": SpectrogramRepresentation,
    "mtf": MTFRepresentation,
    "ph": PersistenceImageRepresentation,
    "sax": SAXRepresentation,
}


def _default_params(name: str) -> dict[str, Any]:
    """Encoder arguments that have no usable default in the registry."""

    if name == "cwt":
        # `cwt` requires `scales`; the registry stores the bare function, so a
        # generic call would fail. Use dyadic scales, as is conventional.
        return {"scales": np.asarray([2**k for k in range(5)], dtype=float)}
    if name == "mp":
        return {"m": 8}
    if name == "msrp":
        return {"scales": np.asarray([1, 2, 4])}
    if name == "shapelet":
        return {"k": 3, "seed": 0}
    return {}


def build_deterministic(name: str, **kwargs: Any) -> DeterministicRepresentation:
    """Instantiate the best adapter for encoder ``name``.

    Uses the dedicated class when one exists, otherwise the generic
    :class:`DeterministicRepresentation` with any arguments the encoder needs
    but the registry cannot supply (``cwt`` scales, ``mp`` subsequence length).

    Raises
    ------
    KeyError
        If ``name`` is not a registered encoder.
    """

    cls = NAMED_REPRESENTATIONS.get(name)
    if cls is not None:
        return cls(**kwargs)
    params = _default_params(name)
    params.update(kwargs)
    return DeterministicRepresentation(name, **params)


def suggest_image_size(length: int, *, maximum: int = 128) -> int:
    """Return a sensible ``image_size`` for a series of ``length`` points.

    Square-image encoders cost ``O(size^2)`` memory, so encoding a long series
    at full length is usually a mistake. This caps the size at ``maximum`` and
    otherwise keeps the series intact.

    Raises
    ------
    ValueError
        If ``length`` or ``maximum`` is not positive.
    """

    if length < 1 or maximum < 1:
        raise ValueError("length and maximum must be positive")
    return int(min(length, maximum, 2 ** int(math.floor(math.log2(max(length, 1)))) or 1))
