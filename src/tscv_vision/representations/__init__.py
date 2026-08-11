"""Unified representation API.

One interface over every way this package turns a time series into something a
model can consume — deterministic encoders today, learned and pretrained
models later — plus a registry that can be queried by scientific provenance
rather than by name alone.

    >>> import numpy as np
    >>> from tscv_vision.representations import get_representation, list_representations
    >>> rep = get_representation("gaf", image_size=16)
    >>> rep.transform(np.sin(np.linspace(0, 6.0, 128))).shape
    (16, 16)
    >>> list_representations(family="gramian")
    ['gadf', 'gaf', 'gdf']

Every representation carries a :class:`RepresentationInfo` recording its
family, its published reference (or the explicit absence of one), whether it
reproduces a canonical method, and how thoroughly it has been validated:

    >>> get_representation_info("eph").canonical_method
    False
    >>> get_representation_info("mp").validation_level.label
    'LEVEL 3 — reference'

Filtering on that metadata is the point — it is how you assemble a defensible
experiment rather than an arbitrary one:

    >>> list_representations(canonical_method=True, min_validation_level=3)
    ['gadf', 'gaf', 'mp', 'mtf', 'mtspec', 'ph']

scikit-learn is not required. Wrap a representation with
:func:`~tscv_vision.representations.base.as_sklearn` when a transformer is
needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import (
    FittedRepresentation,
    FloatArray,
    NotFittedError,
    PretrainedRepresentation,
    Representation,
    as_sklearn,
)
from .deterministic import (
    DeterministicRepresentation,
    GAFRepresentation,
    MTFRepresentation,
    PersistenceImageRepresentation,
    RecurrencePlotRepresentation,
    SAXRepresentation,
    SpectrogramRepresentation,
    build_deterministic,
    paa,
    suggest_image_size,
)
from .fusion import ConcatFusion, FusionRepresentation, LearnedFusion
from .learned import LearnedRepresentation
from .metadata import (
    ENCODER_ALIASES,
    ENCODER_METADATA,
    MULTIVARIATE_METADATA,
    InputKind,
    OutputKind,
    RepresentationInfo,
    ValidationLevel,
    get_encoder_metadata,
    list_encoders,
    validation_matrix_markdown,
    validation_matrix_rows,
)
from .pretrained import PretrainedBackbone, resolve_device

__all__ = [
    # Interfaces
    "Representation",
    "FittedRepresentation",
    "PretrainedRepresentation",
    "LearnedRepresentation",
    "PretrainedBackbone",
    "FusionRepresentation",
    "LearnedFusion",
    "NotFittedError",
    "FloatArray",
    # Deterministic adapters
    "DeterministicRepresentation",
    "GAFRepresentation",
    "RecurrencePlotRepresentation",
    "SpectrogramRepresentation",
    "MTFRepresentation",
    "PersistenceImageRepresentation",
    "SAXRepresentation",
    "ConcatFusion",
    "paa",
    "suggest_image_size",
    # Metadata
    "RepresentationInfo",
    "ValidationLevel",
    "InputKind",
    "OutputKind",
    "ENCODER_METADATA",
    "MULTIVARIATE_METADATA",
    "ENCODER_ALIASES",
    "get_encoder_metadata",
    "list_encoders",
    "validation_matrix_rows",
    "validation_matrix_markdown",
    # Registry
    "REPRESENTATION_REGISTRY",
    "register_representation",
    "list_representations",
    "get_representation",
    "get_representation_info",
    "as_sklearn",
    "resolve_device",
]

#: Factory per representation name. Values take keyword arguments and return a
#: ready-to-use :class:`Representation`.
RepresentationFactory = Callable[..., Representation]

REPRESENTATION_REGISTRY: dict[str, RepresentationFactory] = {}

#: Metadata per representation name. Populated alongside the registry.
REPRESENTATION_INFO: dict[str, RepresentationInfo] = {}


def register_representation(
    name: str,
    factory: RepresentationFactory,
    info: RepresentationInfo,
    *,
    overwrite: bool = False,
) -> None:
    """Register ``factory`` under ``name`` with its metadata.

    Parameters
    ----------
    name:
        Registry key.
    factory:
        Callable returning a :class:`Representation`; keyword arguments passed
        to :func:`get_representation` are forwarded to it.
    info:
        Provenance and validation metadata. Registering without it is not
        possible by design — an unlabelled representation is one nobody can
        judge.
    overwrite:
        Allow replacing an existing entry.

    Raises
    ------
    ValueError
        If ``name`` is already registered and ``overwrite`` is ``False``, or
        if ``info.name`` does not match ``name``.
    """

    if name in REPRESENTATION_REGISTRY and not overwrite:
        raise ValueError(
            f"{name!r} is already registered; pass overwrite=True to replace it"
        )
    if info.name != name:
        raise ValueError(f"info.name is {info.name!r} but registering as {name!r}")
    REPRESENTATION_REGISTRY[name] = factory
    REPRESENTATION_INFO[name] = info


def get_representation(name: str, **kwargs: Any) -> Representation:
    """Instantiate the representation registered as ``name``.

    Parameters
    ----------
    name:
        Registry key, e.g. ``"gaf"``.
    kwargs:
        Forwarded to the factory, e.g. ``image_size=32`` or ``bins=8``.

    Returns
    -------
    Representation
        A new instance.

    Raises
    ------
    KeyError
        If ``name`` is not registered.

    Examples
    --------
    >>> get_representation("mtf", bins=4).info.family
    'markov'
    """

    try:
        factory = REPRESENTATION_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown representation {name!r}; available: "
            f"{sorted(REPRESENTATION_REGISTRY)}"
        ) from None
    return factory(**kwargs)


def get_representation_info(name: str) -> RepresentationInfo:
    """Return the metadata registered for ``name``.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """

    try:
        return REPRESENTATION_INFO[name]
    except KeyError:
        raise KeyError(
            f"unknown representation {name!r}; available: {sorted(REPRESENTATION_INFO)}"
        ) from None


def list_representations(
    *,
    family: str | None = None,
    input_kind: InputKind | None = None,
    output_kind: OutputKind | None = None,
    trainable: bool | None = None,
    pretrained: bool | None = None,
    deterministic: bool | None = None,
    canonical_method: bool | None = None,
    min_validation_level: ValidationLevel | int | None = None,
    include_aliases: bool = False,
) -> list[str]:
    """Return registered names matching every supplied filter, sorted.

    Parameters
    ----------
    family, input_kind, output_kind, trainable, pretrained, deterministic,
    canonical_method:
        Exact-match filters on :class:`RepresentationInfo`; ``None`` disables
        the filter.
    min_validation_level:
        Keep only representations validated at least this thoroughly.
    include_aliases:
        Include alias keys such as ``"tpa"``.

    Examples
    --------
    >>> list_representations(family="time_frequency", trainable=False)
    ['cwt', 'mtspec', 'spec', 'sst']
    >>> list_representations(min_validation_level=ValidationLevel.REFERENCE)
    ['gadf', 'gaf', 'mp', 'mtf', 'mtspec', 'ph', 'sax']
    """

    names = []
    for name, info in REPRESENTATION_INFO.items():
        if not include_aliases and name in ENCODER_ALIASES:
            continue
        if family is not None and info.family != family:
            continue
        if input_kind is not None and info.input_kind != input_kind:
            continue
        if output_kind is not None and info.output_kind != output_kind:
            continue
        if trainable is not None and info.trainable is not trainable:
            continue
        if pretrained is not None and info.pretrained is not pretrained:
            continue
        if deterministic is not None and info.deterministic is not deterministic:
            continue
        if canonical_method is not None and info.canonical_method is not canonical_method:
            continue
        if (
            min_validation_level is not None
            and info.validation_level < ValidationLevel(min_validation_level)
        ):
            continue
        names.append(name)
    return sorted(names)


def _register_builtin_encoders() -> None:
    """Expose every encoder with metadata as a registered representation."""

    def _make(encoder_name: str) -> RepresentationFactory:
        def factory(**kwargs: Any) -> Representation:
            return build_deterministic(encoder_name, **kwargs)

        factory.__name__ = f"build_{encoder_name}_representation"
        factory.__doc__ = f"Build a representation wrapping the {encoder_name!r} encoder."
        return factory

    for encoder_name, info in ENCODER_METADATA.items():
        register_representation(encoder_name, _make(encoder_name), info, overwrite=True)


_register_builtin_encoders()
