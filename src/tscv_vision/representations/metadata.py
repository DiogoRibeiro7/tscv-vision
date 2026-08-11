"""Scientific provenance and validation status for every representation.

Docstring prose is not a machine-checkable record of what a transformation is
or how well it has been validated, so that record lives here instead: one
:class:`RepresentationInfo` per registered representation, carrying its
family, its published reference (or the explicit absence of one), and the
level of validation it has actually reached.

``canonical_method`` is the load-bearing field. It is ``True`` only when the
implementation reproduces a published method and a test pins it to that
method — never merely because the name sounds established.

Validation levels
-----------------

============  =========================================================
Level         Meaning
============  =========================================================
``SMOKE``     Runs and returns the documented shape.
``INVARIANT`` Mathematical invariants are asserted (symmetry, bounds,
              normalisation, equivariance).
``SYNTHETIC`` Compared against the published formula re-implemented
              directly, or against an analytically known answer.
``REFERENCE`` Compared numerically against an independent third-party
              implementation.
``BENCHMARK`` Additionally evaluated on real datasets in a committed
              benchmark run.
============  =========================================================

Core encoders target ``REFERENCE``. Inclusion in publication experiments
requires ``BENCHMARK``. A level is a claim about tests that exist in this
repository; :mod:`tests.test_representations` checks that each level is
backed by a named test.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any, Literal

__all__ = [
    "ValidationLevel",
    "InputKind",
    "OutputKind",
    "RepresentationInfo",
    "ENCODER_METADATA",
    "get_encoder_metadata",
    "list_encoders",
    "validation_matrix_rows",
    "validation_matrix_markdown",
]


class ValidationLevel(IntEnum):
    """How thoroughly a representation has been validated in this repository."""

    SMOKE = 0
    INVARIANT = 1
    SYNTHETIC = 2
    REFERENCE = 3
    BENCHMARK = 4

    @property
    def label(self) -> str:
        """Human-readable ``LEVEL n — name`` label used in documentation."""
        return f"LEVEL {int(self)} — {self.name.lower()}"


InputKind = Literal[
    "univariate",
    "bivariate",
    "multivariate",
    "image",
    "tokens",
    "embedding",
]

OutputKind = Literal[
    "square_image",
    "time_frequency",
    "rectangular_image",
    "embedding",
    "tokens",
    "tensor",
]


@dataclass(frozen=True)
class RepresentationInfo:
    """Machine-readable description of one representation.

    Parameters
    ----------
    name:
        Registry key.
    family:
        Grouping used for filtering, e.g. ``"gramian"`` or ``"topological"``.
    input_kind, output_kind:
        Shape contract, see :data:`InputKind` and :data:`OutputKind`.
    trainable:
        Whether the representation has parameters fitted from data.
    pretrained:
        Whether it wraps externally pretrained weights.
    deterministic:
        Whether repeated calls on the same input give bitwise-identical
        output given the same seed.
    differentiable:
        Whether gradients can flow through it (a NumPy encoder cannot).
    dimension:
        Output shape when fixed, ``None`` when it depends entirely on the
        input. A ``-1`` inside the tuple marks a single input-dependent axis,
        as for a spectrogram whose frequency axis is fixed by ``win`` but
        whose time axis grows with the series.
    canonical_method:
        ``True`` only if this reproduces the referenced published method and
        a test pins it to that method.
    reference:
        Citation for the method, or ``None`` for project-defined transforms.
    complexity:
        Time complexity in the series length ``N``.
    validation_level:
        See :class:`ValidationLevel`.
    validated_by:
        Test node ids or files backing ``validation_level``.
    notes:
        Anything a user must know that the fields above cannot express —
        in particular, how a project-defined variant differs from the
        published method whose name it resembles.
    """

    name: str
    family: str
    input_kind: InputKind
    output_kind: OutputKind
    trainable: bool = False
    pretrained: bool = False
    deterministic: bool = True
    differentiable: bool = False
    dimension: int | tuple[int, ...] | None = None
    canonical_method: bool = False
    reference: str | None = None
    complexity: str = "unspecified"
    validation_level: ValidationLevel = ValidationLevel.SMOKE
    validated_by: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.canonical_method and not self.reference:
            raise ValueError(
                f"{self.name}: canonical_method=True requires a reference; a "
                "method cannot be canonical without the paper it reproduces"
            )
        if self.validation_level >= ValidationLevel.INVARIANT and not self.validated_by:
            raise ValueError(
                f"{self.name}: validation_level {self.validation_level.label} "
                "requires validated_by to name the tests that back it"
            )

    def replace(self, **changes: Any) -> RepresentationInfo:
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of every field."""
        return {
            "name": self.name,
            "family": self.family,
            "input_kind": self.input_kind,
            "output_kind": self.output_kind,
            "trainable": self.trainable,
            "pretrained": self.pretrained,
            "deterministic": self.deterministic,
            "differentiable": self.differentiable,
            "dimension": self.dimension,
            "canonical_method": self.canonical_method,
            "reference": self.reference,
            "complexity": self.complexity,
            "validation_level": int(self.validation_level),
            "validated_by": list(self.validated_by),
            "notes": self.notes,
        }


_DEF = "tests/test_encoder_definitions.py"
_REF = "tests/test_reference_equivalence.py"
_NEW = "tests/test_new_encoders.py"
_PROP = "tests/test_encoder_properties.py"
_REG = "tests/test_regression_outputs.py"
_SST = "tests/test_synchrosqueezed.py"
_HVG = "tests/test_horizontal_visibility.py"


def _info(**kwargs: Any) -> RepresentationInfo:
    return RepresentationInfo(**kwargs)


#: Metadata for every encoder registered in
#: :data:`tscv_vision.encoders.ENCODER_REGISTRY`. Aliases share the entry of
#: the function they resolve to.
ENCODER_METADATA: dict[str, RepresentationInfo] = {
    "gaf": _info(
        name="gaf",
        family="gramian",
        input_kind="univariate",
        output_kind="square_image",
        dimension=None,
        canonical_method=True,
        reference=(
            "Wang & Oates (2015), Imaging Time-Series to Improve "
            "Classification and Imputation, IJCAI"
        ),
        complexity="O(N^2) time and memory",
        validation_level=ValidationLevel.REFERENCE,
        validated_by=(f"{_REF}::test_gaf_matches_pyts", f"{_DEF}::test_gasf_matches_definition"),
    ),
    "gadf": _info(
        name="gadf",
        family="gramian",
        input_kind="univariate",
        output_kind="square_image",
        canonical_method=True,
        reference="Wang & Oates (2015), IJCAI",
        complexity="O(N^2) time and memory",
        validation_level=ValidationLevel.REFERENCE,
        validated_by=(f"{_REF}::test_gaf_matches_pyts", f"{_DEF}::test_gadf_matches_definition"),
    ),
    "rp": _info(
        name="rp",
        family="recurrence",
        input_kind="univariate",
        output_kind="square_image",
        canonical_method=True,
        reference=(
            "Eckmann, Kamphorst & Ruelle (1987), Recurrence Plots of "
            "Dynamical Systems, Europhysics Letters 4:973-977"
        ),
        complexity="O(N^2) time and memory",
        validation_level=ValidationLevel.SYNTHETIC,
        validated_by=(
            f"{_DEF}::test_recurrence_plot_matches_definition",
            f"{_PROP}::test_recurrence_plot_symmetric_diag",
        ),
        notes=(
            "Distances are min-max normalised before thresholding, so eps is a "
            "fraction of the largest pairwise distance rather than an absolute "
            "radius in the original units."
        ),
    ),
    "spec": _info(
        name="spec",
        family="time_frequency",
        input_kind="univariate",
        output_kind="time_frequency",
        canonical_method=True,
        reference="Standard short-time Fourier transform magnitude",
        complexity="O((N/hop) * win log win)",
        validation_level=ValidationLevel.SYNTHETIC,
        validated_by=(
            f"{_DEF}::test_spectrogram_matches_framed_rfft",
            f"{_DEF}::test_spectrogram_locates_a_pure_tone",
        ),
        notes="Magnitudes are normalised by the global maximum, so the output is scale-invariant.",
    ),
    "cwt": _info(
        name="cwt",
        family="time_frequency",
        input_kind="univariate",
        output_kind="time_frequency",
        canonical_method=False,
        reference=(
            "Torrence & Compo (1998), A Practical Guide to Wavelet "
            "Analysis, BAMS 79:61-78"
        ),
        complexity="O(S * N log N)",
        validation_level=ValidationLevel.SMOKE,
        validated_by=(),
        notes=(
            "Only shape-tested. The Morlet path is a bespoke FFT implementation "
            "whose normalisation has not been compared against Torrence & Compo "
            "or PyWavelets; non-Morlet wavelets delegate to PyWavelets. Raising "
            "this to REFERENCE is a v0.3.0 item."
        ),
    ),
    "sst": _info(
        name="sst",
        family="time_frequency",
        input_kind="univariate",
        output_kind="time_frequency",
        canonical_method=True,
        reference=(
            "Daubechies, Lu & Wu (2011), Synchrosqueezed wavelet transforms, "
            "ACHA 30(2):243-261"
        ),
        complexity="O(S * N log N) time, O(S * N) memory",
        validation_level=ValidationLevel.SYNTHETIC,
        validated_by=(
            f"{_SST}::test_constant_sinusoid_ridge_is_the_true_frequency",
            f"{_SST}::test_linear_chirp_ridge_follows_the_analytic_law",
            f"{_SST}::test_two_components_are_resolved",
        ),
        notes=(
            "Reassigns CWT energy to the instantaneous frequency estimated from "
            "the phase derivative, so ridges are far sharper than `cwt` or "
            "`spec`. Magnitude output discards phase; reassignment is quantised "
            "to the frequency grid."
        ),
    ),
    "mtf": _info(
        name="mtf",
        family="markov",
        input_kind="univariate",
        output_kind="square_image",
        canonical_method=True,
        reference="Wang & Oates (2015), IJCAI",
        complexity="O(N^2) time and memory",
        validation_level=ValidationLevel.REFERENCE,
        validated_by=(f"{_REF}::test_mtf_matches_pyts", f"{_DEF}::test_mtf_matches_definition"),
    ),
    "ph": _info(
        name="ph",
        family="topological",
        input_kind="univariate",
        output_kind="square_image",
        canonical_method=True,
        reference=(
            "Adams et al. (2017), Persistence Images: A Stable Vector "
            "Representation of Persistent Homology, JMLR 18(8):1-35"
        ),
        complexity="O(N log N) for the diagram, O(P * bins^2) for the image",
        validation_level=ValidationLevel.REFERENCE,
        validated_by=(
            f"{_REF}::test_persistence_image_matches_persim",
            f"{_REF}::test_persistence_diagram_matches_ripser",
            f"{_DEF}::test_persistence_diagram_matches_brute_force",
        ),
        notes=(
            "Birth and persistence ranges default to the diagram's own extent, "
            "making images series-relative; pass explicit ranges to compare "
            "across series."
        ),
    ),
    "eph": _info(
        name="eph",
        family="heuristic",
        input_kind="univariate",
        output_kind="square_image",
        canonical_method=False,
        reference=None,
        complexity="O(N + P * bins)",
        validation_level=ValidationLevel.INVARIANT,
        validated_by=(f"{_NEW}::test_extrema_persistence_histogram_keeps_old_behaviour",),
        notes=(
            "Project-defined. Pairs consecutive extrema and histograms the "
            "(value, amplitude) pairs. It does not compute persistent homology "
            "and is not a persistence image; it carried that name before 0.2.0."
        ),
    ),
    "gdf": _info(
        name="gdf",
        family="gramian",
        input_kind="univariate",
        output_kind="square_image",
        canonical_method=False,
        reference=None,
        complexity="O(N^2) time and memory",
        validation_level=ValidationLevel.INVARIANT,
        validated_by=(f"{_NEW}::test_gdf_range",),
        notes=(
            "Project-defined pairwise-difference matrix on a min-max scaled "
            "series. Despite the name it is not the Gramian Angular Difference "
            "Field, which is `gadf`."
        ),
    ),
    "msrp": _info(
        name="msrp",
        family="recurrence",
        input_kind="univariate",
        output_kind="tensor",
        canonical_method=False,
        reference=None,
        complexity="O(S * N^2)",
        validation_level=ValidationLevel.INVARIANT,
        validated_by=(f"{_NEW}::test_multi_scale_rp_stack",),
        notes=(
            "Project-defined: recurrence plots of decimated copies, upsampled by "
            "pixel repetition and stacked. Multi-scale recurrence analysis exists "
            "in the literature but this particular construction is ours."
        ),
    ),
    "dtw": _info(
        name="dtw",
        family="elastic",
        input_kind="univariate",
        output_kind="square_image",
        canonical_method=True,
        reference=(
            "Sakoe & Chiba (1978), Dynamic Programming Algorithm "
            "Optimization for Spoken Word Recognition, IEEE TASSP 26:43-49"
        ),
        complexity="O(N^2)",
        validation_level=ValidationLevel.SYNTHETIC,
        validated_by=(f"{_DEF}::test_dtw_matrix_matches_definition",),
        notes=(
            "Self-DTW accumulated-cost matrix, normalised and inverted. It is the "
            "cost surface, not a warping path or a distance between two series."
        ),
    ),
    "sax": _info(
        name="sax",
        family="symbolic",
        input_kind="univariate",
        output_kind="square_image",
        canonical_method=False,
        reference=(
            "Lin, Keogh, Wei & Lonardi (2007), Experiencing SAX, "
            "DMKD 15:107-144"
        ),
        complexity="O(N + segments^2)",
        validation_level=ValidationLevel.REFERENCE,
        validated_by=(
            f"{_REF}::test_sax_breakpoints_match_scipy_norm_ppf",
            f"{_DEF}::test_sax_gaussian_breakpoints_match_published_table",
            f"{_DEF}::test_sax_gaussian_is_invariant_to_affine_rescaling",
        ),
        notes=(
            "The symbolisation (`sax_symbols`) is canonical SAX and is "
            "reference-tested. The image itself — a symbol-equality matrix — is "
            "a project-defined recurrence plot in symbol space."
        ),
    ),
    "msc": _info(
        name="msc",
        family="filterbank",
        input_kind="univariate",
        output_kind="rectangular_image",
        canonical_method=False,
        reference=None,
        complexity="O(K * N)",
        validation_level=ValidationLevel.INVARIANT,
        validated_by=(f"{_NEW}::test_multi_scale_conv_stack",),
        notes="Project-defined stack of moving-average responses at several kernel widths.",
    ),
    "attn": _info(
        name="attn",
        family="attention",
        input_kind="univariate",
        output_kind="square_image",
        canonical_method=False,
        reference=(
            "Vaswani et al. (2017), Attention Is All You Need, NeurIPS "
            "(scaled dot-product attention)"
        ),
        complexity="O(W^2 * window) for W = N - window + 1",
        validation_level=ValidationLevel.SYNTHETIC,
        validated_by=(f"{_DEF}::test_window_attention_matches_softmax_definition",),
        notes=(
            "Parameter-free self-attention between sliding windows. It is not "
            "Temporal Pattern Attention (Shih, Sun & Lee, 2019), which it was "
            "incorrectly named after before 0.2.0."
        ),
    ),
    "vg": _info(
        name="vg",
        family="graph",
        input_kind="univariate",
        output_kind="square_image",
        canonical_method=True,
        reference=(
            "Lacasa et al. (2008), From time series to complex networks: "
            "The visibility graph, PNAS 105:4972-4975"
        ),
        complexity="O(N^2)",
        validation_level=ValidationLevel.SYNTHETIC,
        validated_by=(
            f"{_DEF}::test_visibility_graph_matches_definition",
            f"{_NEW}::test_visibility_graph_symmetry",
        ),
    ),
    "hvg": _info(
        name="hvg",
        family="graph",
        input_kind="univariate",
        output_kind="square_image",
        canonical_method=True,
        reference=(
            "Luque, Lacasa, Ballesteros & Luque (2009), Horizontal visibility "
            "graphs: exact results for random time series, Phys. Rev. E 80:046103"
        ),
        complexity="O(N) time for the edges, O(N^2) memory for the dense matrix",
        validation_level=ValidationLevel.SYNTHETIC,
        validated_by=(
            f"{_HVG}::test_matches_the_definition_by_brute_force",
            f"{_HVG}::test_hand_computed_examples",
            f"{_HVG}::test_invariant_under_monotonic_transformation",
        ),
        notes=(
            "Distinct from `vg`: the horizontal criterion is order-based, so "
            "the HVG is a subgraph of the natural visibility graph and is "
            "invariant under any strictly increasing transformation of the "
            "values. The amplitude and distance weightings are TSCV-Vision "
            "extensions, not part of the published definition."
        ),
    ),
    "shapelet": _info(
        name="shapelet",
        family="subsequence",
        input_kind="univariate",
        output_kind="rectangular_image",
        deterministic=True,
        canonical_method=False,
        reference=None,
        complexity="O(k * N * length)",
        validation_level=ValidationLevel.INVARIANT,
        validated_by=(f"{_NEW}::test_shapelet_transform_basic",),
        notes=(
            "Distances to randomly sampled subsequences of the same series. It is "
            "not the Shapelet Transform of Hills et al. (2014), which searches "
            "for discriminative shapelets against class labels. Deterministic "
            "given `seed`."
        ),
    ),
    "mp": _info(
        name="mp",
        family="subsequence",
        input_kind="univariate",
        output_kind="rectangular_image",
        canonical_method=True,
        reference="Yeh et al. (2016), Matrix Profile I, ICDM",
        complexity="O(N^2 * m) time, O(N^2) memory",
        validation_level=ValidationLevel.REFERENCE,
        validated_by=(
            f"{_REF}::test_matrix_profile_matches_stumpy",
            f"{_DEF}::test_matrix_profile_matches_brute_force",
        ),
        notes=(
            "Brute-force pairwise distances, not STOMP or SCRIMP++. Correct but "
            "quadratic in memory; unsuitable for very long series."
        ),
    ),
    "randproj": _info(
        name="randproj",
        family="projection",
        input_kind="univariate",
        output_kind="square_image",
        canonical_method=False,
        reference=None,
        complexity="O(size^2 * N)",
        validation_level=ValidationLevel.INVARIANT,
        validated_by=(f"{_NEW}::test_randproj_deterministic",),
        notes="Project-defined. Deterministic given `seed`; the projection is not normalised.",
    ),
    "ensemble": _info(
        name="ensemble",
        family="meta",
        input_kind="univariate",
        output_kind="tensor",
        canonical_method=False,
        reference=None,
        complexity="sum of the constituent encoders",
        validation_level=ValidationLevel.INVARIANT,
        validated_by=(f"{_NEW}::test_ensemble_stack", f"{_NEW}::test_ensemble_mean"),
        notes="Meta-encoder: stacks or averages other encoders that share an output shape.",
    ),
}

#: Registry aliases that resolve to the same implementation as another key.
ENCODER_ALIASES: dict[str, str] = {
    "visibility_graph": "vg",
    "matrix_profile": "mp",
    "persistence_image": "ph",
    "window_attention": "attn",
    "tpa": "attn",
}

for _alias, _target in ENCODER_ALIASES.items():
    ENCODER_METADATA[_alias] = ENCODER_METADATA[_target].replace(name=_alias)


def get_encoder_metadata(name: str) -> RepresentationInfo:
    """Return the metadata for encoder ``name``.

    Raises
    ------
    KeyError
        If ``name`` has no metadata entry.
    """

    try:
        return ENCODER_METADATA[name]
    except KeyError:
        raise KeyError(
            f"no metadata for encoder {name!r}; register it in "
            "tscv_vision.representations.metadata.ENCODER_METADATA so its "
            "provenance and validation level are recorded"
        ) from None


def list_encoders(
    *,
    family: str | None = None,
    input_kind: InputKind | None = None,
    output_kind: OutputKind | None = None,
    canonical_method: bool | None = None,
    min_validation_level: ValidationLevel | int | None = None,
    include_aliases: bool = False,
) -> list[str]:
    """Return encoder names matching every supplied filter, sorted.

    Parameters
    ----------
    family, input_kind, output_kind, canonical_method:
        Exact-match filters on the corresponding :class:`RepresentationInfo`
        field; ``None`` disables the filter.
    min_validation_level:
        Keep only encoders validated at least this thoroughly.
    include_aliases:
        Include alias keys such as ``"tpa"``. Off by default so the result is
        one entry per implementation.

    Examples
    --------
    >>> list_encoders(family="gramian")
    ['gadf', 'gaf', 'gdf']
    >>> list_encoders(canonical_method=True, min_validation_level=ValidationLevel.REFERENCE)
    ['gadf', 'gaf', 'mp', 'mtf', 'ph']
    """

    names = []
    for name, info in ENCODER_METADATA.items():
        if not include_aliases and name in ENCODER_ALIASES:
            continue
        if family is not None and info.family != family:
            continue
        if input_kind is not None and info.input_kind != input_kind:
            continue
        if output_kind is not None and info.output_kind != output_kind:
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


#: One-line description of each validation level, used in the generated matrix.
_LEVEL_MEANING: dict[ValidationLevel, str] = {
    ValidationLevel.SMOKE: "Runs and returns the documented shape.",
    ValidationLevel.INVARIANT: (
        "Mathematical invariants asserted (symmetry, bounds, normalisation)."
    ),
    ValidationLevel.SYNTHETIC: (
        "Compared against the published formula re-implemented directly, or "
        "an analytically known answer."
    ),
    ValidationLevel.REFERENCE: (
        "Compared numerically against an independent third-party implementation."
    ),
    ValidationLevel.BENCHMARK: (
        "Additionally evaluated on real datasets in a committed benchmark run."
    ),
}


def validation_matrix_rows() -> list[dict[str, str]]:
    """Return one row per encoder for the validation matrix, sorted by name."""

    rows = []
    for name in list_encoders():
        info = ENCODER_METADATA[name]
        rows.append(
            {
                "encoder": name,
                "family": info.family,
                "provenance": "canonical" if info.canonical_method else "project-defined",
                "reference": info.reference or "—",
                "complexity": info.complexity,
                "validation": info.validation_level.label,
                "tests": ", ".join(t.split("::")[-1] for t in info.validated_by) or "—",
                "notes": info.notes,
            }
        )
    return rows


def validation_matrix_markdown() -> str:
    """Render the validation matrix as the body of ``docs/encoder_validation.md``.

    The document is generated rather than hand-maintained;
    ``tests/test_representations.py`` fails if the committed file is stale.
    """

    lines = [
        "<!-- Generated by tscv_vision.representations.metadata."
        "validation_matrix_markdown(). Do not edit by hand; run "
        "`python scripts/generate_encoder_validation.py` to regenerate. -->",
        "",
        "# Encoder validation matrix",
        "",
        "One row per encoder. `provenance` is `canonical` only when the",
        "implementation reproduces the cited method **and** a test pins it to",
        "that method; everything else is `project-defined`, whatever its name",
        "may suggest.",
        "",
        "## Validation levels",
        "",
        "| Level | Meaning |",
        "| --- | --- |",
        *(f"| {level.label} | {_LEVEL_MEANING[level]} |" for level in ValidationLevel),
        "",
        "Passing unit tests alone is LEVEL 0. The target for core encoders is",
        "LEVEL 3; inclusion in publication experiments requires LEVEL 4, which",
        "no encoder has reached because no archive-scale run is committed yet",
        "(see [../results/README.md](../results/README.md)).",
        "",
        "## Matrix",
        "",
        "| Encoder | Family | Provenance | Validation | Complexity | Backing tests |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in validation_matrix_rows():
        lines.append(
            f"| `{row['encoder']}` | {row['family']} | {row['provenance']} "
            f"| {row['validation']} | `{row['complexity']}` | {row['tests']} |"
        )

    lines += ["", "## References and caveats", ""]
    for row in validation_matrix_rows():
        lines.append(f"### `{row['encoder']}`")
        lines.append("")
        lines.append(f"- **Reference:** {row['reference']}")
        lines.append(f"- **Provenance:** {row['provenance']}")
        if row["notes"]:
            lines.append(f"- **Caveats:** {row['notes']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
