"""Tests for the unified representation API and its provenance metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tscv_vision import encoders
from tscv_vision.representations import (
    ConcatFusion,
    DeterministicRepresentation,
    FittedRepresentation,
    GAFRepresentation,
    LearnedFusion,
    LearnedRepresentation,
    MTFRepresentation,
    NotFittedError,
    PersistenceImageRepresentation,
    PretrainedBackbone,
    PretrainedRepresentation,
    RecurrencePlotRepresentation,
    Representation,
    RepresentationInfo,
    SAXRepresentation,
    SpectrogramRepresentation,
    ValidationLevel,
    as_sklearn,
    get_representation,
    get_representation_info,
    list_encoders,
    list_representations,
    paa,
    register_representation,
    suggest_image_size,
    validation_matrix_markdown,
    validation_matrix_rows,
)
from tscv_vision.representations.metadata import ENCODER_ALIASES, ENCODER_METADATA

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def series() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.sin(np.linspace(0, 12 * np.pi, 128)) + 0.05 * rng.normal(size=128)


# ---------------------------------------------------------------------------
# Metadata


def test_every_builtin_encoder_has_metadata() -> None:
    """A new built-in encoder must declare its provenance before it ships.

    Compared against ``BUILTIN_ENCODERS`` rather than the live registry, since
    users (and other tests) may register their own encoders at runtime and are
    not obliged to document them here.
    """

    missing = sorted(encoders.BUILTIN_ENCODERS - set(ENCODER_METADATA))
    assert not missing, (
        f"encoders without metadata: {missing}. Add an entry to "
        "tscv_vision.representations.metadata.ENCODER_METADATA."
    )
    stale = sorted(set(ENCODER_METADATA) - encoders.BUILTIN_ENCODERS)
    assert not stale, f"metadata for encoders that no longer exist: {stale}"


def test_canonical_claims_require_a_reference() -> None:
    for name, info in ENCODER_METADATA.items():
        if info.canonical_method:
            assert info.reference, f"{name} claims to be canonical without a reference"


def test_canonical_claims_require_reference_level_validation() -> None:
    """Naming something after a paper obliges us to test it against the paper."""

    for name, info in ENCODER_METADATA.items():
        if info.canonical_method:
            assert info.validation_level >= ValidationLevel.SYNTHETIC, (
                f"{name} is marked canonical but is only validated at "
                f"{info.validation_level.label}; either test it against the "
                "published method or set canonical_method=False"
            )


def test_project_defined_encoders_explain_themselves() -> None:
    """A non-canonical encoder must say how it differs from what it resembles."""

    exempt = {"ensemble"}  # a meta-encoder, described by its constituents
    for name in list_encoders(canonical_method=False):
        if name in exempt:
            continue
        info = ENCODER_METADATA[name]
        assert info.notes, (
            f"{name} is project-defined but has no notes explaining what it is "
            "and what it is not"
        )


def test_validation_levels_name_the_tests_that_back_them() -> None:
    for name, info in ENCODER_METADATA.items():
        if info.validation_level >= ValidationLevel.INVARIANT:
            assert info.validated_by, f"{name} claims {info.validation_level.label} with no tests"
            for node in info.validated_by:
                path = REPO / node.split("::")[0]
                assert path.is_file(), f"{name}: {node} names a file that does not exist"


def test_validated_by_tests_actually_exist() -> None:
    """Every named test function must be present in the file it points at."""

    for name, info in ENCODER_METADATA.items():
        for node in info.validated_by:
            file_part, _, test_name = node.partition("::")
            if not test_name:
                continue
            source = (REPO / file_part).read_text(encoding="utf-8")
            assert f"def {test_name}(" in source, (
                f"{name}: {node} names a test that does not exist"
            )


def test_no_encoder_claims_benchmark_level() -> None:
    """LEVEL 4 requires a committed benchmark run, and there is none yet."""

    assert not list_encoders(min_validation_level=ValidationLevel.BENCHMARK)


def test_representation_info_rejects_unsupported_claims() -> None:
    with pytest.raises(ValueError, match="requires a reference"):
        RepresentationInfo(
            name="x",
            family="f",
            input_kind="univariate",
            output_kind="square_image",
            canonical_method=True,
        )
    with pytest.raises(ValueError, match="requires validated_by"):
        RepresentationInfo(
            name="x",
            family="f",
            input_kind="univariate",
            output_kind="square_image",
            validation_level=ValidationLevel.REFERENCE,
        )


def test_representation_info_is_serialisable() -> None:
    payload = ENCODER_METADATA["gaf"].as_dict()
    assert json.loads(json.dumps(payload))["canonical_method"] is True
    assert payload["validation_level"] == 3


def test_aliases_share_the_target_metadata() -> None:
    for alias, target in ENCODER_ALIASES.items():
        assert ENCODER_METADATA[alias].family == ENCODER_METADATA[target].family
        assert ENCODER_METADATA[alias].name == alias
    assert "tpa" not in list_encoders()
    assert "tpa" in list_encoders(include_aliases=True)


def test_list_encoders_filters() -> None:
    assert list_encoders(family="gramian") == ["gadf", "gaf", "gdf"]
    # The time-frequency family grows; assert the filter is sound rather than
    # pinning a list that every new spectral encoder would invalidate.
    time_frequency = list_encoders(output_kind="time_frequency")
    assert {"cwt", "spec", "sst"} <= set(time_frequency)
    assert all(
        ENCODER_METADATA[name].output_kind == "time_frequency"
        for name in time_frequency
    )
    assert "eph" not in list_encoders(canonical_method=True)
    strong = list_encoders(min_validation_level=ValidationLevel.REFERENCE)
    assert {"gaf", "mtf", "ph", "mp"} <= set(strong)
    assert "cwt" not in strong


# ---------------------------------------------------------------------------
# Validation matrix document


def test_validation_matrix_rows_cover_every_encoder() -> None:
    from tscv_vision.representations import MULTIVARIATE_METADATA

    rows = validation_matrix_rows()
    assert {row["encoder"] for row in rows} == set(list_encoders()) | set(
        MULTIVARIATE_METADATA
    )
    for row in rows:
        assert row["provenance"] in {"canonical", "project-defined"}
        assert row["validation"].startswith("LEVEL ")


def test_validation_matrix_doc_is_current() -> None:
    """Regenerate with `python scripts/generate_encoder_validation.py`."""

    doc = REPO / "docs" / "encoder_validation.md"
    assert doc.is_file(), "docs/encoder_validation.md is missing"
    assert doc.read_text(encoding="utf-8") == validation_matrix_markdown(), (
        "docs/encoder_validation.md is stale; regenerate it with "
        "`python scripts/generate_encoder_validation.py`"
    )


# ---------------------------------------------------------------------------
# Registry


def test_registry_covers_every_builtin_encoder() -> None:
    assert set(list_representations(include_aliases=True)) == set(encoders.BUILTIN_ENCODERS)


def _transform_or_skip(name: str, series: np.ndarray) -> np.ndarray:
    """Transform, skipping when the encoder needs an uninstalled extra.

    A missing optional dependency is a fact about the environment, not a
    defect; `info.optional_dependency` records which ones can be absent.
    """

    try:
        return get_representation(name).transform(series)
    except ImportError as exc:
        required = get_representation_info(name).optional_dependency
        assert required, f"{name} raised ImportError but declares no dependency"
        pytest.skip(f"{name} needs {required}: {exc}")


@pytest.mark.parametrize("name", list_representations())
def test_every_representation_transforms(name: str, series: np.ndarray) -> None:
    rep = get_representation(name)
    out = _transform_or_skip(name, series)
    assert isinstance(out, np.ndarray)
    assert out.size > 0
    assert np.all(np.isfinite(out))
    assert rep.info.name == name


@pytest.mark.parametrize("name", list_representations())
def test_every_representation_is_deterministic(name: str, series: np.ndarray) -> None:
    first = _transform_or_skip(name, series)
    second = get_representation(name).transform(series)
    np.testing.assert_array_equal(first, second)


def test_optional_dependencies_are_declared() -> None:
    """Anything that can raise ImportError must say which package it needs."""

    series = np.sin(np.linspace(0, 12 * np.pi, 128))
    for name in list_representations():
        info = get_representation_info(name)
        try:
            get_representation(name).transform(series)
        except ImportError:
            assert info.optional_dependency, (
                f"{name} raised ImportError but declares no optional_dependency"
            )


def test_get_representation_forwards_kwargs() -> None:
    assert get_representation("mtf", bins=4).info.family == "markov"
    rep = get_representation("gaf", image_size=16)
    assert rep.info.dimension == (16, 16)


def test_unknown_names_raise() -> None:
    with pytest.raises(KeyError, match="unknown representation"):
        get_representation("does-not-exist")
    with pytest.raises(KeyError, match="unknown representation"):
        get_representation_info("does-not-exist")
    with pytest.raises(KeyError, match="unknown encoder"):
        DeterministicRepresentation("does-not-exist")


def test_list_representations_filters() -> None:
    time_frequency = list_representations(family="time_frequency", trainable=False)
    assert {"cwt", "spec", "sst"} <= set(time_frequency)
    assert time_frequency == sorted(time_frequency)
    assert list_representations(trainable=True) == []
    assert list_representations(pretrained=True) == []
    assert "gaf" in list_representations(deterministic=True)
    assert list_representations(canonical_method=True, min_validation_level=3) == [
        "gadf",
        "gaf",
        "mp",
        "mtf",
        "mtspec",
        "ph",
        "scat",
    ]


def test_register_representation_requires_matching_metadata() -> None:
    info = RepresentationInfo(
        name="custom-demo", family="demo", input_kind="univariate", output_kind="embedding"
    )

    def factory(**kwargs: Any) -> Representation:  # pragma: no cover - not called
        raise NotImplementedError

    with pytest.raises(ValueError, match="info.name"):
        register_representation("other-name", factory, info)
    register_representation("custom-demo", factory, info)
    try:
        assert get_representation_info("custom-demo").family == "demo"
        with pytest.raises(ValueError, match="already registered"):
            register_representation("custom-demo", factory, info)
        register_representation("custom-demo", factory, info, overwrite=True)
    finally:
        from tscv_vision.representations import REPRESENTATION_INFO, REPRESENTATION_REGISTRY

        REPRESENTATION_REGISTRY.pop("custom-demo", None)
        REPRESENTATION_INFO.pop("custom-demo", None)


# ---------------------------------------------------------------------------
# Deterministic adapters


def test_paa_reduces_by_segment_means() -> None:
    np.testing.assert_allclose(paa(np.arange(8.0), 4), [0.5, 2.5, 4.5, 6.5])
    np.testing.assert_array_equal(paa(np.arange(4.0), 4), np.arange(4.0))
    with pytest.raises(ValueError):
        paa(np.arange(4.0), 5)
    with pytest.raises(ValueError):
        paa(np.arange(4.0), 0)
    with pytest.raises(ValueError, match="1D"):
        paa(np.zeros((2, 2)), 2)


def test_image_size_gives_a_fixed_output_shape() -> None:
    rep = GAFRepresentation(image_size=12)
    for length in (32, 64, 199):
        assert rep.transform(np.sin(np.linspace(0, 6.0, length))).shape == (12, 12)


def test_image_size_refuses_to_upsample() -> None:
    rep = GAFRepresentation(image_size=64)
    with pytest.raises(ValueError, match="shorter than"):
        rep.transform(np.linspace(0.0, 1.0, 16))
    with pytest.raises(ValueError, match="image_size must be positive"):
        DeterministicRepresentation("gaf", image_size=0)


def test_adapters_match_the_underlying_encoders(series: np.ndarray) -> None:
    """The adapters must add an interface, not change the mathematics."""

    np.testing.assert_allclose(GAFRepresentation().transform(series), encoders.gaf(series))
    np.testing.assert_allclose(
        GAFRepresentation(method="difference").transform(series),
        encoders.gaf(series, method="difference"),
    )
    np.testing.assert_allclose(
        RecurrencePlotRepresentation(eps=0.2).transform(series),
        encoders.recurrence_plot(series, eps=0.2),
    )
    np.testing.assert_allclose(
        SpectrogramRepresentation(win=32, hop=16).transform(series),
        encoders.spectrogram(series, win=32, hop=16),
    )
    np.testing.assert_allclose(
        MTFRepresentation(bins=4).transform(series), encoders.mtf(series, bins=4)
    )
    np.testing.assert_allclose(
        PersistenceImageRepresentation(bins=8).transform(series),
        encoders.persistence_image(series, bins=8),
    )
    np.testing.assert_allclose(
        SAXRepresentation(segments=8, alphabet=4).transform(series),
        encoders.sax(series, segments=8, alphabet=4),
    )


def test_gaf_representation_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="summation"):
        GAFRepresentation(method="nope")  # type: ignore[arg-type]


def test_sax_quantile_variant_is_downgraded_and_flagged() -> None:
    standard = SAXRepresentation().info
    variant = SAXRepresentation(breakpoints="quantile").info
    assert variant.validation_level < standard.validation_level
    assert "not comparable across series" in variant.notes


def test_persistence_image_info_reflects_explicit_ranges() -> None:
    relative = PersistenceImageRepresentation(bins=8).info
    absolute = PersistenceImageRepresentation(
        bins=8, birth_range=(-1.0, 1.0), pers_range=(0.0, 2.0)
    ).info
    assert "series-relative" in relative.notes
    assert "comparable across series" in absolute.notes
    assert absolute.dimension == (8, 8)


def test_nan_policy_is_honoured(series: np.ndarray) -> None:
    dirty = series.copy()
    dirty[5] = np.nan
    with pytest.raises(ValueError):
        GAFRepresentation().transform(dirty)
    out = GAFRepresentation(nan_policy="interpolate").transform(dirty)
    assert np.all(np.isfinite(out))


def test_repr_and_params_round_trip() -> None:
    rep = GAFRepresentation(method="difference", image_size=8)
    assert rep.get_params() == {
        "method": "difference",
        "image_size": 8,
        "nan_policy": "raise",
    }
    assert "GAFRepresentation(" in repr(rep)
    assert "method='difference'" in repr(rep)


def test_suggest_image_size() -> None:
    assert suggest_image_size(64) == 64
    assert suggest_image_size(1000) == 128
    assert suggest_image_size(1000, maximum=32) == 32
    with pytest.raises(ValueError):
        suggest_image_size(0)


# ---------------------------------------------------------------------------
# Batch helpers


def test_transform_many_and_stack(series: np.ndarray) -> None:
    rep = GAFRepresentation(image_size=8)
    batch = [series, series * 2.0, series + 1.0]
    assert len(rep.transform_many(batch)) == 3
    assert rep.transform_stack(batch).shape == (3, 8, 8)
    assert len(list(rep.iter_transform(batch))) == 3


def test_transform_stack_reports_ragged_output() -> None:
    rep = GAFRepresentation()  # no image_size: output follows input length
    with pytest.raises(ValueError, match="differing shapes"):
        rep.transform_stack([np.linspace(0, 1, 16), np.linspace(0, 1, 32)])
    with pytest.raises(ValueError, match="empty"):
        rep.transform_stack([])


# ---------------------------------------------------------------------------
# Fusion


def test_concat_fusion_shapes(series: np.ndarray) -> None:
    fusion = ConcatFusion(
        [get_representation("gaf", image_size=8), get_representation("mtf", image_size=8)]
    )
    out = fusion.transform(series)
    assert out.shape == (128,)
    assert fusion.view_names == ["gaf", "mtf"]
    info = fusion.info
    assert info.family == "fusion"
    assert info.canonical_method is False
    assert info.output_kind == "embedding"


def test_concat_fusion_validation_level_is_the_weakest_view(series: np.ndarray) -> None:
    fusion = ConcatFusion([get_representation("gaf"), get_representation("cwt")])
    assert fusion.info.validation_level == ValidationLevel.INVARIANT


def test_concat_fusion_modes(series: np.ndarray) -> None:
    views = [get_representation("gaf", image_size=8), get_representation("mtf", image_size=8)]
    mean = ConcatFusion(views, mode="mean").transform(series)
    assert mean.shape == (64,)
    weighted = ConcatFusion(views, mode="weighted", weights=[0.25, 0.75]).transform(series)
    assert weighted.shape == (64,)
    with pytest.raises(ValueError, match="requires weights"):
        ConcatFusion(views, mode="weighted")
    with pytest.raises(ValueError, match="one entry per view"):
        ConcatFusion(views, weights=[1.0])
    with pytest.raises(ValueError, match="at least one view"):
        ConcatFusion([])


# ---------------------------------------------------------------------------
# Abstract contracts


def test_abstract_bases_cannot_be_instantiated() -> None:
    for cls in (
        Representation,
        FittedRepresentation,
        PretrainedRepresentation,
        LearnedRepresentation,
        LearnedFusion,
        PretrainedBackbone,
    ):
        with pytest.raises(TypeError):
            cls()  # type: ignore[abstract]


class _MeanShift(FittedRepresentation):
    """Minimal fitted representation used to exercise the contract."""

    def __init__(self) -> None:
        self.mean_: float | None = None

    def _fit(self, X: Any, y: Any = None) -> None:
        self.mean_ = float(np.mean([np.mean(x) for x in X]))

    def transform(self, x: np.ndarray) -> np.ndarray:
        self.check_fitted()
        assert self.mean_ is not None
        return np.asarray(x, dtype=float) - self.mean_

    @property
    def info(self) -> RepresentationInfo:
        return RepresentationInfo(
            name="mean-shift",
            family="demo",
            input_kind="univariate",
            output_kind="embedding",
            trainable=True,
        )


def test_fitted_representation_refuses_to_transform_before_fit() -> None:
    rep = _MeanShift()
    with pytest.raises(NotFittedError, match="training data only"):
        rep.transform(np.zeros(4))
    batch = [np.zeros(4), np.ones(4)]
    out = rep.fit_transform(batch)
    assert len(out) == 2
    np.testing.assert_allclose(out[0], -0.5)


def test_learned_representation_reseeds_on_every_fit() -> None:
    class _Draws(LearnedRepresentation):
        def _fit(self, X: Any, y: Any = None) -> None:
            self.draw_ = float(self.rng.normal())

        def transform(self, x: np.ndarray) -> np.ndarray:
            self.check_fitted()
            return np.asarray(x, dtype=float)

        def state_dict(self) -> dict[str, Any]:
            return {"draw": self.draw_}

        def load_state_dict(self, state: dict[str, Any]) -> None:
            self.draw_ = state["draw"]

        @property
        def info(self) -> RepresentationInfo:
            return RepresentationInfo(
                name="draws",
                family="demo",
                input_kind="univariate",
                output_kind="embedding",
                trainable=True,
                deterministic=True,
            )

    rep = _Draws(random_state=7)
    first = rep.fit([np.zeros(2)]).state_dict()
    second = rep.fit([np.zeros(2)]).state_dict()
    assert first == second, "re-fitting with the same seed must give the same parameters"

    other = _Draws(random_state=8)
    assert other.fit([np.zeros(2)]).state_dict() != first
    rep.load_state_dict({"draw": 1.5})
    assert rep.state_dict() == {"draw": 1.5}


def test_pretrained_backbone_requires_a_named_checkpoint() -> None:
    class _Dummy(PretrainedBackbone):
        def encode(self, X: Any) -> np.ndarray:
            return np.zeros((len(list(X)), 3))

        @property
        def info(self) -> RepresentationInfo:
            return self._base_info(
                RepresentationInfo(
                    name="dummy",
                    family="demo",
                    input_kind="image",
                    output_kind="embedding",
                )
            )

    with pytest.raises(ValueError, match="checkpoint must name"):
        _Dummy(model_name="m", checkpoint="")
    with pytest.raises(ValueError, match="batch_size"):
        _Dummy(model_name="m", checkpoint="c", batch_size=0)

    rep = _Dummy(model_name="ViT-B-32", checkpoint="laion2b")
    assert rep.transform(np.zeros((4, 4))).shape == (3,)
    assert len(rep.transform_many([np.zeros((4, 4))] * 2)) == 2
    info = rep.info
    assert info.pretrained is True
    assert "ViT-B-32 @ laion2b" in (info.reference or "")
    assert "contamination" in info.notes


# ---------------------------------------------------------------------------
# scikit-learn interoperability


def test_as_sklearn_wraps_a_representation(series: np.ndarray) -> None:
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    transformer = as_sklearn(get_representation("gaf", image_size=8))
    X = np.stack([series, series * 2.0, series + 1.0])
    out = transformer.fit_transform(X)
    assert out.shape == (3, 8, 8)
    assert clone(transformer) is not transformer


def test_as_sklearn_fits_fitted_representations_inside_fit() -> None:
    pytest.importorskip("sklearn")
    rep = _MeanShift()
    transformer = as_sklearn(rep, stack=True)
    X = np.stack([np.zeros(4), np.ones(4)])
    out = transformer.fit_transform(X)
    assert out.shape == (2, 4)
    assert rep.mean_ == pytest.approx(0.5)


def test_as_sklearn_can_return_ragged_output() -> None:
    pytest.importorskip("sklearn")
    transformer = as_sklearn(get_representation("gaf"), stack=False)
    out = transformer.fit_transform([np.linspace(0, 1, 16), np.linspace(0, 1, 32)])
    assert [o.shape for o in out] == [(16, 16), (32, 32)]
