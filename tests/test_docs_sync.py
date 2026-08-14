"""Guard the documentation against drifting away from the code.

Stale signatures and stale feature dimensionalities were a real defect in this
project's history, so they are checked mechanically rather than by review:

* every ``### `fn(...) -> T` `` heading in ``docs/api.md`` must match
  :func:`inspect.signature`;
* the feature-dimension table must match :func:`~tscv_vision.features.feature_layout`;
* the documented encoder-registry keys must match
  :data:`~tscv_vision.encoders.ENCODER_REGISTRY`;
* the package version must be consistent across ``pyproject.toml``,
  ``setup.py``, citation metadata and ``tscv_vision.__version__``.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest

import tscv_vision
from tscv_vision import (
    analytics,
    encoders,
    evaluation,
    features,
    multivariate,
    representations,
    research,
    scattering,
    sliding,
    stats,
)

REPO = Path(__file__).resolve().parents[1]
API_DOC = REPO / "docs" / "api.md"

#: Documented name -> object. Names absent here are not signature-checked
#: (prose entries, classes documented by behaviour, ``...`` abbreviations).
DOCUMENTED: dict[str, Any] = {
    "gaf": encoders.gaf,
    "recurrence_plot": encoders.recurrence_plot,
    "spectrogram": encoders.spectrogram,
    "cwt": encoders.cwt,
    "mtf": encoders.mtf,
    "persistence_diagram": encoders.persistence_diagram,
    "persistence_image": encoders.persistence_image,
    "extrema_persistence_histogram": encoders.extrema_persistence_histogram,
    "gdf": encoders.gdf,
    "multi_scale_rp": encoders.multi_scale_rp,
    "dtw_matrix": encoders.dtw_matrix,
    "sax_symbols": encoders.sax_symbols,
    "sax": encoders.sax,
    "multi_scale_conv": encoders.multi_scale_conv,
    "window_attention": encoders.window_attention,
    "visibility_graph": encoders.visibility_graph,
    "shapelet_transform": encoders.shapelet_transform,
    "matrix_profile": encoders.matrix_profile,
    "random_projection_image": encoders.random_projection_image,
    "ensemble": encoders.ensemble,
    "extract_feature_vector": features.extract_feature_vector,
    "extract_batch": features.extract_batch,
    "feature_layout": features.feature_layout,
    "feature_vector_length": features.feature_vector_length,
    "sliding_windows": sliding.sliding_windows,
    "group_mean_disparity": research.group_mean_disparity,
    "add_laplace_noise": research.add_laplace_noise,
    "add_dp_noise": research.add_dp_noise,
    "cross_correlation_lag": analytics.cross_correlation_lag,
    "group_significance": analytics.group_significance,
    "evaluate": evaluation.evaluate,
    "run_benchmark": evaluation.run_benchmark,
    "compare_methods": evaluation.compare_methods,
    "summary_markdown": evaluation.summary_markdown,
    "nemenyi_critical_difference": stats.nemenyi_critical_difference,
    "cross_recurrence_plot": multivariate.cross_recurrence_plot,
    "joint_recurrence_plot": multivariate.joint_recurrence_plot,
    "wavelet_coherence": multivariate.wavelet_coherence,
    "delay_embed": multivariate.delay_embed,
    "scattering_transform": scattering.scattering_transform,
    "scattering_meta": scattering.scattering_meta,
    "list_representations": representations.list_representations,
    "get_representation": representations.get_representation,
    "get_representation_info": representations.get_representation_info,
    "register_representation": representations.register_representation,
    "list_encoders": representations.list_encoders,
    "get_encoder_metadata": representations.get_encoder_metadata,
    "holm_bonferroni": stats.holm_bonferroni,
}

# Signatures appear either as a section heading or as a bullet in a list.
_SIGNATURE = re.compile(
    r"^(?:### |- )`([A-Za-z_][A-Za-z0-9_]*)\((.*?)\)(?: -> (.+?))?`", re.M
)


def _render_signature(func: Any) -> tuple[str, str]:
    """Return ``(parameters, return_annotation)`` as they appear in the docs."""

    sig = inspect.signature(func)
    parts: list[str] = []
    seen_kwonly = False
    for param in sig.parameters.values():
        if param.kind is param.VAR_POSITIONAL:
            parts.append(f"*{param.name}")
            seen_kwonly = True
            continue
        if param.kind is param.VAR_KEYWORD:
            parts.append(f"**{param.name}")
            continue
        if param.kind is param.KEYWORD_ONLY and not seen_kwonly:
            parts.append("*")
            seen_kwonly = True
        if param.default is inspect.Parameter.empty:
            parts.append(param.name)
        else:
            parts.append(f"{param.name}={param.default!r}")
    ret = sig.return_annotation
    ret_text = "" if ret is inspect.Signature.empty else str(ret)
    return ", ".join(parts), ret_text


def _doc_entries() -> dict[str, tuple[str, str]]:
    text = API_DOC.read_text(encoding="utf-8")
    return {m.group(1): (m.group(2), m.group(3) or "") for m in _SIGNATURE.finditer(text)}


def test_api_doc_has_headings() -> None:
    entries = _doc_entries()
    assert len(entries) > 20, "docs/api.md lost its function headings"


@pytest.mark.parametrize("name", sorted(DOCUMENTED))
def test_documented_signature_matches_code(name: str) -> None:
    entries = _doc_entries()
    assert name in entries, f"{name} is no longer documented in docs/api.md"
    doc_params, doc_return = entries[name]
    if "..." in doc_params:
        pytest.skip(f"{name} is documented with an abbreviated signature")
    params, ret = _render_signature(DOCUMENTED[name])
    assert doc_params == params, (
        f"docs/api.md documents {name}({doc_params}) but the code has {name}({params})"
    )
    if doc_return:
        assert doc_return == ret, (
            f"docs/api.md says {name} returns {doc_return}, code says {ret}"
        )


def test_documented_names_all_exist() -> None:
    """Every heading in the API doc must name something importable."""

    from tscv_vision import pipeline

    modules = (
        encoders,
        features,
        sliding,
        analytics,
        research,
        stats,
        evaluation,
        pipeline,
        representations,
        multivariate,
        scattering,
    )
    for name in _doc_entries():
        if name in DOCUMENTED:
            continue
        bare = name.split(".")[-1]
        assert any(hasattr(mod, bare) for mod in modules), (
            f"docs/api.md documents {name!r}, which no module exports"
        )


def _doc_table(header: str) -> list[list[str]]:
    text = API_DOC.read_text(encoding="utf-8")
    start = text.index(header)
    rows = []
    for line in text[start:].splitlines()[2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip().strip("*").strip("`") for cell in line.strip("|").split("|")])
    return rows


def test_feature_dimension_table_is_current() -> None:
    """The documented per-feature sizes must match ``feature_layout``."""

    rows = _doc_table("| Feature | bins=8 | bins=16 | bins=32 |")
    documented = {row[0]: [int(v) for v in row[1:]] for row in rows}
    total = documented.pop("total")
    for bins_index, bins in enumerate((8, 16, 32)):
        layout = features.feature_layout(bins=bins)
        # Optional extractors (e.g. `wavelet`) are environment-dependent, so
        # the table lists the core set only; every documented row must exist.
        for name, sizes in documented.items():
            assert name in layout, f"docs list feature {name!r}, which is not registered"
            assert layout[name] == sizes[bins_index], (
                f"docs say {name} contributes {sizes[bins_index]} values at "
                f"bins={bins}, code gives {layout[name]}"
            )
        assert sum(documented[n][bins_index] for n in documented) == total[bins_index]


def test_documented_registry_keys_match_code() -> None:
    text = API_DOC.read_text(encoding="utf-8")
    section = text[text.index("### Registry") : text.index("## Feature extraction")]
    documented = set(re.findall(r"`([a-z_]+)`", section))
    documented -= {"register_encoder", "get_encoder", "name", "func"}
    actual = set(encoders.ENCODER_REGISTRY)
    assert documented == actual, (
        f"documented-but-missing: {sorted(documented - actual)}; "
        f"registered-but-undocumented: {sorted(actual - documented)}"
    )


def test_version_is_consistent_across_the_project() -> None:
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    setup_py = (REPO / "setup.py").read_text(encoding="utf-8")
    zenodo = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    citation = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    py_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    setup_version = re.search(r'version\s*=\s*"([^"]+)"', setup_py)
    citation_version = re.search(r'^version:\s*"([^"]+)"', citation, re.M)
    assert py_version and setup_version and citation_version
    assert py_version.group(1) == tscv_vision.__version__
    assert setup_version.group(1) == tscv_vision.__version__
    assert zenodo["version"] == tscv_vision.__version__
    assert citation_version.group(1) == tscv_vision.__version__


def _pyproject_extras() -> dict[str, set[str]]:
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    start = text.index("[tool.poetry.extras]")
    end = text.index("[", start + 1)
    while text[end : end + 6] not in {"[build", "[tool."} and end < len(text):
        end = text.index("[", end + 1)
    section = text[start:end]
    extras: dict[str, set[str]] = {}
    for name, body in re.findall(r"^(\w[\w-]*)\s*=\s*(\[[^\]]*\])", section, re.M | re.S):
        extras[name] = set(re.findall(r'"([^"]+)"', body))
    return extras


def _setup_extras() -> dict[str, set[str]]:
    text = (REPO / "setup.py").read_text(encoding="utf-8")
    start = text.index("extras_require={")
    end = text.index("\n    },", start)
    section = text[start:end]
    extras: dict[str, set[str]] = {}
    for name, body in re.findall(r'"([\w-]+)":\s*(\[[^\]]*\])', section, re.S):
        # Strip version specifiers so the two files can pin differently.
        extras[name] = {
            re.split(r"[<>=!~]", pkg)[0] for pkg in re.findall(r'"([^"]+)"', body)
        }
    return extras


def test_extras_match_between_pyproject_and_setup_py() -> None:
    """A package installable via Poetry must be installable via pip too."""

    poetry, setuptools_extras = _pyproject_extras(), _setup_extras()
    assert set(poetry) == set(setuptools_extras), (
        f"only in pyproject: {sorted(set(poetry) - set(setuptools_extras))}; "
        f"only in setup.py: {sorted(set(setuptools_extras) - set(poetry))}"
    )
    for name in poetry:
        assert poetry[name] == setuptools_extras[name], (
            f"extra {name!r} differs: pyproject has {sorted(poetry[name])}, "
            f"setup.py has {sorted(setuptools_extras[name])}"
        )


def test_pep561_marker_is_packaged() -> None:
    """Strict internal typing should be visible to downstream type checkers."""

    assert (REPO / "src" / "tscv_vision" / "py.typed").is_file()
    manifest = (REPO / "MANIFEST.in").read_text(encoding="utf-8")
    setup_py = (REPO / "setup.py").read_text(encoding="utf-8")
    assert "src/tscv_vision/py.typed" in manifest
    assert '"tscv_vision": ["py.typed"]' in setup_py


def test_every_optional_import_has_an_extra() -> None:
    """Optional third-party imports must have a documented install route."""

    declared = {pkg for pkgs in _pyproject_extras().values() for pkg in pkgs}
    # Import name -> distribution name, where they differ.
    import_to_dist = {
        "sklearn": "scikit-learn",
        "pywt": "pywavelets",
        "umap": "umap-learn",
        "kafka": "kafka-python",
        "cv2": "opencv-python",
        "yaml": "pyyaml",
        "cupyx": "cupy",
    }
    # Packages a user is never asked to install: reference implementations used
    # only by the optional test suite, and upstream research code with no
    # published wheel we can depend on.
    exempt = {"mamba_ssm", "retnet", "tensorflow", "prometheus_client", "torchvision"}

    import ast
    import sys

    src = REPO / "src" / "tscv_vision"
    found: set[str] = set()
    for path in src.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])

    third_party = {
        import_to_dist.get(name, name)
        for name in found
        if name not in exempt
        and name not in sys.stdlib_module_names
        and name not in {"numpy", "tscv_vision", "__future__"}
    }
    missing = sorted(third_party - declared)
    assert not missing, (
        f"these optional imports have no extra in pyproject.toml: {missing}. "
        "Add one so users have a documented way to install them."
    )


def test_changelog_documents_the_current_version() -> None:
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"[{tscv_vision.__version__}]" in changelog


def test_module_scope_classification_is_documented() -> None:
    scoped = tscv_vision.VALIDATED_CORE_MODULES | tscv_vision.CONTRIB_MODULES
    exported_modules = {
        name
        for name in tscv_vision.__all__
        if name
        not in {
            "AutoTSCV",
            "WindowedDataset",
            "VALIDATED_CORE_MODULES",
            "CONTRIB_MODULES",
        }
    }
    assert tscv_vision.VALIDATED_CORE_MODULES.isdisjoint(tscv_vision.CONTRIB_MODULES)
    assert exported_modules == scoped

    scope_doc = (REPO / "docs" / "scope.md").read_text(encoding="utf-8")
    for module in sorted(scoped):
        assert f"`{module}`" in scope_doc


def test_readme_does_not_claim_a_fixed_feature_dimension() -> None:
    """Regression: the README advertised 310 dims long after that changed."""

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    stale = re.findall(r"\b(\d{3,4})\s*(?:dims|dimensions|-dimensional)\b", readme)
    for number in stale:
        assert int(number) == features.feature_vector_length(bins=32), (
            f"README advertises {number} feature dimensions; the current value "
            f"is {features.feature_vector_length(bins=32)}. Prefer linking to "
            "feature_vector_length() instead of hard-coding a number."
        )
