"""Validation of the scattering encoder against its Kymatio backend.

This layer contributes validation, ordering and an image layout — not the
transform — so the tests check exactly that: the coefficients are the
backend's, unaltered, and the rearrangement into an image is a permutation
described by the metadata.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscv_vision import encoders
from tscv_vision.scattering import (
    _default_scattering,
    scattering_meta,
    scattering_transform,
)

pytest.importorskip("kymatio", reason="the scattering cascade is the backend's")

pytestmark = pytest.mark.optional

N = 2**11
J = 6
Q = 8


@pytest.fixture
def tone() -> np.ndarray:
    return np.sin(2 * np.pi * 0.05 * np.arange(N))


def _backend(n: int = N, j: int = J, q: int = Q):  # type: ignore[no-untyped-def]
    from kymatio.numpy import Scattering1D

    return Scattering1D(J=j, shape=(n,), Q=q)


# ---------------------------------------------------------------------------
# The coefficients are the backend's


def test_tensor_is_the_backend_output_verbatim(tone: np.ndarray) -> None:
    """No rescaling, no reordering: exactly what Kymatio returns."""

    np.testing.assert_array_equal(
        scattering_transform(tone, J=J, Q=Q, format="tensor"),
        np.asarray(_backend()(tone), dtype=float),
    )


def test_image_is_a_permutation_of_the_backend_coefficients(tone: np.ndarray) -> None:
    """The image rearranges rows; it must not invent or drop any."""

    raw = np.asarray(_backend()(tone), dtype=float)
    image = scattering_transform(tone, J=J, Q=Q, format="image", log_scale=False)
    assert image.shape == raw.shape
    # Same multiset of rows, up to the normalisation applied to the image.
    scaled = np.abs(raw) / np.abs(raw).max()
    np.testing.assert_allclose(
        np.sort(image, axis=None), np.sort(scaled, axis=None), atol=1e-12
    )


def test_image_rows_follow_the_documented_order(tone: np.ndarray) -> None:
    """Rows are sorted by (order, -xi1, -xi2), and the metadata says which."""

    meta = scattering_meta(N, J=J, Q=Q)
    order = np.asarray(meta["order"])
    xi = np.asarray(meta["xi"])

    assert np.all(np.diff(order) >= 0), "orders are not grouped"
    for value in np.unique(order):
        block = xi[order == value, 0]
        finite = block[np.isfinite(block)]
        assert np.all(np.diff(finite) <= 1e-12), "xi1 is not descending within an order"

    image = scattering_transform(tone, J=J, Q=Q, format="image", log_scale=False)
    assert image.shape[0] == meta["n_coefficients"]


def test_image_matches_the_metadata_permutation(tone: np.ndarray) -> None:
    """Reconstruct the image from the backend output using only the metadata."""

    raw = np.abs(np.asarray(_backend()(tone), dtype=float))
    backend_meta = _backend().meta()
    order = np.asarray(backend_meta["order"], dtype=float)
    xi = np.asarray(backend_meta["xi"], dtype=float)
    index = np.lexsort(
        (
            -np.nan_to_num(xi[:, 1], nan=-np.inf),
            -np.nan_to_num(xi[:, 0], nan=-np.inf),
            order,
        )
    )
    expected = raw[index] / raw[index].max()
    np.testing.assert_allclose(
        scattering_transform(tone, J=J, Q=Q, format="image", log_scale=False),
        expected,
        atol=1e-12,
    )


def test_metadata_describes_every_row(tone: np.ndarray) -> None:
    meta = scattering_meta(N, J=J, Q=Q)
    image = scattering_transform(tone, J=J, Q=Q, format="image")
    assert meta["n_coefficients"] == image.shape[0]
    assert np.asarray(meta["xi"]).shape == (image.shape[0], 2)
    assert np.asarray(meta["j"]).shape == (image.shape[0], 2)
    assert set(np.unique(np.asarray(meta["order"]))) <= {0, 1, 2}


# ---------------------------------------------------------------------------
# What the second order is for


def test_second_order_responds_to_amplitude_modulation() -> None:
    """First-order averaging destroys AM; the second order recovers it.

    That is the entire reason the cascade has a second layer, so it is worth
    asserting rather than assuming.
    """

    t = np.arange(N)
    carrier = np.sin(2 * np.pi * 0.1 * t)
    modulated = (1.0 + 0.8 * np.sin(2 * np.pi * 0.002 * t)) * carrier

    meta = _backend().meta()
    second = np.asarray(meta["order"]) == 2

    def energy(signal: np.ndarray) -> float:
        raw = scattering_transform(signal, J=J, Q=Q, format="tensor")
        return float(np.abs(raw[second]).mean())

    assert energy(modulated) > 2.0 * energy(carrier)


def test_modulation_image_layout() -> None:
    """Rows are spectral bands, columns modulation rates, both descending."""

    t = np.arange(N)
    modulated = (1.0 + 0.8 * np.sin(2 * np.pi * 0.002 * t)) * np.sin(2 * np.pi * 0.1 * t)
    image = scattering_transform(modulated, J=J, Q=Q, format="modulation")

    meta = _backend().meta()
    second = np.asarray(meta["order"]) == 2
    xi = np.asarray(meta["xi"], dtype=float)
    assert image.shape == (
        np.unique(xi[second, 0]).size,
        np.unique(xi[second, 1]).size,
    )
    assert np.all(image >= 0.0) and np.all(image <= 1.0)
    assert image.max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Invariance


def test_is_far_more_shift_stable_than_the_raw_signal() -> None:
    """Averaging by phi buys *approximate* local shift invariance.

    Stated as a comparison rather than an absolute tolerance: the transform is
    not exactly invariant, and claiming a made-up threshold would misdescribe
    it. What it delivers is an order of magnitude less sensitivity to a time
    shift than the signal itself. Windows are taken from a longer series so
    the shift is a genuine translation rather than a wrap-around.
    """

    length = N + 300
    grid = np.arange(length)
    long = np.sin(2 * np.pi * 0.05 * grid) + 0.3 * np.sin(2 * np.pi * 0.013 * grid)
    reference = long[:N]
    base = scattering_transform(reference, J=J, Q=Q, format="tensor")

    def relative(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).sum() / np.abs(a).sum())

    for shift in (1, 8, 32, 128):
        window = long[shift : shift + N]
        coefficient_change = relative(base, scattering_transform(window, J=J, Q=Q, format="tensor"))
        signal_change = relative(reference, window)
        assert coefficient_change < signal_change / 5.0, (
            f"shift {shift}: coefficients moved {coefficient_change:.3f} vs "
            f"signal {signal_change:.3f}"
        )
        assert coefficient_change < 0.25


def test_normalised_image_is_amplitude_invariant(tone: np.ndarray) -> None:
    np.testing.assert_allclose(
        scattering_transform(tone, J=J, Q=Q),
        scattering_transform(11.0 * tone, J=J, Q=Q),
        atol=1e-10,
    )


def test_deterministic(tone: np.ndarray) -> None:
    np.testing.assert_array_equal(
        scattering_transform(tone, J=J, Q=Q),
        scattering_transform(tone, J=J, Q=Q),
    )


# ---------------------------------------------------------------------------
# Formats, configuration, edges


@pytest.mark.parametrize("fmt", ["tensor", "image", "modulation"])
def test_formats_are_finite(fmt: str, tone: np.ndarray) -> None:
    out = scattering_transform(tone, J=J, Q=Q, format=fmt)  # type: ignore[arg-type]
    assert out.ndim == 2
    assert np.all(np.isfinite(out))


def test_log_scale_compresses_the_dynamic_range(tone: np.ndarray) -> None:
    linear = scattering_transform(tone, J=J, Q=Q, log_scale=False)
    logged = scattering_transform(tone, J=J, Q=Q, log_scale=True)
    assert logged.mean() > linear.mean()
    assert logged.max() == pytest.approx(1.0)


def test_larger_j_gives_more_paths_and_coarser_time(tone: np.ndarray) -> None:
    small = scattering_transform(tone, J=4, Q=Q, format="tensor")
    large = scattering_transform(tone, J=7, Q=Q, format="tensor")
    assert large.shape[0] > small.shape[0]
    assert large.shape[1] < small.shape[1]


def test_q_accepts_a_tuple(tone: np.ndarray) -> None:
    out = scattering_transform(tone, J=J, Q=(8, 1), format="tensor")
    assert out.ndim == 2 and np.all(np.isfinite(out))


def test_length_requirement() -> None:
    with pytest.raises(ValueError, match="too short for J"):
        scattering_transform(np.sin(np.linspace(0, 10, 32)), J=6)
    # Exactly 2 ** J is accepted.
    assert scattering_transform(np.sin(np.linspace(0, 10, 64)), J=6).ndim == 2


def test_invalid_arguments(tone: np.ndarray) -> None:
    for kwargs, match in [
        ({"J": 0}, "J must be"),
        ({"Q": 0}, "Q must be"),
        ({"Q": ()}, "Q must be"),
        ({"format": "nope"}, "format"),
    ]:
        with pytest.raises(ValueError, match=match):
            scattering_transform(tone, **kwargs)  # type: ignore[arg-type]


def test_nan_policy(tone: np.ndarray) -> None:
    dirty = tone.copy()
    dirty[100] = np.nan
    with pytest.raises(ValueError):
        scattering_transform(dirty, J=J, Q=Q)
    assert np.all(
        np.isfinite(scattering_transform(dirty, J=J, Q=Q, nan_policy="interpolate"))
    )


def test_registry_picks_a_workable_j_for_short_series() -> None:
    """The registry calls encoders with only a series, so J must adapt."""

    for n in (128, 256, 1024, 4096):
        out = _default_scattering(np.sin(np.linspace(0, 40.0, n)))
        assert out.ndim == 2 and np.all(np.isfinite(out))
    np.testing.assert_array_equal(
        encoders.get_encoder("scat")(np.sin(np.linspace(0, 40.0, 1024))),
        _default_scattering(np.sin(np.linspace(0, 40.0, 1024))),
    )


def test_registry_default_does_not_warn_about_border_effects() -> None:
    """Kymatio warns when the signal is short relative to 2**J."""

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for n in (128, 512, 2048):
            _default_scattering(np.sin(np.linspace(0, 40.0, n)))


def test_metadata_records_the_backend_and_the_naming_caveat() -> None:
    from tscv_vision.representations import get_representation_info

    info = get_representation_info("scat")
    assert info.canonical_method is True
    assert info.optional_dependency.startswith("kymatio")
    assert "not joint time-frequency scattering" in info.notes
