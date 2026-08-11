"""Validation of the delay-embedding occupancy density.

The encoder is a TSCV-Vision rendering, but the object it renders — Takens'
delay embedding — has exact consequences for maps with a known functional
form, so the tests check geometry numerically rather than by eye.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscv_vision import encoders


def logistic_map(n: int, r: float = 4.0, x0: float = 0.4) -> np.ndarray:
    out = np.empty(n)
    value = x0
    for i in range(n):
        value = r * value * (1.0 - value)
        out[i] = value
    return out


def ar1(n: int, phi: float = 0.9, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    value = 0.0
    for i in range(n):
        value = phi * value + rng.normal()
        out[i] = value
    return out


def occupancy(image: np.ndarray) -> float:
    """Fraction of state-space cells the trajectory visits."""
    return float(np.mean(image > 0))


# ---------------------------------------------------------------------------
# The embedding recovers known geometry


def test_logistic_map_traces_its_analytic_parabola() -> None:
    """At delay 1 the embedding of x_{n+1} = 4 x_n (1 - x_n) *is* that parabola.

    Every occupied cell must lie on the curve, which is a far stronger claim
    than "the picture looks concentrated".
    """

    series = logistic_map(4000)
    bins = 32
    image = encoders.delay_embedding_density(
        series, delay=1, bins=bins, normalize=False
    )
    rows, cols = np.nonzero(image)

    lo, hi = float(series.min()), float(series.max())
    width = (hi - lo) / bins
    centres = lo + (np.arange(bins) + 0.5) * width
    horizontal = centres[cols]
    vertical = centres[rows]
    predicted = 4.0 * horizontal * (1.0 - horizontal)
    assert np.abs(vertical - predicted).max() < 3 * width


def test_periodic_signal_is_far_more_concentrated_than_noise() -> None:
    """A limit cycle is a curve; noise fills the plane."""

    sine = np.sin(np.linspace(0, 40.0, 4000))
    noise = np.random.default_rng(0).normal(size=4000)
    sine_image = encoders.delay_embedding_density(sine, delay=8, normalize=False)
    noise_image = encoders.delay_embedding_density(noise, delay=8, normalize=False)
    assert occupancy(sine_image) < 0.5 * occupancy(noise_image)


def test_chaotic_map_is_more_concentrated_than_noise_at_its_natural_delay() -> None:
    chaos = encoders.delay_embedding_density(
        logistic_map(4000), delay=1, normalize=False
    )
    noise = encoders.delay_embedding_density(
        np.random.default_rng(0).normal(size=4000), delay=1, normalize=False
    )
    assert occupancy(chaos) < 0.25 * occupancy(noise)


def test_signal_families_are_distinguishable() -> None:
    t = np.linspace(0, 40.0, 2000)
    families = {
        "sine": np.sin(t),
        "quasi_periodic": np.sin(t) + np.sin(np.sqrt(2.0) * t),
        "logistic": logistic_map(2000),
        "ar1": ar1(2000),
        "noise": np.random.default_rng(0).normal(size=2000),
    }
    images = {
        name: encoders.delay_embedding_density(sig, delay=8, normalize=False)
        for name, sig in families.items()
    }
    # The clean limit cycle is the most concentrated of the five.
    assert occupancy(images["sine"]) == min(occupancy(im) for im in images.values())
    # Adding a second incommensurate frequency fills more of the plane.
    assert occupancy(images["quasi_periodic"]) > occupancy(images["sine"])
    names = list(images)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not np.array_equal(images[a], images[b]), f"{a} == {b}"


# ---------------------------------------------------------------------------
# Mass and normalisation


def test_unnormalised_mass_is_the_number_of_embedded_points() -> None:
    series = np.sin(np.linspace(0, 20.0, 500))
    for dimension, delay in ((2, 1), (2, 7), (3, 4), (5, 3)):
        image = encoders.delay_embedding_density(
            series, dimension=dimension, delay=delay, normalize=False
        )
        expected = series.size - (dimension - 1) * delay
        assert image.sum() == pytest.approx(float(expected))


def test_normalisation_maps_the_peak_to_one() -> None:
    series = np.sin(np.linspace(0, 20.0, 500))
    image = encoders.delay_embedding_density(series, delay=8)
    assert image.max() == pytest.approx(1.0)
    assert image.min() >= 0.0
    raw = encoders.delay_embedding_density(series, delay=8, normalize=False)
    np.testing.assert_allclose(image, raw / raw.max(), atol=1e-12)


def test_gaussian_mode_preserves_mass_and_smooths() -> None:
    series = np.sin(np.linspace(0, 20.0, 1000))
    counts = encoders.delay_embedding_density(series, delay=8, normalize=False)
    smoothed = encoders.delay_embedding_density(
        series, delay=8, density="gaussian", sigma=1.5, normalize=False
    )
    # A normalised kernel conserves total mass, up to what falls off the edge.
    assert smoothed.sum() <= counts.sum() + 1e-9
    assert smoothed.sum() > 0.8 * counts.sum()
    # Smoothing spreads occupancy into neighbouring cells.
    assert occupancy(smoothed) > occupancy(counts)


def test_larger_sigma_smooths_more() -> None:
    series = np.sin(np.linspace(0, 20.0, 1000))
    narrow = encoders.delay_embedding_density(
        series, delay=8, density="gaussian", sigma=0.6, normalize=False
    )
    wide = encoders.delay_embedding_density(
        series, delay=8, density="gaussian", sigma=3.0, normalize=False
    )
    assert occupancy(wide) > occupancy(narrow)


# ---------------------------------------------------------------------------
# Invariances documented in the docstring


def test_invariant_to_affine_rescaling() -> None:
    """The grid spans the observed range, so gain and offset cancel."""

    series = np.sin(np.linspace(0, 20.0, 800))
    base = encoders.delay_embedding_density(series, delay=8, normalize=False)
    for scale, offset in ((3.0, 0.0), (1.0, 12.0), (0.25, -4.0)):
        np.testing.assert_array_equal(
            encoders.delay_embedding_density(
                scale * series + offset, delay=8, normalize=False
            ),
            base,
        )


def test_chronological_order_is_discarded() -> None:
    """The documented information loss, asserted rather than assumed.

    Reversing time traverses the same state-space cells in the opposite
    direction. For a symmetric projection the occupancy is therefore the
    transpose, and the set of occupied cells is unchanged.
    """

    series = np.sin(np.linspace(0, 20.0, 800))
    forward = encoders.delay_embedding_density(series, delay=8, normalize=False)
    backward = encoders.delay_embedding_density(series[::-1], delay=8, normalize=False)
    assert occupancy(forward) == pytest.approx(occupancy(backward), abs=0.01)
    np.testing.assert_allclose(forward.sum(), backward.sum())


def test_projection_selects_coordinates() -> None:
    series = np.sin(np.linspace(0, 20.0, 800))
    first = encoders.delay_embedding_density(
        series, dimension=4, delay=5, projection=(0, 1)
    )
    other = encoders.delay_embedding_density(
        series, dimension=4, delay=5, projection=(0, 3)
    )
    assert not np.array_equal(first, other)
    # Swapping the pair transposes the image.
    swapped = encoders.delay_embedding_density(
        series, dimension=4, delay=5, projection=(1, 0)
    )
    np.testing.assert_allclose(swapped, first.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Shape, determinism, edge cases


@pytest.mark.parametrize("bins", [2, 16, 64, 129])
def test_shape_and_dtype(bins: int) -> None:
    image = encoders.delay_embedding_density(
        np.sin(np.linspace(0, 20.0, 400)), delay=6, bins=bins
    )
    assert image.shape == (bins, bins)
    assert image.dtype == np.float64
    assert np.all(np.isfinite(image))


def test_deterministic() -> None:
    series = np.random.default_rng(0).normal(size=500)
    np.testing.assert_array_equal(
        encoders.delay_embedding_density(series, delay=3),
        encoders.delay_embedding_density(series, delay=3),
    )


def test_constant_input_occupies_a_single_cell() -> None:
    image = encoders.delay_embedding_density(np.full(50, 3.0), bins=16)
    assert np.count_nonzero(image) == 1
    assert image[8, 8] == pytest.approx(1.0)
    raw = encoders.delay_embedding_density(np.full(50, 3.0), bins=16, normalize=False)
    assert raw.sum() == pytest.approx(49.0)


def test_shortest_usable_input() -> None:
    # dimension 2, delay 1 needs exactly 2 samples for one embedded point.
    image = encoders.delay_embedding_density(np.array([0.0, 1.0]), bins=4, normalize=False)
    assert image.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="too short to embed"):
        encoders.delay_embedding_density(np.array([0.0]), bins=4)
    with pytest.raises(ValueError, match="too short to embed"):
        encoders.delay_embedding_density(np.arange(5.0), dimension=3, delay=4)


def test_invalid_arguments() -> None:
    series = np.sin(np.linspace(0, 20.0, 200))
    for kwargs, match in [
        ({"delay": 0}, "delay"),
        ({"dimension": 1}, "dimension"),
        ({"bins": 1}, "bins"),
        ({"projection": (0, 0)}, "must differ"),
        ({"projection": (0, 5)}, "outside"),
        ({"projection": (-1, 1)}, "outside"),
        ({"projection": (0, 1, 2)}, "pair"),
        ({"density": "nope"}, "density"),
        ({"density": "gaussian", "sigma": 0.0}, "sigma"),
        ({"density": "gaussian", "sigma": -1.0}, "sigma"),
    ]:
        with pytest.raises(ValueError, match=match):
            encoders.delay_embedding_density(series, **kwargs)  # type: ignore[arg-type]


def test_nan_policy() -> None:
    series = np.sin(np.linspace(0, 20.0, 200))
    dirty = series.copy()
    dirty[10] = np.nan
    with pytest.raises(ValueError):
        encoders.delay_embedding_density(dirty)
    assert np.all(
        np.isfinite(encoders.delay_embedding_density(dirty, nan_policy="interpolate"))
    )


def test_registry_and_metadata() -> None:
    from tscv_vision.representations import get_representation_info

    series = np.sin(np.linspace(0, 20.0, 200))
    np.testing.assert_array_equal(
        encoders.get_encoder("ded")(series), encoders.delay_embedding_density(series)
    )
    info = get_representation_info("ded")
    assert info.canonical_method is False
    assert "not a recurrence plot" in info.notes.lower()
