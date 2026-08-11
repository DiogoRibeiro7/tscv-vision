"""Validation of the cross recurrence plot and the shared recurrence utilities."""

from __future__ import annotations

import numpy as np
import pytest

from tscv_vision import encoders
from tscv_vision.multivariate import (
    _pairwise_distance,
    _recurrence_threshold,
    cross_recurrence_plot,
    delay_embed,
)


@pytest.fixture
def sine() -> np.ndarray:
    return np.sin(np.linspace(0, 20.0, 300))


# ---------------------------------------------------------------------------
# Shared utilities


def test_delay_embed_matches_the_definition() -> None:
    np.testing.assert_array_equal(
        delay_embed(np.arange(5.0), dimension=2, delay=1),
        np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
    )
    np.testing.assert_array_equal(
        delay_embed(np.arange(7.0), dimension=3, delay=2),
        np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]),
    )
    # dimension=1 is the degenerate column case.
    np.testing.assert_array_equal(
        delay_embed(np.arange(3.0)), np.array([[0.0], [1.0], [2.0]])
    )


def test_delay_embed_validation() -> None:
    for kwargs, match in [
        ({"dimension": 0}, "dimension"),
        ({"delay": 0}, "delay"),
        ({"dimension": 9}, "too short"),
    ]:
        with pytest.raises(ValueError, match=match):
            delay_embed(np.arange(5.0), **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1D"):
        delay_embed(np.zeros((2, 2)))


@pytest.mark.parametrize(
    ("metric", "expected"),
    [("euclidean", 5.0), ("manhattan", 7.0), ("chebyshev", 4.0)],
)
def test_pairwise_distance_metrics(metric: str, expected: float) -> None:
    left = np.array([[0.0, 0.0]])
    right = np.array([[3.0, 4.0]])
    assert _pairwise_distance(left, right, metric) == pytest.approx(expected)  # type: ignore[arg-type]


def test_pairwise_distance_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="metric must be"):
        _pairwise_distance(np.zeros((1, 1)), np.zeros((1, 1)), "cosine")  # type: ignore[arg-type]


def test_threshold_is_the_requested_quantile() -> None:
    distances = np.arange(100.0).reshape(10, 10)
    assert _recurrence_threshold(distances, None, 0.5) == pytest.approx(49.5)
    assert _recurrence_threshold(distances, 7.0, 0.5) == 7.0
    for kwargs, match in [((None, 0.0), "recurrence_rate"), ((None, 1.0), "recurrence_rate")]:
        with pytest.raises(ValueError, match=match):
            _recurrence_threshold(distances, *kwargs)
    with pytest.raises(ValueError, match="epsilon"):
        _recurrence_threshold(distances, -1.0, 0.5)


# ---------------------------------------------------------------------------
# Consistency with the existing recurrence plot


def test_self_cross_recurrence_matches_encoders_recurrence_plot() -> None:
    """The property the shared utilities were meant to guarantee.

    `encoders.recurrence_plot` keeps its own implementation because of its
    compiled backends, so agreement is asserted rather than assumed.
    """

    rng = np.random.default_rng(0)
    for _ in range(20):
        series = rng.normal(size=int(rng.integers(4, 60)))
        np.testing.assert_allclose(
            cross_recurrence_plot(series, series, binary=False),
            encoders.recurrence_plot(series),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Structure of the plot


def test_identical_series_fill_the_main_diagonal(sine: np.ndarray) -> None:
    plot = cross_recurrence_plot(sine, sine, recurrence_rate=0.05)
    np.testing.assert_array_equal(np.diag(plot), np.ones(sine.size))
    assert np.array_equal(plot, plot.T)


def test_phase_shift_moves_the_diagonal_by_the_lag() -> None:
    """The offset of the dominant diagonal recovers the known lag."""

    grid = np.linspace(0, 20.0, 300)
    step = grid[1] - grid[0]
    shift_radians = 1.0
    plot = cross_recurrence_plot(
        np.sin(grid), np.sin(grid + shift_radians), recurrence_rate=0.05
    )
    offsets = np.arange(-40, 41)
    strength = [np.trace(plot, offset=int(k)) / (plot.shape[0] - abs(int(k))) for k in offsets]
    recovered = offsets[int(np.argmax(strength))]
    expected = -shift_radians / step
    assert abs(recovered - expected) <= 1.0


def test_different_frequencies_have_no_persistent_diagonal() -> None:
    grid = np.linspace(0, 20.0, 300)
    same = cross_recurrence_plot(np.sin(grid), np.sin(grid), recurrence_rate=0.05)
    different = cross_recurrence_plot(np.sin(grid), np.sin(2.7 * grid), recurrence_rate=0.05)
    assert np.mean(np.diag(same)) > 5 * np.mean(np.diag(different))


def test_independent_noise_has_no_structure() -> None:
    rng = np.random.default_rng(1)
    plot = cross_recurrence_plot(
        rng.normal(size=200), rng.normal(size=200), recurrence_rate=0.1
    )
    # The main diagonal is no denser than the plot as a whole.
    assert abs(np.mean(np.diag(plot)) - plot.mean()) < 0.08


def test_unequal_lengths_give_a_rectangular_plot(sine: np.ndarray) -> None:
    plot = cross_recurrence_plot(sine, sine[:120])
    assert plot.shape == (300, 120)
    transposed = cross_recurrence_plot(sine[:120], sine)
    assert transposed.shape == (120, 300)
    np.testing.assert_array_equal(transposed, plot.T)


def test_embedding_shrinks_both_axes(sine: np.ndarray) -> None:
    plot = cross_recurrence_plot(sine, sine[:120], dimension=3, delay=4)
    assert plot.shape == (300 - 8, 120 - 8)


# ---------------------------------------------------------------------------
# Thresholds


def test_automatic_threshold_hits_the_target_rate() -> None:
    rng = np.random.default_rng(2)
    for rate in (0.05, 0.1, 0.3):
        plot = cross_recurrence_plot(
            rng.normal(size=150), rng.normal(size=150), recurrence_rate=rate
        )
        assert plot.mean() == pytest.approx(rate, abs=0.01)


def test_explicit_epsilon_is_used_verbatim() -> None:
    left = np.array([0.0, 1.0, 2.0, 3.0])
    plot = cross_recurrence_plot(left, left, epsilon=1.0)
    expected = (np.abs(left[:, None] - left[None, :]) <= 1.0).astype(float)
    np.testing.assert_array_equal(plot, expected)
    assert cross_recurrence_plot(left, left, epsilon=0.0).sum() == 4.0


def test_automatic_threshold_is_scale_invariant() -> None:
    """A rate target rescales with the data; a fixed epsilon does not."""

    rng = np.random.default_rng(3)
    series = rng.normal(size=120)
    base = cross_recurrence_plot(series, series, recurrence_rate=0.1)
    scaled = cross_recurrence_plot(50.0 * series, 50.0 * series, recurrence_rate=0.1)
    np.testing.assert_array_equal(base, scaled)

    fixed = cross_recurrence_plot(series, series, epsilon=0.5)
    fixed_scaled = cross_recurrence_plot(50.0 * series, 50.0 * series, epsilon=0.5)
    assert not np.array_equal(fixed, fixed_scaled)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_metrics_all_work(metric: str, sine: np.ndarray) -> None:
    plot = cross_recurrence_plot(sine, sine, dimension=3, delay=2, metric=metric)  # type: ignore[arg-type]
    assert plot.shape == (296, 296)
    assert set(np.unique(plot)) <= {0.0, 1.0}


def test_metrics_agree_in_one_dimension(sine: np.ndarray) -> None:
    """With m=1 every metric reduces to the absolute difference."""

    plots = [
        cross_recurrence_plot(sine, sine, metric=m, binary=False)  # type: ignore[arg-type]
        for m in ("euclidean", "manhattan", "chebyshev")
    ]
    np.testing.assert_allclose(plots[0], plots[1], atol=1e-12)
    np.testing.assert_allclose(plots[1], plots[2], atol=1e-12)


# ---------------------------------------------------------------------------
# Continuous form, edges, determinism


def test_continuous_form_is_similarity_in_zero_to_one(sine: np.ndarray) -> None:
    plot = cross_recurrence_plot(sine, sine, binary=False)
    assert np.all(plot >= 0.0) and np.all(plot <= 1.0)
    np.testing.assert_allclose(np.diag(plot), 1.0, atol=1e-12)


def test_constant_series_are_everywhere_identical() -> None:
    plot = cross_recurrence_plot(np.full(10, 2.0), np.full(6, 2.0), binary=False)
    assert plot.shape == (10, 6)
    np.testing.assert_array_equal(plot, np.ones((10, 6)))
    binary = cross_recurrence_plot(np.full(10, 2.0), np.full(6, 2.0))
    np.testing.assert_array_equal(binary, np.ones((10, 6)))


def test_deterministic(sine: np.ndarray) -> None:
    np.testing.assert_array_equal(
        cross_recurrence_plot(sine, sine[:100]),
        cross_recurrence_plot(sine, sine[:100]),
    )


def test_invalid_arguments(sine: np.ndarray) -> None:
    for kwargs, match in [
        ({"dimension": 0}, "dimension"),
        ({"delay": 0}, "delay"),
        ({"epsilon": -1.0}, "epsilon"),
        ({"recurrence_rate": 0.0}, "recurrence_rate"),
        ({"recurrence_rate": 1.0}, "recurrence_rate"),
        ({"metric": "cosine"}, "metric"),
    ]:
        with pytest.raises(ValueError, match=match):
            cross_recurrence_plot(sine, sine, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cross_recurrence_plot(np.array([]), sine)
    with pytest.raises(ValueError, match="too short"):
        cross_recurrence_plot(np.arange(3.0), sine, dimension=5, delay=2)


def test_nan_policy_is_applied_to_each_series(sine: np.ndarray) -> None:
    dirty = sine.copy()
    dirty[7] = np.nan
    with pytest.raises(ValueError):
        cross_recurrence_plot(dirty, sine)
    plot = cross_recurrence_plot(dirty, sine, nan_policy="interpolate")
    assert np.all(np.isfinite(plot))


def test_metadata_is_recorded() -> None:
    """Multivariate encoders are outside the univariate registry but not outside
    the provenance system."""

    from tscv_vision.representations import (
        MULTIVARIATE_METADATA,
        get_encoder_metadata,
        list_encoders,
    )

    info = get_encoder_metadata("cross_recurrence_plot")
    assert info.canonical_method is True
    assert info.input_kind == "bivariate"
    assert "Marwan" in (info.reference or "")
    assert "cross_recurrence_plot" in MULTIVARIATE_METADATA
    # It must not leak into the univariate registry listing.
    assert "cross_recurrence_plot" not in list_encoders()
