"""Validation of the horizontal visibility graph.

The linear stack algorithm is checked against the quadratic definition it
optimises, against examples worked by hand, and against the property that
separates it from the natural visibility graph already in the package.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscv_vision import encoders


def _brute_force(x: np.ndarray) -> np.ndarray:
    """The definition, transcribed directly: O(N^3) and obviously correct."""

    n = x.size
    adjacency = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if all(x[k] < min(x[i], x[j]) for k in range(i + 1, j)):
                adjacency[i, j] = adjacency[j, i] = 1.0
    return adjacency


# ---------------------------------------------------------------------------
# Correctness


@pytest.mark.parametrize("seed", range(8))
def test_matches_the_definition_by_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(25):
        series = rng.normal(size=int(rng.integers(2, 40)))
        np.testing.assert_array_equal(
            encoders.horizontal_visibility_graph(series), _brute_force(series)
        )


@pytest.mark.parametrize(
    "series",
    [
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 2.0, 1.0, 1.0],
        [5.0] * 6,
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],
    ],
)
def test_ties_and_patterns_match_the_definition(series: list[float]) -> None:
    arr = np.asarray(series)
    np.testing.assert_array_equal(
        encoders.horizontal_visibility_graph(arr), _brute_force(arr)
    )


def test_hand_computed_examples() -> None:
    """Worked out by hand from `x[k] < min(x[i], x[j])`."""

    # [1, 3, 2, 4]: 1-3 and 3-2 and 2-4 adjacent; 3-4 sees over the 2;
    # 1 sees nothing past 3, and 3 blocks 1 from 2 and 4.
    np.testing.assert_array_equal(
        encoders.horizontal_visibility_graph(np.array([1.0, 3.0, 2.0, 4.0])),
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ),
    )

    # A single valley: the two peaks see each other over it.
    np.testing.assert_array_equal(
        encoders.horizontal_visibility_graph(np.array([2.0, 1.0, 2.0])),
        np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
    )

    # A single peak blocks the outer pair.
    np.testing.assert_array_equal(
        encoders.horizontal_visibility_graph(np.array([1.0, 2.0, 1.0])),
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    )


# ---------------------------------------------------------------------------
# Structural properties


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_symmetry_zero_diagonal_and_adjacency(seed: int) -> None:
    series = np.random.default_rng(seed).normal(size=64)
    adjacency = encoders.horizontal_visibility_graph(series)
    np.testing.assert_array_equal(adjacency, adjacency.T)
    np.testing.assert_array_equal(np.diag(adjacency), np.zeros(64))
    # Consecutive samples are always horizontally visible: there is nothing
    # between them to block the line.
    np.testing.assert_array_equal(np.diag(adjacency, k=1), np.ones(63))
    assert set(np.unique(adjacency)) <= {0.0, 1.0}


def test_deterministic() -> None:
    series = np.random.default_rng(0).normal(size=128)
    np.testing.assert_array_equal(
        encoders.horizontal_visibility_graph(series),
        encoders.horizontal_visibility_graph(series),
    )


def test_invariant_under_monotonic_transformation() -> None:
    """The criterion is ordinal, so any strictly increasing map leaves it fixed."""

    series = np.random.default_rng(3).normal(size=96)
    base = encoders.horizontal_visibility_graph(series)
    for transform in (
        lambda v: 3.0 * v + 7.0,
        np.exp,
        lambda v: v**3,
        lambda v: np.arctan(v),
    ):
        np.testing.assert_array_equal(
            encoders.horizontal_visibility_graph(transform(series)), base
        )


def test_invariant_under_time_reversal() -> None:
    series = np.random.default_rng(4).normal(size=48)
    forward = encoders.horizontal_visibility_graph(series)
    backward = encoders.horizontal_visibility_graph(series[::-1])
    np.testing.assert_array_equal(backward, forward[::-1, ::-1])


# ---------------------------------------------------------------------------
# Distinct from the natural visibility graph


def test_hvg_is_a_strict_subgraph_of_the_nvg() -> None:
    """Every horizontal edge is a natural edge; the converse fails."""

    rng = np.random.default_rng(5)
    strictly_smaller = 0
    for _ in range(20):
        series = rng.normal(size=40)
        horizontal = encoders.horizontal_visibility_graph(series)
        natural = encoders.visibility_graph(series)
        assert np.all(horizontal <= natural), "HVG has an edge the NVG lacks"
        if horizontal.sum() < natural.sum():
            strictly_smaller += 1
    assert strictly_smaller > 0, "HVG and NVG were never different"


def test_hvg_and_nvg_differ_on_a_hand_picked_example() -> None:
    """A shallow ramp: visible by line of sight, blocked horizontally."""

    series = np.array([0.0, 1.0, 3.0])
    horizontal = encoders.horizontal_visibility_graph(series)
    natural = encoders.visibility_graph(series)
    # Natural: 0 and 2 see each other over the 1, because the ramp is convex.
    assert natural[0, 2] == 1.0
    # Horizontal: x[1] = 1 is not below min(0, 3) = 0, so it blocks.
    assert horizontal[0, 2] == 0.0


def test_monotonic_invariance_separates_the_two_graphs() -> None:
    """A cubic map leaves the HVG untouched and changes the NVG."""

    series = np.linspace(-2.0, 2.0, 24) + 0.4 * np.sin(np.linspace(0, 12, 24))
    assert np.array_equal(
        encoders.horizontal_visibility_graph(series),
        encoders.horizontal_visibility_graph(series**3),
    )
    assert not np.array_equal(
        encoders.visibility_graph(series), encoders.visibility_graph(series**3)
    )


# ---------------------------------------------------------------------------
# Weighted extensions


def test_amplitude_weighting() -> None:
    series = np.array([1.0, 3.0, 2.0, 4.0])
    binary = encoders.horizontal_visibility_graph(series)
    weighted = encoders.horizontal_visibility_graph(
        series, weighted=True, weight="amplitude"
    )
    # Same support, different values, normalised to [0, 1].
    np.testing.assert_array_equal(weighted > 0, binary > 0)
    assert weighted.max() == pytest.approx(1.0)
    np.testing.assert_array_equal(weighted, weighted.T)
    # Edge (1, 2) has amplitude |3 - 2| = 1; edge (0, 1) has |1 - 3| = 2.
    assert weighted[0, 1] > weighted[1, 2]


def test_distance_weighting() -> None:
    series = np.array([1.0, 3.0, 2.0, 4.0])
    weighted = encoders.horizontal_visibility_graph(
        series, weighted=True, weight="distance"
    )
    assert weighted[0, 1] == pytest.approx(0.5)  # 1 / (1 + 1)
    assert weighted[1, 3] == pytest.approx(1 / 3)  # 1 / (1 + 2)
    np.testing.assert_array_equal(weighted, weighted.T)


def test_weighted_flag_must_be_set_explicitly() -> None:
    series = np.array([1.0, 3.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="requires weighted=True"):
        encoders.horizontal_visibility_graph(series, weight="amplitude")
    with pytest.raises(ValueError, match="weight must be"):
        encoders.horizontal_visibility_graph(series, weight="nope")  # type: ignore[arg-type]
    # weighted=True with the default weight is still binary.
    np.testing.assert_array_equal(
        encoders.horizontal_visibility_graph(series, weighted=True),
        encoders.horizontal_visibility_graph(series),
    )


# ---------------------------------------------------------------------------
# Edge cases and integration


def test_short_and_constant_inputs() -> None:
    single = encoders.horizontal_visibility_graph(np.array([1.0]))
    assert single.shape == (1, 1) and single.sum() == 0.0
    pair = encoders.horizontal_visibility_graph(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(pair, np.array([[0.0, 1.0], [1.0, 0.0]]))
    constant = encoders.horizontal_visibility_graph(np.full(5, 2.0))
    np.testing.assert_array_equal(constant, _brute_force(np.full(5, 2.0)))


def test_nan_policy() -> None:
    series = np.array([1.0, np.nan, 3.0, 2.0])
    with pytest.raises(ValueError):
        encoders.horizontal_visibility_graph(series)
    filled = encoders.horizontal_visibility_graph(series, nan_policy="interpolate")
    assert np.all(np.isfinite(filled))
    with pytest.raises(ValueError):
        encoders.horizontal_visibility_graph(np.array([]))


def test_registry_and_metadata() -> None:
    from tscv_vision.representations import get_representation, get_representation_info

    series = np.random.default_rng(0).normal(size=32)
    np.testing.assert_array_equal(
        encoders.get_encoder("hvg")(series),
        encoders.horizontal_visibility_graph(series),
    )
    assert get_representation("hvg").transform(series).shape == (32, 32)
    info = get_representation_info("hvg")
    assert info.canonical_method is True
    assert "Luque" in (info.reference or "")


@pytest.mark.slow
def test_scaling_is_not_quadratic_in_time() -> None:
    """Edge finding is O(N); only materialising the matrix is quadratic."""

    import timeit

    rng = np.random.default_rng(0)
    timings = {}
    for n in (128, 256, 512, 1024, 2048):
        series = rng.normal(size=n)
        timings[n] = min(
            timeit.repeat(
                lambda s=series: encoders._hvg_edges(s), number=20, repeat=5
            )
        )
    # Doubling N must not quadruple the edge-finding time.
    for small, large in ((128, 256), (256, 512), (512, 1024), (1024, 2048)):
        assert timings[large] < 3.0 * timings[small], (
            f"edge finding scaled badly from N={small} to N={large}: {timings}"
        )
