"""Validation of the Ordinal Pattern Transition Field.

The encoder itself is a TSCV-Vision composition, but its first stage is
Bandt & Pompe's ordinal encoding, which has published consequences that can be
checked rather than merely smoke-tested — most usefully the *forbidden
patterns* of deterministic chaos.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tscv_vision import encoders
from tscv_vision.encoders import _ordinal_patterns


def logistic_map(n: int, r: float = 4.0, x0: float = 0.4) -> np.ndarray:
    """Deterministic chaos: x_{n+1} = r x_n (1 - x_n)."""

    out = np.empty(n)
    value = x0
    for i in range(n):
        value = r * value * (1.0 - value)
        out[i] = value
    return out


@pytest.fixture
def sine() -> np.ndarray:
    return np.sin(np.linspace(0, 12.0, 128))


# ---------------------------------------------------------------------------
# Stage 1: ordinal encoding


@pytest.mark.parametrize("order", [2, 3, 4, 5])
def test_lehmer_codes_are_a_bijection_onto_the_state_space(order: int) -> None:
    """Every permutation of `order` elements gets its own integer in [0, m!)."""

    rng = np.random.default_rng(0)
    codes = set()
    for _ in range(4000):
        window = rng.normal(size=order)
        codes.add(int(_ordinal_patterns(window, order, 1, "stable", 0)[0]))
    assert codes == set(range(math.factorial(order)))


def test_pattern_depends_only_on_ordering() -> None:
    ascending = _ordinal_patterns(np.array([1.0, 2.0, 3.0]), 3, 1, "stable", 0)
    also_ascending = _ordinal_patterns(np.array([-9.0, 0.5, 100.0]), 3, 1, "stable", 0)
    descending = _ordinal_patterns(np.array([3.0, 2.0, 1.0]), 3, 1, "stable", 0)
    assert ascending == also_ascending
    assert ascending != descending


def test_delay_changes_the_embedding(sine: np.ndarray) -> None:
    assert not np.array_equal(
        _ordinal_patterns(sine, 3, 1, "stable", 0),
        _ordinal_patterns(sine, 3, 4, "stable", 0),
    )


def test_logistic_map_has_forbidden_patterns_and_noise_does_not() -> None:
    """Amigó, Zambrano & Sanjuán (2007): deterministic maps forbid patterns.

    The logistic map at r=4 admits exactly five of the six order-3 patterns;
    an i.i.d. sequence admits all six. This is the published property that
    makes ordinal encoding useful, so it is worth asserting rather than
    assuming.
    """

    chaotic = _ordinal_patterns(logistic_map(5000), 3, 1, "stable", 0)
    noise = _ordinal_patterns(
        np.random.default_rng(0).normal(size=5000), 3, 1, "stable", 0
    )
    assert len(np.unique(chaotic)) == 5
    assert len(np.unique(noise)) == 6

    # The effect strengthens with order: more patterns are forbidden.
    chaotic4 = _ordinal_patterns(logistic_map(5000), 4, 1, "stable", 0)
    noise4 = _ordinal_patterns(
        np.random.default_rng(1).normal(size=5000), 4, 1, "stable", 0
    )
    assert len(np.unique(chaotic4)) < len(np.unique(noise4)) == 24


# ---------------------------------------------------------------------------
# Stage 2: transition matrix


def test_observed_rows_are_probability_distributions(sine: np.ndarray) -> None:
    matrix = encoders.ordinal_transition_field(sine, mode="transition_matrix")
    assert matrix.shape == (6, 6)
    totals = matrix.sum(axis=1)
    observed = totals > 0
    np.testing.assert_allclose(totals[observed], 1.0, atol=1e-12)
    # Unobserved states stay at zero rather than being given a uniform row,
    # which would invent a conditional distribution that was never measured.
    assert np.all(matrix[~observed] == 0.0)


def test_transition_matrix_counts_match_a_direct_tally() -> None:
    series = logistic_map(500)
    codes = _ordinal_patterns(series, 3, 1, "stable", 0)
    matrix = encoders.ordinal_transition_field(series, mode="transition_matrix")

    expected = np.zeros((6, 6))
    for a, b in zip(codes[:-1], codes[1:], strict=True):
        expected[a, b] += 1.0
    totals = expected.sum(axis=1, keepdims=True)
    expected = np.divide(expected, totals, out=np.zeros_like(expected), where=totals > 0)
    np.testing.assert_allclose(matrix, expected, atol=1e-12)


def test_transition_matrix_size_is_factorial(sine: np.ndarray) -> None:
    for order in (2, 3, 4):
        matrix = encoders.ordinal_transition_field(
            sine, order=order, mode="transition_matrix"
        )
        assert matrix.shape == (math.factorial(order), math.factorial(order))


# ---------------------------------------------------------------------------
# Stage 3: field


def test_field_shape_and_range(sine: np.ndarray) -> None:
    field = encoders.ordinal_transition_field(sine, order=3, delay=1)
    assert field.shape == (128 - 2, 128 - 2)
    assert np.all(field >= 0.0) and np.all(field <= 1.0)
    assert field.dtype == np.float64


def test_field_entries_are_the_transition_probabilities(sine: np.ndarray) -> None:
    codes = _ordinal_patterns(sine, 3, 1, "stable", 0)
    matrix = encoders.ordinal_transition_field(sine, mode="transition_matrix")
    field = encoders.ordinal_transition_field(sine)
    for i in (0, 5, 30):
        for j in (1, 17, 60):
            assert field[i, j] == pytest.approx(matrix[codes[i], codes[j]])


def test_image_size_downsamples_deterministically(sine: np.ndarray) -> None:
    small = encoders.ordinal_transition_field(sine, image_size=16)
    assert small.shape == (16, 16)
    np.testing.assert_array_equal(
        small, encoders.ordinal_transition_field(sine, image_size=16)
    )
    assert np.all(small >= 0.0) and np.all(small <= 1.0)


# ---------------------------------------------------------------------------
# The documented invariant


def test_invariant_under_monotonic_transformation(sine: np.ndarray) -> None:
    """Only the ordering is used, so any strictly increasing map is a no-op."""

    base = encoders.ordinal_transition_field(sine)
    for transform in (
        lambda v: 3.0 * v + 7.0,
        np.exp,
        lambda v: v**3,
        lambda v: np.arctan(v),
        lambda v: np.sign(v) * np.abs(v) ** 0.5,
    ):
        np.testing.assert_array_equal(
            encoders.ordinal_transition_field(transform(sine)), base
        )


def test_not_invariant_under_a_decreasing_transformation() -> None:
    """A reversing map permutes the patterns, so the statistics must change.

    Tested on an asymmetric signal on purpose. Negating a sine merely shifts
    its phase, leaving the ordinal statistics identical — invariance there is a
    property of the signal, not of the encoder.
    """

    for signal in (logistic_map(300), np.tile(np.linspace(0.0, 1.0, 7), 40)):
        assert not np.allclose(
            encoders.ordinal_transition_field(-signal, mode="transition_matrix"),
            encoders.ordinal_transition_field(signal, mode="transition_matrix"),
        )


# ---------------------------------------------------------------------------
# Test signals compared


def test_signal_families_have_distinguishable_transition_structure() -> None:
    """Structured, chaotic and random signals should not look alike."""

    n = 2000
    trend = np.linspace(0.0, 1.0, n)
    sine = np.sin(np.linspace(0, 200.0, n))
    noise = np.random.default_rng(0).normal(size=n)
    chaos = logistic_map(n)

    matrices = {
        name: encoders.ordinal_transition_field(sig, mode="transition_matrix")
        for name, sig in (
            ("trend", trend),
            ("sine", sine),
            ("noise", noise),
            ("chaos", chaos),
        )
    }
    # A monotonic trend visits exactly one pattern, so exactly one row is used.
    assert np.count_nonzero(matrices["trend"].sum(axis=1)) == 1
    # Noise visits every state and spreads mass broadly.
    assert np.count_nonzero(matrices["noise"].sum(axis=1)) == 6
    # Chaos visits fewer states than noise: forbidden patterns.
    assert np.count_nonzero(matrices["chaos"].sum(axis=1)) < 6
    # All four are mutually distinct.
    names = list(matrices)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not np.allclose(matrices[a], matrices[b]), f"{a} == {b}"


# ---------------------------------------------------------------------------
# Ties


def test_tie_policies() -> None:
    tied = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 1.0])

    stable = encoders.ordinal_transition_field(tied)
    assert np.all(np.isfinite(stable))

    with pytest.raises(ValueError, match="tied values"):
        encoders.ordinal_transition_field(tied, tie_policy="raise")

    first = encoders.ordinal_transition_field(tied, tie_policy="jitter", seed=1)
    again = encoders.ordinal_transition_field(tied, tie_policy="jitter", seed=1)
    np.testing.assert_array_equal(first, again)


def test_raise_policy_accepts_untied_input(sine: np.ndarray) -> None:
    assert encoders.ordinal_transition_field(sine, tie_policy="raise").shape == (126, 126)


def test_jitter_seeds_can_differ_on_tied_data() -> None:
    tied = np.repeat(np.arange(5.0), 6)
    outputs = {
        encoders.ordinal_transition_field(tied, tie_policy="jitter", seed=s).tobytes()
        for s in range(6)
    }
    assert len(outputs) > 1, "jitter never changed the tie-breaking"


# ---------------------------------------------------------------------------
# Safety and edge cases


def test_order_is_capped_because_the_state_space_is_factorial(sine: np.ndarray) -> None:
    assert encoders.MAX_ORDINAL_ORDER == 7
    with pytest.raises(ValueError, match=r"order must be in \[2, 7\]"):
        encoders.ordinal_transition_field(sine, order=8)
    with pytest.raises(ValueError, match=r"order must be in \[2, 7\]"):
        encoders.ordinal_transition_field(sine, order=1)


def test_series_too_short_for_a_transition() -> None:
    with pytest.raises(ValueError, match="at least 2 are needed"):
        encoders.ordinal_transition_field(np.arange(3.0), order=3)
    # Exactly two windows is the minimum that works.
    assert encoders.ordinal_transition_field(np.arange(4.0), order=3).shape == (2, 2)


def test_invalid_arguments(sine: np.ndarray) -> None:
    for kwargs, match in [
        ({"delay": 0}, "delay"),
        ({"tie_policy": "nope"}, "tie_policy"),
        ({"mode": "nope"}, "mode"),
        ({"image_size": 0}, "image_size"),
        ({"image_size": 10_000}, "exceeds"),
    ]:
        with pytest.raises(ValueError, match=match):
            encoders.ordinal_transition_field(sine, **kwargs)  # type: ignore[arg-type]


def test_nan_policy(sine: np.ndarray) -> None:
    dirty = sine.copy()
    dirty[10] = np.nan
    with pytest.raises(ValueError):
        encoders.ordinal_transition_field(dirty)
    assert np.all(
        np.isfinite(encoders.ordinal_transition_field(dirty, nan_policy="interpolate"))
    )


def test_deterministic(sine: np.ndarray) -> None:
    np.testing.assert_array_equal(
        encoders.ordinal_transition_field(sine),
        encoders.ordinal_transition_field(sine),
    )


def test_registry_and_metadata(sine: np.ndarray) -> None:
    from tscv_vision.representations import get_representation_info

    np.testing.assert_array_equal(
        encoders.get_encoder("otf")(sine), encoders.ordinal_transition_field(sine)
    )
    info = get_representation_info("otf")
    assert info.canonical_method is False, "this composition is not a published method"
    assert "TSCV-Vision" in info.notes
