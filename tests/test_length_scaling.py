"""Tests for the series-length scaling benchmark.

Timing is not asserted: it is not reproducible on a shared machine. What is
asserted is the structure of the sweep, the image sizes it reports, and the
exponent fit, all of which are deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscv_vision.benchmark import benchmark_length_scaling, scaling_exponent

LENGTHS = (16, 32, 64)


def test_sweep_returns_one_row_per_cell() -> None:
    rows = benchmark_length_scaling(("gaf", "rp"), lengths=LENGTHS, repeats=1)
    assert len(rows) == 2 * len(LENGTHS)
    assert [r["representation"] for r in rows[: len(LENGTHS)]] == ["gaf"] * len(LENGTHS)
    for row in rows:
        assert row["encode_seconds"] >= 0.0
        assert row["encode_peak_mib"] > 0.0
        assert row["n_features"] > 0.0


def test_quadratic_encoders_report_n_squared_pixels() -> None:
    """`gaf` and `rp` are documented ``O(N^2)`` in memory; the image proves it."""

    rows = benchmark_length_scaling(("gaf", "rp"), lengths=LENGTHS, repeats=1)
    for row in rows:
        assert row["image_values"] == float(row["length"]) ** 2


def test_features_can_be_skipped() -> None:
    rows = benchmark_length_scaling(
        ("gaf",), lengths=(16,), repeats=1, measure_features=False
    )
    (row,) = rows
    assert np.isnan(row["feature_seconds"])
    assert np.isnan(row["feature_peak_mib"])
    assert np.isnan(row["n_features"])
    # The encoder half is still measured.
    assert row["encode_peak_mib"] > 0.0


def test_the_sweep_is_reproducible_for_a_given_seed() -> None:
    """Same seed, same series, so the shape-derived columns must agree."""

    a = benchmark_length_scaling(("gaf",), lengths=LENGTHS, repeats=1, seed=7)
    b = benchmark_length_scaling(("gaf",), lengths=LENGTHS, repeats=1, seed=7)
    for x, y in zip(a, b, strict=True):
        assert x["image_values"] == y["image_values"]
        assert x["n_features"] == y["n_features"]


@pytest.mark.parametrize("power", [1.0, 2.0, 3.0])
def test_scaling_exponent_recovers_a_known_power_law(power: float) -> None:
    lengths = [128.0, 256.0, 512.0, 1024.0]
    values = [n**power for n in lengths]
    assert scaling_exponent(lengths, values) == pytest.approx(power, abs=1e-9)


def test_scaling_exponent_needs_two_usable_points() -> None:
    assert np.isnan(scaling_exponent([128.0], [1.0]))
    # Non-positive and non-finite values cannot go on a log axis and are dropped.
    assert np.isnan(scaling_exponent([128.0, 256.0], [0.0, float("nan")]))
