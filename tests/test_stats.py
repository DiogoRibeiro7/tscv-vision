"""Behavioural tests for :mod:`tscv_vision.stats` (no SciPy required).

Numerical equivalence with ``scipy.stats`` lives in
``tests/test_reference_equivalence.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscv_vision import stats
from tscv_vision.analytics import cross_correlation_lag, group_significance


def test_betainc_boundaries_and_symmetry() -> None:
    assert stats.betainc(2.0, 3.0, 0.0) == 0.0
    assert stats.betainc(2.0, 3.0, 1.0) == 1.0
    # I_x(a, b) = 1 - I_{1-x}(b, a)
    for x in (0.1, 0.37, 0.9):
        assert stats.betainc(2.5, 4.0, x) == pytest.approx(
            1.0 - stats.betainc(4.0, 2.5, 1.0 - x), rel=1e-12
        )
    with pytest.raises(ValueError):
        stats.betainc(0.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        stats.betainc(1.0, 1.0, 1.5)


def test_student_t_is_symmetric_and_approaches_normal() -> None:
    for t in (0.5, 1.0, 3.0):
        assert stats.student_t_sf(t, 10.0) == pytest.approx(
            1.0 - stats.student_t_sf(-t, 10.0), rel=1e-12
        )
    assert stats.student_t_sf(0.0, 7.0) == pytest.approx(0.5, rel=1e-12)
    # Large df converges to the standard normal.
    assert stats.student_t_sf(1.96, 1e6) == pytest.approx(stats.normal_sf(1.96), abs=1e-6)
    with pytest.raises(ValueError):
        stats.student_t_sf(1.0, 0.0)


def test_chi2_sf_edges() -> None:
    assert stats.chi2_sf(0.0, 3.0) == 1.0
    assert stats.chi2_sf(-1.0, 3.0) == 1.0
    assert 0.0 < stats.chi2_sf(100.0, 3.0) < 1e-15
    with pytest.raises(ValueError):
        stats.chi2_sf(1.0, 0.0)


def test_welch_ttest_uses_satterthwaite_df() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
    res = stats.welch_ttest(a, b)
    # Very unequal variances pull df well below n_a + n_b - 2 = 8.
    assert 4.0 <= res.df < 5.0
    assert res.pvalue < 0.05
    assert res.statistic < 0.0


def test_welch_ttest_identical_samples() -> None:
    x = np.array([1.0, 2.0, 3.0])
    res = stats.welch_ttest(x, x)
    assert res.statistic == 0.0
    assert res.pvalue == pytest.approx(1.0)
    const = stats.welch_ttest(np.zeros(4), np.zeros(4))
    assert const.pvalue == 1.0


def test_welch_ttest_validation() -> None:
    with pytest.raises(ValueError, match="1D"):
        stats.welch_ttest(np.zeros((2, 2)), np.zeros(4))
    with pytest.raises(ValueError, match="two observations"):
        stats.welch_ttest(np.array([1.0]), np.array([1.0, 2.0]))


def test_group_significance_delegates_to_welch() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    t, p = group_significance(a, b)
    ref = stats.welch_ttest(a, b)
    assert (t, p) == (ref.statistic, ref.pvalue)


def test_wilcoxon_exact_and_normal_branches() -> None:
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    exact = stats.wilcoxon_signed_rank(d)
    assert exact.method == "exact"
    assert exact.statistic == 0.0
    assert exact.pvalue == pytest.approx(2.0 / 2**6)

    tied = np.array([1.0, 1.0, 2.0, -2.0, 3.0, -3.0])
    assert stats.wilcoxon_signed_rank(tied).method == "normal"
    assert stats.wilcoxon_signed_rank(np.arange(1.0, 40.0)).method == "normal"


def test_wilcoxon_ignores_zero_differences() -> None:
    with_zeros = stats.wilcoxon_signed_rank(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    without = stats.wilcoxon_signed_rank(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert with_zeros.pvalue == pytest.approx(without.pvalue)
    with pytest.raises(ValueError, match="all differences are zero"):
        stats.wilcoxon_signed_rank(np.zeros(5))


def test_average_ranks_handles_ties() -> None:
    scores = np.array([[1.0, 1.0, 0.0], [0.5, 0.2, 0.9]])
    ranks = stats.average_ranks(scores)
    # Row 0: two-way tie for first -> 1.5, 1.5, 3.  Row 1: 2, 3, 1.
    np.testing.assert_allclose(ranks, [(1.5 + 2) / 2, (1.5 + 3) / 2, (3 + 1) / 2])
    lower = stats.average_ranks(scores, higher_is_better=False)
    np.testing.assert_allclose(lower + ranks, 4.0)


def test_friedman_detects_a_consistent_winner() -> None:
    n = 12
    scores = np.column_stack(
        [np.full(n, 0.9), np.full(n, 0.6), np.linspace(0.1, 0.3, n)]
    )
    res = stats.friedman_test(scores)
    assert res.pvalue < 1e-4
    np.testing.assert_allclose(res.ranks, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="at least two"):
        stats.friedman_test(np.zeros((1, 3)))


def test_friedman_applies_tie_correction() -> None:
    scores = np.array(
        [
            [0.90, 0.90, 0.10, 0.10],
            [0.80, 0.70, 0.70, 0.20],
            [0.75, 0.75, 0.60, 0.40],
            [0.95, 0.85, 0.85, 0.10],
            [0.70, 0.70, 0.70, 0.30],
        ]
    )
    res = stats.friedman_test(scores)
    uncorrected = (12.0 * scores.shape[0] / (scores.shape[1] * (scores.shape[1] + 1.0))) * (
        float(np.sum(res.ranks**2)) - scores.shape[1] * (scores.shape[1] + 1.0) ** 2 / 4.0
    )
    assert res.statistic > uncorrected
    assert res.pvalue < stats.chi2_sf(uncorrected, scores.shape[1] - 1)


def test_nemenyi_critical_difference_shrinks_with_more_datasets() -> None:
    small = stats.nemenyi_critical_difference(5, 10)
    large = stats.nemenyi_critical_difference(5, 100)
    assert large < small
    assert stats.nemenyi_critical_difference(5, 10, alpha=0.10) < small
    with pytest.raises(ValueError, match="alpha"):
        stats.nemenyi_critical_difference(5, 10, alpha=0.01)
    with pytest.raises(ValueError, match="n_methods"):
        stats.nemenyi_critical_difference(50, 10)


def test_holm_bonferroni_is_monotone_and_bounded() -> None:
    adjusted = stats.holm_bonferroni(np.array([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
    assert np.all(adjusted <= 1.0)
    np.testing.assert_allclose(stats.holm_bonferroni(np.array([0.9, 0.95])), [1.0, 1.0])


def test_cross_correlation_lag_recovers_a_known_shift() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    lag = 5
    x = base
    y = np.roll(base, lag)
    # x leads y by `lag`.
    assert cross_correlation_lag(x[lag:], y[lag:], max_lag=10) == lag
    with pytest.raises(ValueError, match="max_lag"):
        cross_correlation_lag(x, y, max_lag=len(x))
    with pytest.raises(ValueError, match="equal length"):
        cross_correlation_lag(x, y[:-1])


def test_cross_causal_lag_alias_is_deprecated() -> None:
    from tscv_vision.analytics import cross_causal_lag

    rng = np.random.default_rng(1)
    x = rng.normal(size=64)
    with pytest.warns(DeprecationWarning, match="cross_correlation_lag"):
        got = cross_causal_lag(x, x, max_lag=3)
    assert got == 0
