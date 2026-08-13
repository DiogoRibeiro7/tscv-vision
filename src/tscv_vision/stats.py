"""Small, dependency-free statistical tests used across the package.

The core package is NumPy-only, so the distribution functions needed by
:mod:`tscv_vision.analytics` and :mod:`tscv_vision.benchmark` are implemented
here rather than pulled in from SciPy. Every routine is validated against
``scipy.stats`` in ``tests/test_reference_equivalence.py`` (marked ``optional``).

References
----------
Press et al., *Numerical Recipes*, 3rd ed., §6.2 and §6.4 (incomplete beta and
gamma functions).  Demšar (2006), "Statistical Comparisons of Classifiers over
Multiple Data Sets", JMLR 7:1-30 (Wilcoxon, Friedman and Nemenyi procedures).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

_FPMIN = 1e-300
_EPS = 3e-16
_ITMAX = 300


# ---------------------------------------------------------------------------
# Special functions


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta function."""

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _ITMAX + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function :math:`I_x(a, b)`.

    Raises
    ------
    ValueError
        If ``a``/``b`` are not positive or ``x`` lies outside ``[0, 1]``.
    """

    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _gamma_series(a: float, x: float) -> float:
    """Lower regularized incomplete gamma ``P(a, x)`` via its series."""

    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(_ITMAX):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a: float, x: float) -> float:
    """Upper regularized incomplete gamma ``Q(a, x)`` via a continued fraction."""

    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammainc_upper(a: float, x: float) -> float:
    """Upper regularized incomplete gamma function :math:`Q(a, x)`."""

    if a <= 0.0:
        raise ValueError("a must be positive")
    if x < 0.0:
        raise ValueError("x must be non-negative")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gamma_series(a, x)
    return _gamma_cf(a, x)


def normal_sf(z: float) -> float:
    """Upper tail of the standard normal distribution."""

    return 0.5 * math.erfc(z / math.sqrt(2.0))


def student_t_sf(t: float, df: float) -> float:
    """Upper tail :math:`P(T > t)` of Student's t distribution with ``df``."""

    if df <= 0.0:
        raise ValueError("df must be positive")
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0
    half = 0.5 * betainc(0.5 * df, 0.5, df / (df + t * t))
    return half if t > 0 else 1.0 - half


def chi2_sf(x: float, df: float) -> float:
    """Upper tail :math:`P(X > x)` of the chi-square distribution."""

    if df <= 0.0:
        raise ValueError("df must be positive")
    if x <= 0.0:
        return 1.0
    return gammainc_upper(0.5 * df, 0.5 * x)


# ---------------------------------------------------------------------------
# Tests


class TTestResult(NamedTuple):
    """Result of :func:`welch_ttest`."""

    statistic: float
    pvalue: float
    df: float


def welch_ttest(a: Array, b: Array) -> TTestResult:
    """Two-sided Welch (unequal-variance) t-test.

    Unlike a normal-approximation z-test, the p-value is obtained from the
    Student t distribution with Welch--Satterthwaite degrees of freedom

    .. math::

        \\nu = \\frac{(s_a^2/n_a + s_b^2/n_b)^2}
                     {\\frac{(s_a^2/n_a)^2}{n_a-1} + \\frac{(s_b^2/n_b)^2}{n_b-1}}

    so it is exact for small samples rather than only asymptotically valid.

    Parameters
    ----------
    a, b:
        1D samples with at least two observations each.

    Returns
    -------
    TTestResult
        ``(statistic, pvalue, df)``.

    Raises
    ------
    ValueError
        If the inputs are not 1D or have fewer than two observations.

    Examples
    --------
    >>> res = welch_ttest(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    >>> round(res.df, 3)
    4.0
    """

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("inputs must be 1D")
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        raise ValueError("each sample needs at least two observations")
    vx = float(x.var(ddof=1)) / nx
    vy = float(y.var(ddof=1)) / ny
    denom = vx + vy
    if denom <= 0.0:
        # Both samples are constant: no evidence of a difference either way.
        return TTestResult(0.0, 1.0, float(nx + ny - 2))
    t = (float(x.mean()) - float(y.mean())) / math.sqrt(denom)
    df = denom**2 / (vx**2 / (nx - 1) + vy**2 / (ny - 1))
    p = 2.0 * student_t_sf(abs(t), df)
    return TTestResult(float(t), float(min(1.0, p)), float(df))


class WilcoxonResult(NamedTuple):
    """Result of :func:`wilcoxon_signed_rank`."""

    statistic: float
    pvalue: float
    method: str


def _rankdata(values: Array) -> Array:
    """Average ranks (1-based) of ``values``, ties share their mean rank."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def _signed_rank_exact_sf(w: float, n: int) -> float:
    """``P(W+ <= w)`` under the null, by exact enumeration of rank subsets."""

    total = n * (n + 1) // 2
    counts = np.zeros(total + 1, dtype=float)
    counts[0] = 1.0
    for rank in range(1, n + 1):
        shifted = np.zeros_like(counts)
        shifted[rank:] = counts[:-rank]
        counts += shifted
    cutoff = int(math.floor(w))
    return float(counts[: cutoff + 1].sum() / 2.0**n)


def wilcoxon_signed_rank(x: Array, y: Array | None = None) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test for paired samples.

    This is the recommended pairwise test for comparing two methods over many
    datasets (Demšar, 2006), because it does not assume normality or
    commensurability of the per-dataset scores.

    Parameters
    ----------
    x:
        Paired differences, or the first sample when ``y`` is given.
    y:
        Optional second sample; ``x - y`` is then tested.

    Returns
    -------
    WilcoxonResult
        ``(statistic, pvalue, method)`` where ``statistic`` is ``min(W+, W-)``
        and ``method`` is ``"exact"`` or ``"normal"``.

    Raises
    ------
    ValueError
        If shapes mismatch or every difference is zero.
    """

    d = np.asarray(x, dtype=float)
    if y is not None:
        d = d - np.asarray(y, dtype=float)
    if d.ndim != 1:
        raise ValueError("inputs must be 1D")
    d = d[d != 0.0]
    n = d.size
    if n == 0:
        raise ValueError("all differences are zero; the test is undefined")
    ranks = _rankdata(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    stat = min(w_plus, w_minus)

    _, tie_counts = np.unique(np.abs(d), return_counts=True)
    has_ties = bool(np.any(tie_counts > 1))
    if n <= 25 and not has_ties:
        p = 2.0 * _signed_rank_exact_sf(stat, n)
        return WilcoxonResult(stat, float(min(1.0, p)), "exact")

    mean = n * (n + 1) / 4.0
    tie_term = float(np.sum(tie_counts**3 - tie_counts))
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0.0:
        return WilcoxonResult(stat, 1.0, "normal")
    z = (stat - mean) / math.sqrt(var)
    p = 2.0 * normal_sf(abs(z))
    return WilcoxonResult(stat, float(min(1.0, p)), "normal")


def average_ranks(scores: Array, *, higher_is_better: bool = True) -> Array:
    """Average rank of each method over datasets.

    Parameters
    ----------
    scores:
        ``(n_datasets, n_methods)`` matrix of scores.
    higher_is_better:
        If ``True`` the largest score gets rank 1.

    Returns
    -------
    ndarray
        ``(n_methods,)`` mean ranks; ties share their average rank.
    """

    mat = np.asarray(scores, dtype=float)
    if mat.ndim != 2:
        raise ValueError("scores must be 2D (n_datasets, n_methods)")
    signed = -mat if higher_is_better else mat
    ranks = np.vstack([_rankdata(row) for row in signed])
    return np.asarray(ranks.mean(axis=0), dtype=float)


class FriedmanResult(NamedTuple):
    """Result of :func:`friedman_test`."""

    statistic: float
    pvalue: float
    ranks: Array


def friedman_test(scores: Array, *, higher_is_better: bool = True) -> FriedmanResult:
    """Friedman test for differences among methods across datasets.

    Parameters
    ----------
    scores:
        ``(n_datasets, n_methods)`` score matrix.
    higher_is_better:
        Ranking direction, see :func:`average_ranks`.

    Returns
    -------
    FriedmanResult
        Chi-square statistic with ``n_methods - 1`` degrees of freedom, its
        p-value, and the average ranks.

    Raises
    ------
    ValueError
        If fewer than two methods or two datasets are supplied.
    """

    mat = np.asarray(scores, dtype=float)
    if mat.ndim != 2:
        raise ValueError("scores must be 2D (n_datasets, n_methods)")
    n, k = mat.shape
    if k < 2 or n < 2:
        raise ValueError("need at least two methods and two datasets")
    signed = -mat if higher_is_better else mat
    rank_rows = np.vstack([_rankdata(row) for row in signed])
    ranks = np.asarray(rank_rows.mean(axis=0), dtype=float)
    rank_sums = np.asarray(rank_rows.sum(axis=0), dtype=float)
    stat = (12.0 / (n * k * (k + 1.0))) * float(np.sum(rank_sums**2))
    stat -= 3.0 * n * (k + 1.0)

    tie_sum = 0.0
    for row in signed:
        _, counts = np.unique(row, return_counts=True)
        tie_sum += float(np.sum(counts**3 - counts))
    correction = 1.0 - tie_sum / (n * (k**3 - k))
    stat = 0.0 if correction <= 0.0 else stat / correction
    return FriedmanResult(float(stat), float(chi2_sf(stat, k - 1)), ranks)


# Studentized range statistic q_alpha / sqrt(2) for the Nemenyi test, indexed by
# the number of compared methods (Demšar 2006, Table 5).
_NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
    15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}
_NEMENYI_Q10 = {
    2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780,
    9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077, 14: 3.120,
    15: 3.159, 16: 3.196, 17: 3.230, 18: 3.261, 19: 3.291, 20: 3.319,
}


def nemenyi_critical_difference(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Critical difference of average ranks for the Nemenyi post-hoc test.

    Two methods differ significantly when their average ranks differ by more
    than the returned value.

    Raises
    ------
    ValueError
        If ``alpha`` is unsupported or ``n_methods`` is outside ``[2, 20]``.
    """

    table = {0.05: _NEMENYI_Q05, 0.10: _NEMENYI_Q10}.get(round(alpha, 2))
    if table is None:
        raise ValueError("alpha must be 0.05 or 0.10")
    if n_methods not in table:
        raise ValueError("n_methods must be between 2 and 20")
    if n_datasets < 1:
        raise ValueError("n_datasets must be positive")
    q = table[n_methods]
    return float(q * math.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets)))


def holm_bonferroni(pvalues: Array) -> Array:
    """Holm--Bonferroni step-down adjustment of ``pvalues``."""

    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be 1D")
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


__all__ = [
    "TTestResult",
    "WilcoxonResult",
    "FriedmanResult",
    "betainc",
    "gammainc_upper",
    "normal_sf",
    "student_t_sf",
    "chi2_sf",
    "welch_ttest",
    "wilcoxon_signed_rank",
    "average_ranks",
    "friedman_test",
    "nemenyi_critical_difference",
    "holm_bonferroni",
]
