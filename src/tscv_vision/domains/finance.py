"""Finance-specific encoders and features."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Simple pre-trained logistic model coefficients for regime detection
REGIME_COEFFS: Array = np.array([0.0, 50.0])


def microstructure_features(prices: Array) -> Array:
    """Compute basic market microstructure features.

    Parameters
    ----------
    prices : Array
        Price series of shape ``(N,)``.

    Returns
    -------
    Array
        Features ``[mean_return, volatility, max_drawdown]``.

    Raises
    ------
    ValueError
        If ``prices`` is not 1D or has fewer than two points.
    """
    pr = np.asarray(prices, dtype=float)
    if pr.ndim != 1 or pr.size < 2:
        raise ValueError("prices must be 1D with at least two points")
    returns = np.diff(pr) / pr[:-1]
    mean_ret = returns.mean()
    volatility = returns.std()
    drawdown = np.min(pr / np.maximum.accumulate(pr) - 1.0)
    return np.array([mean_ret, volatility, drawdown])


def volatility_clustering(prices: Array, short: int = 5, long: int = 20) -> float:
    """Estimate volatility clustering via short/long variance ratio."""
    if long <= short:
        raise ValueError("long window must exceed short window")
    pr = np.asarray(prices, dtype=float)
    returns = np.diff(pr)
    if returns.size < long:
        raise ValueError("series too short for given windows")
    short_var = returns[-short:].var()
    long_var = returns[-long:].var()
    return 0.0 if long_var == 0 else float(short_var / long_var)


def detect_regime(prices: Array, coeffs: Array = REGIME_COEFFS) -> int:
    """Classify bull (1) or bear (0) regime using a logistic model."""
    mean_ret = microstructure_features(prices)[0]
    score = coeffs[0] + coeffs[1] * mean_ret
    return int(score > 0)


def generate_price_series(
    n: int = 100,
    drift: float = 0.001,
    volatility: float = 0.01,
    *,
    seed: int = 0,
) -> Array:
    """Generate a synthetic geometric random walk for prices."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, volatility, size=n)
    return np.asarray(100.0 * np.exp(np.cumsum(returns)), dtype=float)


def augment_regime_switch(
    prices: Array,
    switch_point: int | None = None,
    factor: float = -1.0,
) -> Array:
    """Simulate a regime switch by flipping return signs after ``switch_point``."""
    pr = np.asarray(prices, dtype=float)
    if pr.ndim != 1:
        raise ValueError("prices must be 1D")
    n = pr.size
    sp = switch_point if switch_point is not None else n // 2
    if not 0 < sp < n:
        raise ValueError("invalid switch_point")
    log_ret = np.log(pr[1:] / pr[:-1])
    log_ret[sp - 1 :] *= factor
    augmented = pr[0] * np.exp(np.cumsum(np.concatenate([[0.0], log_ret])))
    return np.asarray(augmented, dtype=float)
