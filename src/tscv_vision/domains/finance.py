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
