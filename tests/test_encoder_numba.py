"""Regression tests for compiled encoder backends.

The Numba and Cython implementations should match the pure NumPy output to
within tight numerical tolerances.  Tests are skipped when the optional
dependencies are unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscv_vision import encoders

TOLERANCE = {"rtol": 1e-6, "atol": 1e-6}


@pytest.fixture(params=["numba", "cython"])
def backend(request: pytest.FixtureRequest) -> str:
    """Return the compiled backend to exercise or skip if unavailable."""

    if request.param == "numba":
        if not encoders._HAS_NUMBA:  # pragma: no cover - optional path
            pytest.skip("numba not installed")
    else:
        if not encoders._HAS_CYTHON:  # pragma: no cover - optional path
            pytest.skip("cython extension not built")
    return request.param


@pytest.mark.parametrize("n,method", [(32, "summation"), (33, "difference")])
def test_gaf_compiled_matches(n: int, method: str, backend: str) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    base = encoders.gaf(x, method=method)
    compiled = encoders.gaf(x, method=method, **{f"use_{backend}": True})
    np.testing.assert_allclose(compiled, base, **TOLERANCE)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("eps", [None, 0.3])
def test_recurrence_compiled_matches(metric: str, eps: float | None, backend: str) -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.0, 1.0, 20)
    base = encoders.recurrence_plot(x, metric=metric, eps=eps)
    compiled = encoders.recurrence_plot(
        x, metric=metric, eps=eps, **{f"use_{backend}": True}
    )
    np.testing.assert_allclose(compiled, base, **TOLERANCE)


@pytest.mark.parametrize("win,hop", [(16, 4), (32, 16)])
def test_spectrogram_compiled_matches(win: int, hop: int, backend: str) -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=128)
    base = encoders.spectrogram(x, win=win, hop=hop)
    compiled = encoders.spectrogram(x, win=win, hop=hop, **{f"use_{backend}": True})
    np.testing.assert_allclose(compiled, base, **TOLERANCE)

