import numpy as np
import pytest

from tscv_vision import encoders

numba = pytest.importorskip("numba")


def test_numba_gaf_matches() -> None:
    x = np.linspace(-1.0, 1.0, 32)
    np.testing.assert_allclose(
        encoders.gaf(x), encoders.gaf(x, use_numba=True)
    )


def test_numba_recurrence_matches() -> None:
    x = np.linspace(0.0, 1.0, 16)
    np.testing.assert_allclose(
        encoders.recurrence_plot(x),
        encoders.recurrence_plot(x, use_numba=True),
    )


def test_numba_spectrogram_matches() -> None:
    x = np.sin(np.linspace(0.0, 4 * np.pi, 64))
    np.testing.assert_allclose(
        encoders.spectrogram(x, win=16, hop=8),
        encoders.spectrogram(x, win=16, hop=8, use_numba=True),
        rtol=1e-6,
        atol=1e-6,
    )
