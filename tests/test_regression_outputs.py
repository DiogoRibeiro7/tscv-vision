from __future__ import annotations

import numpy as np

from tscv_vision import encoders


def test_known_gaf_output() -> None:
    x = np.array([1.0, 1.0])
    img = encoders.gaf(x)
    expected = -np.ones((2, 2))
    np.testing.assert_allclose(img, expected)


def test_known_rp_output() -> None:
    x = np.array([0.0, 1.0])
    rp = encoders.recurrence_plot(x)
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(rp, expected, atol=1e-10)


def test_known_spectrogram_output() -> None:
    x = np.zeros(8)
    spec = encoders.spectrogram(x, win=8, hop=4, window="rect")
    expected = np.zeros((5, 1))
    np.testing.assert_allclose(spec, expected)
