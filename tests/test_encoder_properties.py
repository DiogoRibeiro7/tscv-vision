from __future__ import annotations

import numpy as np

from tscv_vision import encoders


def test_gaf_symmetry_property() -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.normal(size=16)
        img = encoders.gaf(x)
        assert np.allclose(img, img.T)
        assert np.all((img >= -1.0) & (img <= 1.0))


def test_recurrence_plot_symmetric_diag() -> None:
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.normal(size=20)
        rp = encoders.recurrence_plot(x)
        assert np.allclose(rp, rp.T)
        assert np.allclose(np.diag(rp), 1.0)


def test_spectrogram_scale_invariance() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=128)
    spec1 = encoders.spectrogram(x, win=32, hop=16)
    spec2 = encoders.spectrogram(2 * x, win=32, hop=16)
    np.testing.assert_allclose(spec1, spec2)
    assert np.all((spec1 >= 0.0) & (spec1 <= 1.0))
