from __future__ import annotations

import numpy as np
import pytest

from tscv_vision import encoders


def test_gaf_modes() -> None:
    x = np.linspace(-1.0, 1.0, 8)
    img_sum = encoders.gaf(x, method="summation")
    img_diff = encoders.gaf(x, method="difference")
    assert img_sum.shape == (8, 8)
    assert img_diff.shape == (8, 8)
    assert np.all(img_sum <= 1.0)


def test_recurrence_plot_binary() -> None:
    x = np.array([0.0, 1.0, 2.0])
    rp = encoders.recurrence_plot(x, metric="manhattan", eps=0.5)
    assert rp.shape == (3, 3)
    assert set(np.unique(rp)) <= {0.0, 1.0}


def test_spectrogram_dims() -> None:
    x = np.sin(np.linspace(0.0, 4 * np.pi, 128))
    spec = encoders.spectrogram(x, win=32, hop=16)
    expected = int(np.ceil((len(x) - 32) / 16)) + 1
    assert spec.shape == (17, expected)
    assert spec.max() <= 1.0 + 1e-12


def test_spectrogram_pads_last_frame() -> None:
    x = np.sin(np.linspace(0.0, 4 * np.pi, 130))
    spec = encoders.spectrogram(x, win=32, hop=16)
    expected = int(np.ceil((len(x) - 32) / 16)) + 1
    assert spec.shape == (17, expected)


def test_encoder_validations_and_modes() -> None:
    x = np.linspace(-1.0, 1.0, 16)
    # constant input triggers zero scaling branch
    const = encoders.gaf(np.ones(4))
    assert const.shape == (4, 4)
    # non-finite values raise
    with pytest.raises(ValueError):
        encoders.gaf(np.array([0.0, np.inf]))
    # invalid GAF method
    with pytest.raises(ValueError):
        encoders.gaf(x, method="foo")
    # invalid metric
    with pytest.raises(ValueError):
        encoders.recurrence_plot(x, metric="foo")
    # spectrogram validations
    with pytest.raises(ValueError):
        encoders.spectrogram(x, win=4)
    with pytest.raises(ValueError):
        encoders.spectrogram(np.arange(8.0), win=16)
    x_long = np.linspace(-1.0, 1.0, 64)
    with pytest.raises(ValueError):
        encoders.spectrogram(x_long, win=32, window="foo")
    # default hop and rect window branches
    encoders.spectrogram(x, win=16)
    encoders.spectrogram(x_long, win=32, hop=8, window="rect")


def test_register_encoder() -> None:
    def dummy(x: np.ndarray) -> np.ndarray:
        return encoders.gaf(x)

    encoders.register_encoder("dummy", dummy)
    assert "dummy" in encoders.ENCODER_REGISTRY

