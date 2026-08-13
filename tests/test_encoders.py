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


def test_non_finite_inputs_raise() -> None:
    arr = np.array([0.0, np.nan])
    with pytest.raises(ValueError):
        encoders.gaf(arr)
    with pytest.raises(ValueError):
        encoders.recurrence_plot(arr)
    with pytest.raises(ValueError):
        encoders.spectrogram(arr, win=8)


def test_short_constant_and_extreme_signals() -> None:
    const_short = np.ones(2)
    assert encoders.gaf(const_short).shape == (2, 2)
    assert encoders.recurrence_plot(const_short).shape == (2, 2)
    const_long = np.ones(16)
    spec = encoders.spectrogram(const_long, win=8, hop=4)
    assert spec.shape[1] > 0 and np.isfinite(spec).all()
    # very short signal length 1
    single = np.array([5.0])
    assert encoders.gaf(single).shape == (1, 1)
    assert encoders.recurrence_plot(single).shape == (1, 1)


def test_param_validations() -> None:
    x = np.arange(8.0)
    with pytest.raises(ValueError):
        encoders.recurrence_plot(x, eps=-0.1)
    with pytest.raises(ValueError):
        encoders.recurrence_plot(x, eps=1.1)
    with pytest.raises(ValueError):
        encoders.recurrence_plot(x, eps=np.nan)
    with pytest.raises(ValueError):
        encoders.spectrogram(x, win=8, hop=0)
    with pytest.raises(ValueError):
        encoders.spectrogram(x, win=8, hop=16)


def test_extreme_value_range() -> None:
    pair = np.array([1e300, -1e300])
    assert np.isfinite(encoders.gaf(pair)).all()
    assert np.isfinite(encoders.recurrence_plot(pair)).all()
    spec = encoders.spectrogram(np.tile(pair, 4), win=8)
    assert np.isfinite(spec).all()
    with pytest.raises(ValueError, match="range is too large"):
        encoders.window_attention(np.tile(pair, 4), window=2)
    with pytest.raises(ValueError, match="range is too large"):
        encoders.shapelet_transform(np.tile(pair, 4), k=2, length=2, seed=0)


def test_register_encoder() -> None:
    def dummy(x: np.ndarray) -> np.ndarray:
        return encoders.gaf(x)

    encoders.register_encoder("dummy", dummy)
    assert "dummy" in encoders.ENCODER_REGISTRY


@pytest.mark.parametrize("name", ["cwt", "msrp", "mp", "matrix_profile"])
def test_parameterized_registry_entries_have_defaults(name: str) -> None:
    x = np.sin(np.linspace(0.0, 4 * np.pi, 64))
    out = encoders.get_encoder(name)(x)
    assert out.size > 0
    assert np.isfinite(out).all()

