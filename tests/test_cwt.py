import numpy as np
import pytest

from tscv_vision import encoders, sliding


def test_cwt_shape_and_sliding() -> None:
    x = np.sin(np.linspace(0, 2 * np.pi, 32))
    scales = np.array([1, 2, 4])
    img = encoders.cwt(x, scales)
    assert img.shape == (3, 32)
    imgs, _ = sliding.encode_sliding(x, encoder="cwt", size=16, hop=16, cwt_scales=scales)
    assert imgs.shape == (2, 3, 16)


def test_cwt_morlet_output_is_finite_and_normalised() -> None:
    x = np.sin(np.linspace(0, 6 * np.pi, 64))
    img = encoders.cwt(x, np.array([1.0, 2.0, 4.0, 8.0]))
    assert np.all(np.isfinite(img))
    assert np.min(img) >= 0.0
    assert np.max(img) <= 1.0
    assert np.max(img) > 0.99
    assert not np.allclose(img[0], img[-1])


def test_cwt_zero_signal_has_zero_energy() -> None:
    img = encoders.cwt(np.zeros(32), np.array([1.0, 2.0, 4.0]))
    np.testing.assert_allclose(img, 0.0)


@pytest.mark.parametrize("scales", [np.array([[1.0, 2.0]]), np.array([0.0, 2.0])])
def test_cwt_rejects_invalid_scales(scales: np.ndarray) -> None:
    with pytest.raises(ValueError, match="positive"):
        encoders.cwt(np.arange(8.0), scales)


def test_cwt_extra_wavelet() -> None:
    pytest.importorskip("pywt")
    x = np.sin(np.linspace(0, 2 * np.pi, 32))
    scales = np.array([1, 2])
    img = encoders.cwt(x, scales, wavelet="mexh")
    assert img.shape == (2, 32)

