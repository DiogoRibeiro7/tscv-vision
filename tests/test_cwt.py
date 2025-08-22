import numpy as np

from tscv_vision import encoders, sliding


def test_cwt_shape_and_sliding():
    x = np.sin(np.linspace(0, 2 * np.pi, 32))
    scales = np.array([1, 2, 4])
    img = encoders.cwt(x, scales)
    assert img.shape == (3, 32)
    imgs, _ = sliding.encode_sliding(x, encoder="cwt", size=16, hop=16, cwt_scales=scales)
    assert imgs.shape == (2, 3, 16)

