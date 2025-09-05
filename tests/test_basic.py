import numpy as np

from tscv_vision import encoders, features


def test_basic_gaf_and_features() -> None:
    x = np.sin(np.linspace(0, 2 * np.pi, 64))
    img = encoders.gaf(x)
    feats = features.intensity_stats(img)
    assert img.shape == (64, 64)
    assert feats.shape == (6,)
