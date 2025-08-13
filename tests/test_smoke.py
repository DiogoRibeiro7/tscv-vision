from __future__ import annotations
import numpy as np
from tscv_vision import encoders, features


def test_end_to_end():
    x = np.sin(np.linspace(0, 8*np.pi, 256)) + 0.05*np.random.randn(256)
    img = encoders.gaf(x)
    vec = features.extract_feature_vector(img, bins=16)
    assert vec.ndim == 1
    assert vec.size == (6 + 16 + 16 + 256)
