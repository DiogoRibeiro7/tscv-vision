from __future__ import annotations
import numpy as np
from tscv_vision.sliding import encode_sliding


def test_sliding_gaf_basic():
    x = np.sin(np.linspace(0, 12*np.pi, 512))
    images, starts = encode_sliding(x, encoder="gaf", size=64, hop=32)
    assert images.ndim == 3
    n = images.shape[0]
    assert n == 1 + (len(x) - 64) // 32
    assert images.shape[1] == images.shape[2] == 64
    assert starts.shape == (n,)


def test_sliding_spec_dims():
    x = np.sin(np.linspace(0, 10*np.pi, 300))
    images, _ = encode_sliding(x, encoder="spec", size=100, hop=50, spec_win=32, spec_hop=8)
    assert images.ndim == 3
    assert images.shape[1] == 32 // 2 + 1
