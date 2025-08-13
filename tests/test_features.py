from __future__ import annotations

import numpy as np

from tscv_vision import features


def test_histogram_constant() -> None:
    img = np.zeros((4, 4))
    h = features.histogram(img, bins=4)
    assert h.shape == (4,)
    assert h[0] == 1.0


def test_gradient_histogram_constant() -> None:
    img = np.ones((4, 4))
    g = features.gradient_histogram(img, bins=8)
    assert g.shape == (8,)
    assert g[0] == 1.0


def test_lbp_constant() -> None:
    img = np.full((4, 4), 5.0)
    h = features.lbp(img)
    assert h.shape == (256,)
    assert np.isclose(h[-1], 1.0)


def test_extract_feature_vector() -> None:
    img = np.arange(16, dtype=float).reshape(4, 4)
    vec = features.extract_feature_vector(img, bins=8)
    assert vec.shape == (6 + 8 + 16 + 256,)


def test_extract_feature_vector_multichannel() -> None:
    base = np.arange(16, dtype=float).reshape(4, 4)
    img = np.stack([base, base * 0], axis=-1)
    vec = features.extract_feature_vector(img, bins=8)
    assert vec.shape[0] == 2 * (6 + 8 + 16 + 256)

