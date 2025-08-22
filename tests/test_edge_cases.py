from __future__ import annotations

import numpy as np
import pytest

from tscv_vision import features
from tscv_vision.sliding import encode_sliding, sliding_windows


def test_sliding_windows_empty_and_single() -> None:
    with pytest.raises(ValueError):
        sliding_windows(np.array([]), size=4)
    with pytest.raises(ValueError):
        sliding_windows(np.array([1.0]), size=1)
    view = sliding_windows(np.array([1.0, 2.0]), size=2)
    assert view.shape == (1, 2)


def test_encode_sliding_large_array() -> None:
    x = np.sin(np.linspace(0.0, 20 * np.pi, 10000))
    imgs, starts = encode_sliding(x, size=256, hop=128, encoder="rp")
    assert imgs.shape[0] == starts.shape[0]


def test_extract_batch_empty_and_extreme() -> None:
    empty = np.empty((0, 4, 4))
    vecs = features.extract_batch(empty)
    assert vecs.shape == (0, 0)
    huge = np.full((4, 4), 1e6)
    vec = features.extract_feature_vector(huge)
    assert np.isfinite(vec).all()
