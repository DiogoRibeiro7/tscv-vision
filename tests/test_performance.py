from __future__ import annotations

import sys
import timeit

import numpy as np

from tscv_vision import features
from tscv_vision.sliding import encode_sliding, sliding_windows


def test_extract_batch_speed() -> None:
    rng = np.random.default_rng(0)
    imgs = rng.normal(size=(20, 32, 32))

    def baseline() -> np.ndarray:
        return np.vstack([features.extract_feature_vector(im, bins=8) for im in imgs])

    t_base = timeit.timeit(baseline, number=3)
    t_opt = timeit.timeit(lambda: features.extract_batch(imgs, bins=8), number=3)
    assert t_opt <= t_base


def test_encode_sliding_lazy_memory() -> None:
    x = np.sin(np.linspace(0.0, 4 * np.pi, 512))
    imgs, _ = encode_sliding(x, size=64, hop=32)
    gen = encode_sliding(x, size=64, hop=32, lazy=True)
    assert sys.getsizeof(gen) < imgs.nbytes


def test_sliding_windows_speed() -> None:
    x = np.arange(10000, dtype=float)

    def baseline() -> np.ndarray:
        return np.array([x[i : i + 64] for i in range(0, len(x) - 64 + 1, 32)])

    t_base = timeit.timeit(baseline, number=1)
    t_opt = timeit.timeit(lambda: sliding_windows(x, size=64, hop=32), number=1)
    assert t_opt <= t_base
