from __future__ import annotations

import numpy as np

from tscv_vision import features


def test_extract_batch_matches_single() -> None:
    rng = np.random.default_rng(0)
    imgs = rng.normal(size=(3, 8, 8))
    batch = features.extract_batch(imgs, bins=8)
    assert batch.shape[0] == 3
    single = np.vstack([features.extract_feature_vector(im, bins=8) for im in imgs])
    np.testing.assert_allclose(batch, single)


def test_extract_batch_empty() -> None:
    empty = np.zeros((0, 4, 4))
    batch = features.extract_batch(empty, bins=4)
    assert batch.shape == (0, 0)


def test_extract_batch_multichannel() -> None:
    img = np.stack([np.eye(4), np.eye(4)], axis=-1)
    imgs = np.stack([img, img], axis=0)
    batch = features.extract_batch(imgs, bins=8)
    assert batch.shape[0] == 2
    assert batch.shape[1] == 2 * (6 + 8 + 16 + 256)
