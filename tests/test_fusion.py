import numpy as np

from tscv_vision.fusion import fuse


def test_concat_fusion_shapes():
    a = np.ones((2, 3))
    b = np.zeros((2, 3))
    out = fuse([a, b], mode="concat")
    assert out.shape == (2, 6)


def test_mean_and_weighted_fusion():
    a = np.array([[1.0, 2.0]])
    b = np.array([[3.0, 4.0]])
    mean = fuse([a, b], mode="mean")
    assert np.allclose(mean, [[2.0, 3.0]])
    weighted = fuse([a, b], mode="weighted", weights=[0.25, 0.75])
    assert np.allclose(weighted, [[2.5, 3.5]])
