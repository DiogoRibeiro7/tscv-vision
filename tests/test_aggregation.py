import numpy as np

from tscv_vision.aggregation import aggregate


def test_multiple_aggregators():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = aggregate(x, ["mean", "max"])
    assert out.shape == (4,)
    assert np.allclose(out[:2], [2.0, 3.0])
    assert np.allclose(out[2:], [3.0, 4.0])
