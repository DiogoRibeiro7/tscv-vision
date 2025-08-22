import numpy as np

from tscv_vision import irregular


def test_resample_irregular():
    t = np.array([0.0, 0.5, 1.7, 3.0])
    x = np.array([0.0, 1.0, 0.0, 1.0])
    t_new, x_new = irregular.resample_irregular(t, x, step=1.0)
    assert np.allclose(t_new, np.array([0.0, 1.0, 2.0, 3.0]))
    assert x_new.shape == t_new.shape

