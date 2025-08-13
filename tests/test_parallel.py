import numpy as np

from tscv_vision.parallel import map_parallel
from tscv_vision.sliding import encode_sliding


def square(x: int) -> int:
    return x * x


def test_map_parallel() -> None:
    data = [0, 1, 2, 3]
    out = map_parallel(square, data, workers=2)
    assert out == [0, 1, 4, 9]


def test_encode_sliding_parallel() -> None:
    x = np.sin(np.linspace(0.0, 2 * np.pi, 128))
    imgs1, starts1 = encode_sliding(x, size=32, hop=16)
    imgs2, starts2 = encode_sliding(x, size=32, hop=16, workers=2)
    assert np.allclose(imgs1, imgs2)
    assert np.array_equal(starts1, starts2)
