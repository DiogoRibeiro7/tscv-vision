import numpy as np

from tscv_vision import dataset


def test_stream_directory(tmp_path):
    a = tmp_path / "a.npy"
    b = tmp_path / "b.npy"
    np.save(a, np.array([1, 2, 3], dtype=float))
    np.save(b, np.array([4, 5, 6], dtype=float))
    streams = list(dataset.stream_directory(str(tmp_path)))
    assert len(streams) == 2
    assert np.allclose(streams[0], [1, 2, 3])
    assert np.allclose(streams[1], [4, 5, 6])

