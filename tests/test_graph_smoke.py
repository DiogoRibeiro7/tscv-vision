import numpy as np


def test_small_graph_generation() -> None:
    rng = np.random.default_rng(0)
    n_nodes = 100
    edges = rng.integers(0, n_nodes, size=(100, 2), endpoint=False)
    degree = np.bincount(edges[:, 0], minlength=n_nodes)
    assert degree.sum() == 100
