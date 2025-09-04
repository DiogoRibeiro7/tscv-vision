import numpy as np
import pytest

pytestmark = pytest.mark.skip(reason="stress test; run locally")


def test_large_sparse_graph() -> None:
    """Generate a large sparse graph to stress memory usage.

    The test constructs a graph with ~1e6 edges and computes simple
    statistics to ensure algorithms handle the size without excessive
    allocations.
    """

    rng = np.random.default_rng(0)
    n_nodes = 100_000
    edges = rng.integers(0, n_nodes, size=(1_000_000, 2), endpoint=False)
    degree = np.bincount(edges[:, 0], minlength=n_nodes)
    assert degree.shape == (n_nodes,)
