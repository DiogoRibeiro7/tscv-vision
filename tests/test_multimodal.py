import numpy as np

from tscv_vision.multimodal import (
    coral_align,
    cross_modal_concat,
    federated_average,
    fuse_series,
    granger_causality,
    temporal_graph_propagate,
)


def test_fuse_series_weighted():
    data = np.vstack([np.arange(4), np.arange(4) * 2])
    fused = fuse_series(data, weights=np.array([0.25, 0.75]))
    assert fused.shape == (1, 4)
    expected = np.array([0.0, 0.25 + 1.5, 0.5 + 3.0, 0.75 + 4.5])
    assert np.allclose(fused[0], expected)


def test_cross_modal_concat():
    s = np.arange(3)
    meta = np.array([1.0, 2.0])
    out = cross_modal_concat(s, meta)
    assert out.shape == (5,)


def test_coral_align_shapes():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(10, 3))
    tgt = rng.normal(size=(8, 3))
    aligned = coral_align(src, tgt)
    assert aligned.shape == (10, 3)


def test_temporal_graph_propagate():
    series = np.array([[1.0, 2.0], [0.0, 1.0]])
    adj = np.array([[1.0, 1.0], [1.0, 1.0]])
    out = temporal_graph_propagate(series, adj, steps=1)
    assert out.shape == series.shape
    # with uniform adjacency, nodes average
    assert np.allclose(out, np.mean(series, axis=0, keepdims=True).repeat(2, axis=0))


def test_granger_causality_detects():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = np.roll(x, 1) + rng.normal(scale=0.1, size=100)
    score = granger_causality(x, y, maxlag=1)
    assert score > 0


def test_federated_average():
    params = [np.ones((2, 2)), np.zeros((2, 2))]
    avg = federated_average(params, weights=[0.25, 0.75])
    assert np.allclose(avg, np.full((2, 2), 0.25))
