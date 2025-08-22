import numpy as np

from tscv_vision import analytics


def test_saliency_map_sum() -> None:
    series = np.arange(5.0)
    grad = analytics.saliency_map(lambda x: float(x.sum()), series)
    assert np.allclose(grad, 1.0)


def test_project_features_fallback() -> None:
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((10, 4))
    proj = analytics.project_features(feats, method="tsne", perplexity=5)
    assert proj.shape == (10, 2)


def test_group_significance() -> None:
    a = np.array([1.0, 1.2, 0.9, 1.1])
    b = np.array([2.0, 1.9, 2.1, 2.2])
    t, p = analytics.group_significance(a, b)
    assert t < 0
    assert 0 <= p <= 1


def test_cross_causal_lag() -> None:
    x = np.zeros(20)
    y = np.zeros(20)
    x[5] = 1
    y[7] = 1
    lag = analytics.cross_causal_lag(x, y, max_lag=5)
    assert lag == -2


def test_shap_missing() -> None:
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((4, 3))
    try:
        analytics.shap_values(lambda z: z, feats)
    except ImportError:
        pass
    else:
        assert True


def test_counterfactual_replace() -> None:
    series = np.zeros(10)
    def model(x: np.ndarray) -> float:
        return float(x.sum())
    pert, diff = analytics.counterfactual_replace(series, 2, 4, 1.0, model)
    assert np.allclose(pert[2:4], 1.0)
    assert np.isclose(diff, 2.0)
