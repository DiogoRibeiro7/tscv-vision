import numpy as np

from tscv_vision import encoders
from tscv_vision.pipeline import AdaptivePipeline, FeatureEnsemble, select_features


def _sum_encoder(x: np.ndarray) -> np.ndarray:
    s = float(np.sum(x))
    return np.full((2, 2), s)


def _zero_encoder(x: np.ndarray) -> np.ndarray:
    return np.zeros((2, 2))


def setup_module(module):
    encoders.register_encoder("sum", _sum_encoder)
    encoders.register_encoder("zero", _zero_encoder)


def test_select_features_methods() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = (X[:, 0] > 0).astype(int)
    for method in ["mutual_info", "correlation", "stability"]:
        idx = select_features(X, y, method=method, k=2, random_state=0)
        assert idx.shape == (2,)


def test_adaptive_pipeline_selection(tmp_path) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 8))
    y = (X.sum(axis=1) > 0).astype(int)
    pipe = AdaptivePipeline(encoders=["sum", "zero"], random_state=0, k=2)
    feats = pipe.fit_transform(X, y)
    assert pipe.selected_encoder_ == "sum"
    assert feats.shape == (30, 2)
    p = tmp_path / "pipe.pkl"
    pipe.save(p)
    loaded = AdaptivePipeline.load(p)
    np.testing.assert_allclose(feats, loaded.transform(X))


def test_feature_ensemble() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8))
    y = (X.sum(axis=1) > 0).astype(int)
    ens = FeatureEnsemble(["sum", "zero"], random_state=0)
    feats = ens.fit_transform(X, y)
    assert feats.shape[0] == 20
    fdim = ens._extract_all(X, "sum").shape[1]
    assert feats.shape[1] == 2 * fdim


def test_adaptive_pipeline_optimize() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 8))
    y = (X.sum(axis=1) > 0).astype(int)
    pipe = AdaptivePipeline(encoders=["sum", "zero"], random_state=0)
    best_enc, best_k, score = pipe.optimize(X, y, n_iter=4)
    assert best_enc == "sum"
    assert best_k > 0
    assert 0 <= score <= 1
