import numpy as np

from tscv_vision import automl


def test_suggest_encoders() -> None:
    series = np.sin(np.linspace(0, 2 * np.pi, 256))
    recs = automl.suggest_encoders(series, max_encoders=3)
    assert 0 < len(recs) <= 3
    assert all(isinstance(r, str) for r in recs)


def test_rank_feature_importance_corr() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X[:, 0] * 0.5 + rng.normal(scale=0.1, size=100)
    order = automl.rank_feature_importance(X, y)
    assert order[0] == 0


def test_meta_learner() -> None:
    ml = automl.MetaLearner()
    ml.update(100, {"enc": "gaf"})
    ml.update(200, {"enc": "spec"})
    assert ml.suggest(150) == {"enc": "gaf"}


def test_active_window_selection() -> None:
    series = np.linspace(0, 1, 50)
    win, hop = automl.active_window_selection(series, [10, 20])
    assert win in {10, 20}
    assert 1 <= hop <= win


def test_evolve_hyperparams() -> None:
    def objective(params: dict[str, int]) -> float:
        return -(params["a"] - 2) ** 2

    search = {"a": [0, 1, 2, 3]}
    best = automl.evolve_hyperparams(objective, search, generations=3, population=4)
    assert best["a"] == 2


def test_select_feature_subset() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 5))
    y = X[:, 2] + rng.normal(scale=0.01, size=50)

    def obj(x: np.ndarray, target: np.ndarray) -> float:
        preds = x.mean(axis=1)
        return -float(np.mean((preds - target) ** 2))

    idx = automl.select_feature_subset(X, y, max_features=2, objective=obj)
    assert 2 in idx
    assert idx.size <= 2
