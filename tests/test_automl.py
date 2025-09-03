import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1] / "src" / "tscv_vision"
package = types.ModuleType("tscv_vision")
package.__path__ = [str(BASE)]  # type: ignore[attr-defined]
sys.modules["tscv_vision"] = package

for name in ["encoders", "features"]:
    spec = importlib.util.spec_from_file_location(f"tscv_vision.{name}", BASE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"tscv_vision.{name}"] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]

spec = importlib.util.spec_from_file_location("tscv_vision.automl", BASE / "automl.py")
assert spec and spec.loader
automl = importlib.util.module_from_spec(spec)
sys.modules["tscv_vision.automl"] = automl
spec.loader.exec_module(automl)  # type: ignore[assignment]
AutoTSCV = automl.AutoTSCV


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


def test_auto_tscv_basic() -> None:
    t = np.linspace(0, 2 * np.pi, 64)
    X = [np.sin(t), np.sin(t + np.pi / 2), np.linspace(0, 1, 64), np.linspace(0, 1, 64) + 0.1]
    y = np.array([0, 0, 1, 1])
    model = AutoTSCV()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.mean(preds == y) >= 0.5
    assert model.profile_ is not None


def test_auto_tscv_drift_detection() -> None:
    t = np.linspace(0, 1, 32)
    base = [np.sin(2 * np.pi * t)]
    auto = AutoTSCV()
    auto.profile(base)
    shifted = [np.sin(2 * np.pi * t) + 5]
    assert auto.detect_drift(shifted, threshold=0.1)
