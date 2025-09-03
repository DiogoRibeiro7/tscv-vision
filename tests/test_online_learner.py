import importlib.util
import pathlib
import sys
import types

import numpy as np

PKG_PATH = pathlib.Path(__file__).resolve().parents[1] / "src" / "tscv_vision"
pkg = types.ModuleType("tscv_vision")
pkg.__path__ = [str(PKG_PATH)]
sys.modules["tscv_vision"] = pkg
spec = importlib.util.spec_from_file_location("tscv_vision.streaming", PKG_PATH / "streaming.py")
_mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["tscv_vision.streaming"] = _mod
spec.loader.exec_module(_mod)
OnlineLearner = _mod.OnlineLearner


def test_online_learner_learns_identity() -> None:
    rng = np.random.default_rng(0)
    learner = OnlineLearner()
    for x in rng.normal(size=200):
        learner.update(float(x), float(x))
    pred = learner.predict(0.5)
    assert pred is not None
    assert abs(pred - 0.5) < 0.2


def test_online_learner_drift_detection() -> None:
    events: list[float] = []
    learner = OnlineLearner(on_drift=lambda err: events.append(err), drift_threshold=1.5)
    rng = np.random.default_rng(0)
    for x in rng.normal(size=100):
        learner.update(float(x), float(x))
    for x in rng.normal(loc=5.0, size=200):
        learner.update(float(x), float(x))
    assert events


def _feat_fn(win: np.ndarray) -> np.ndarray:
    return np.array([win[-1], 1.0])


def test_dynamic_feature_selection() -> None:
    rng = np.random.default_rng(0)
    learner = OnlineLearner(feature_fn=_feat_fn)
    for x in rng.normal(size=50):
        learner.update(float(x), float(x))
    assert learner.selected_features == [0]
