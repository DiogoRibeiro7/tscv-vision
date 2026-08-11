import numpy as np

from tscv_vision.streaming import OnlineLearner


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
