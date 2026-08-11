import numpy as np
import pytest

pytest.importorskip("sklearn")

from tscv_vision.domains import (  # noqa: E402
    DomainAdapter,
    PrototypicalClassifier,
    classification_metrics,
    finance,
    healthcare,
    uncertainty_sampling,
)

pytestmark = pytest.mark.optional


def test_domain_adapter_and_metrics() -> None:
    series0 = finance.generate_price_series(40, drift=-0.01)
    series1 = finance.generate_price_series(40, drift=0.01)
    adapter = DomainAdapter(finance.microstructure_features)
    adapter.fit([series0, series1], np.array([0, 1]))
    preds = adapter.predict([series0, series1])
    metrics = classification_metrics(np.array([0, 1]), preds)
    assert metrics["accuracy"] == 1.0


def test_prototypical_classifier_basic() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])
    clf = PrototypicalClassifier().fit(X, y)
    pred = clf.predict(np.array([[0.1, 0.2], [0.9, 0.8]]))
    assert pred.tolist() == [0, 1]


def test_uncertainty_sampling() -> None:
    s0 = finance.generate_price_series(40, drift=-0.01)
    s1 = finance.generate_price_series(40, drift=0.01)
    adapter = DomainAdapter(finance.microstructure_features)
    adapter.fit([s0, s1], np.array([0, 1]))
    pool = [finance.generate_price_series(40) for _ in range(5)]
    idx = uncertainty_sampling(adapter, pool, 2)
    assert idx.shape == (2,)
    assert np.all((0 <= idx) & (idx < len(pool)))


def test_domain_augmentations() -> None:
    prices = finance.generate_price_series(20)
    aug_prices = finance.augment_regime_switch(prices, switch_point=10)
    assert aug_prices.shape == prices.shape
    assert not np.allclose(prices, aug_prices)

    ecg = healthcare.generate_ecg(100)
    aug_ecg = healthcare.augment_noise(ecg, scale=0.1)
    assert aug_ecg.shape == ecg.shape
    assert not np.allclose(ecg, aug_ecg)
