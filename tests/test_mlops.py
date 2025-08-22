import numpy as np
import pytest

from tscv_vision import mlops


def test_validate_features_raises_on_nan() -> None:
    arr = np.array([1.0, np.nan])
    with pytest.raises(ValueError):
        mlops.validate_features(arr)


def test_drift_detector_detects_change() -> None:
    baseline = np.zeros(100)
    current = np.ones(100)
    det = mlops.DriftDetector(threshold=0.01)
    assert det.has_drift(baseline, current)


def test_assign_variant_is_deterministic() -> None:
    assert mlops.assign_variant("user") == mlops.assign_variant("user")


def test_resource_scaler() -> None:
    scaler = mlops.ResourceScaler(max_replicas=10)
    assert scaler.required_replicas(throughput=5.0, target=12.0) == 3


def test_feature_service_endpoint() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = mlops.create_feature_service()
    client = TestClient(app)
    resp = client.post("/extract", json={"data": [0.0, 1.0, 2.0, 3.0]})
    assert resp.status_code == 200
    features = np.array(resp.json()["features"])  # type: ignore[index]
    assert features.ndim == 1 and features.size > 0
