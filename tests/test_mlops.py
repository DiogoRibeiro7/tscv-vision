import importlib.util
import pathlib
import sys
import threading
import time
import types

import numpy as np
import pytest

pkg_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "tscv_vision"
pkg = types.ModuleType("tscv_vision")
pkg.__path__ = [str(pkg_path)]
sys.modules["tscv_vision"] = pkg

spec = importlib.util.spec_from_file_location("tscv_vision.mlops", pkg_path / "mlops.py")
assert spec and spec.loader
mlops = importlib.util.module_from_spec(spec)
sys.modules["tscv_vision.mlops"] = mlops
spec.loader.exec_module(mlops)


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


def test_model_registry_concurrent() -> None:
    reg = mlops.ModelRegistry()

    def worker(v: int) -> None:
        reg.register("enc", f"{v}", {"acc": float(v)})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert reg.latest("enc") == "2"


def test_abtester_detects_difference() -> None:
    tester = mlops.ABTester()
    for v in (0.1, 0.2, 0.3):
        tester.add("A", v)
    for v in (0.4, 0.5, 0.6):
        tester.add("B", v)
    res = tester.compare()
    assert res.lift > 0
    assert res.p_value < 0.1


def test_safe_encode_fallback_and_timeout() -> None:
    series = np.arange(10.0)

    def primary_fail(x: mlops.Array) -> mlops.Array:
        raise RuntimeError("boom")

    def primary_slow(x: mlops.Array) -> mlops.Array:
        time.sleep(0.05)
        return x

    def fallback(x: mlops.Array) -> mlops.Array:
        return np.zeros_like(x)

    out1 = mlops.safe_encode(series, primary_fail, fallback)
    assert np.all(out1 == 0)

    out2 = mlops.safe_encode(series, primary_slow, fallback, timeout=0.01)
    assert np.all(out2 == 0)


def test_batch_process_resume_and_progress() -> None:
    data = [np.arange(8.0)] * 200

    def func(batch: mlops.Array) -> list[mlops.Array]:
        return [row * 2 for row in batch]

    calls: list[int] = []

    def progress(n: int) -> None:
        calls.append(n)

    out = mlops.batch_process(data, func, batch_size=32, start=50, progress=progress)
    assert len(out) == len(data) - 50
    assert calls and calls[-1] == len(out)
    assert np.array_equal(out[0], np.arange(8.0) * 2)


def test_monitoring_app_endpoints() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    det = mlops.DriftDetector(threshold=0.0)
    app = mlops.create_monitoring_app(det)
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    drift = client.post(
        "/drift", json={"baseline": [0.0, 0.0], "current": [1.0, 1.0]}
    ).json()["drift"]
    assert drift
    assert client.post("/quality", json={"score": 0.5}).json()["score"] == 0.5


def test_feature_service_endpoint() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = mlops.create_feature_service()
    client = TestClient(app)
    resp = client.post("/extract", json={"data": [0.0, 1.0, 2.0, 3.0]})
    assert resp.status_code == 200
    features = np.array(resp.json()["features"])  # type: ignore[index]
    assert features.ndim == 1 and features.size > 0
