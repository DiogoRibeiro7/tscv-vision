from __future__ import annotations

from tscv_vision.domains import iot


def test_iot_features_shape() -> None:
    sig = iot.generate_sensor_series(16, seed=0)
    feats = iot.iot_features(sig)
    assert feats.shape == (4,)
