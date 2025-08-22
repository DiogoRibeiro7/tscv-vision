import numpy as np

from tscv_vision.domains import (
    astronomy,
    audio,
    climate,
    finance,
    healthcare,
    iot,
    manufacturing,
)


def test_finance_microstructure_and_regime() -> None:
    prices = np.linspace(100, 110, 50)
    feats = finance.microstructure_features(prices)
    assert feats.shape == (3,)
    regime = finance.detect_regime(prices)
    assert regime in (0, 1)


def test_healthcare_ecg_features() -> None:
    t = np.linspace(0, 1, 250)
    ecg = np.sin(2 * np.pi * 5 * t)
    hr, sdnn = healthcare.ecg_features(ecg)
    assert hr >= 0 and sdnn >= 0


def test_iot_fusion_and_anomaly() -> None:
    sensors = np.vstack([np.ones(10), np.zeros(10)])
    fused = iot.fuse_sensors(sensors)
    assert fused.shape == (10,)
    score = iot.anomaly_score(fused)
    assert score >= 0


def test_audio_mfcc_and_rhythm() -> None:
    signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
    coeffs = audio.mfcc_features(signal)
    assert coeffs.shape[0] == 13
    assert isinstance(audio.rhythm_score(signal), float)


def test_astronomy_periodicity() -> None:
    t = np.linspace(0, 1, 128)
    sig = np.sin(2 * np.pi * 3 * t)
    freq, amp = astronomy.periodicity_features(sig, fs=128)
    assert np.isclose(freq, 3, atol=1)
    assert amp > 0


def test_climate_seasonal_features() -> None:
    data = np.sin(np.linspace(0, 2 * np.pi, 24))
    feats = climate.seasonal_trend_features(data, period=12)
    assert feats.shape == (3,)


def test_manufacturing_vibration_features() -> None:
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(128)
    feats = manufacturing.vibration_features(sig)
    assert feats.shape == (3,)
    score = manufacturing.quality_score(sig)
    assert isinstance(score, float)

