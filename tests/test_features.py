from __future__ import annotations

import numpy as np
import pytest

from tscv_vision import features


def test_histogram_constant() -> None:
    img = np.zeros((4, 4))
    h = features.histogram(img, bins=4)
    assert h.shape == (4,)
    assert h[0] == 1.0


def test_gradient_histogram_constant() -> None:
    img = np.ones((4, 4))
    g = features.gradient_histogram(img, bins=8)
    assert g.shape == (8,)
    assert g[0] == 1.0


def test_lbp_constant() -> None:
    img = np.full((4, 4), 5.0)
    h = features.lbp(img)
    assert h.shape == (256,)
    assert np.isclose(h[-1], 1.0)


def test_extract_feature_vector() -> None:
    img = np.arange(16, dtype=float).reshape(4, 4)
    vec = features.extract_feature_vector(img, bins=8)
    expected = 0
    for func in features.FEATURES_REGISTRY.values():
        try:
            expected += func(img, bins=8).size
        except ImportError:
            continue
    assert vec.shape == (expected,)


def test_extract_feature_vector_unknown_selected() -> None:
    import pytest

    img = np.zeros((4, 4))
    with pytest.raises(KeyError):
        features.extract_feature_vector(img, selected=["does_not_exist"])


def test_extract_feature_vector_empty_parts_raises(monkeypatch) -> None:
    import pytest

    def _always_missing(img: np.ndarray, bins: int) -> np.ndarray:
        raise ImportError("simulated missing dependency")

    registry = {name: _always_missing for name in features.FEATURES_REGISTRY}
    monkeypatch.setattr(features, "FEATURES_REGISTRY", registry)
    img = np.zeros((4, 4))
    with pytest.raises(RuntimeError, match="No feature extractors"):
        features.extract_feature_vector(img)


def test_extract_feature_vector_multichannel() -> None:
    base = np.arange(16, dtype=float).reshape(4, 4)
    img = np.stack([base, base * 0], axis=-1)
    vec = features.extract_feature_vector(img, bins=8)
    per_ch = 0
    for func in features.FEATURES_REGISTRY.values():
        try:
            per_ch += func(base, bins=8).size
        except ImportError:
            continue
    assert vec.shape[0] == 2 * per_ch


def test_glcm_features_shape() -> None:
    img = np.array([[0, 1], [1, 0]], dtype=float)
    f = features.glcm_features(img)
    assert f.shape == (12,)


def test_lbp_variants() -> None:
    img = np.full((4, 4), 5.0)
    ri = features.lbp_ri(img)
    uni = features.lbp_uniform(img)
    assert ri.shape == (256,)
    assert uni.shape[0] == features._LBP_UNI_BINS  # type: ignore[attr-defined]


def test_extract_feature_vector_reuses_lbp_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    img = np.arange(64, dtype=float).reshape(8, 8)
    selected = ["lbp", "lbp_ri", "lbp_uniform"]
    expected = np.concatenate([features.lbp(img), features.lbp_ri(img), features.lbp_uniform(img)])
    original = features._lbp_codes  # type: ignore[attr-defined]
    calls = 0

    def counted_lbp_codes(
        image: np.ndarray, radius: int, points: int = features._LBP_POINTS  # type: ignore[attr-defined]
    ) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(image, radius, points)

    monkeypatch.setattr(features, "_lbp_codes", counted_lbp_codes)
    got = features.extract_feature_vector(img, selected=selected)
    np.testing.assert_allclose(got, expected)
    assert calls == 1


def test_extract_feature_vector_reuses_gradient_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = np.arange(64, dtype=float).reshape(8, 8)
    selected = ["gradient", "edge_density", "orientation", "contour"]
    expected = np.concatenate(
        [
            features.gradient_histogram(img, bins=16),
            features.edge_density(img),
            features.orientation_histogram(img, bins=32),
            features.contour_ratio(img),
        ]
    )
    original = features._conv2  # type: ignore[attr-defined]
    calls = 0

    def counted_conv2(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(image, kernel)

    monkeypatch.setattr(features, "_conv2", counted_conv2)
    got = features.extract_feature_vector(img, selected=selected)
    np.testing.assert_allclose(got, expected)
    assert calls == 2


def test_gabor_and_fft_features() -> None:
    img = np.zeros((8, 8))
    gabor = features.gabor_features(img)
    fft = features.fft_features(img)
    assert gabor.shape[0] == 16
    assert fft.shape == (4,)


def test_rank_and_select() -> None:
    X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 0.0], [1.0, 2.0, 1.0]])
    idx = features.rank_features(X)
    assert idx[0] == 2
    top = features.select_top_k(X, 2)
    assert top.shape == (3, 2)


def test_fuse_features() -> None:
    a = np.ones((2, 3))
    b = np.zeros((2, 3))
    fused = features.fuse_features([a, b], mode="mean")
    np.testing.assert_allclose(fused, 0.5)

