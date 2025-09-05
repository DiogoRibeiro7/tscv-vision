from __future__ import annotations

import numpy as np
import pytest

from tscv_vision.sliding import encode_sliding, features_for_sliding, sliding_windows


def test_sliding_windows_view() -> None:
    x = np.arange(10, dtype=float)
    view = sliding_windows(x, size=4, hop=2)
    assert view.shape == (4, 4)
    assert not view.flags.writeable
    assert np.allclose(view[0], x[:4])


def test_encode_sliding_rp() -> None:
    x = np.sin(np.linspace(0.0, 4 * np.pi, 128))
    imgs, starts = encode_sliding(x, encoder="rp", size=32, hop=16, eps=0.3)
    assert imgs.shape == (1 + (128 - 32) // 16, 32, 32)
    assert starts[0] == 0


def test_features_for_sliding() -> None:
    x = np.sin(np.linspace(0.0, 6 * np.pi, 180))
    sel = ["intensity", "hist", "gradient", "lbp"]
    feats, starts = features_for_sliding(
        x, encoder="gaf", size=60, hop=30, bins=8, feature_names=sel
    )
    assert feats.shape[0] == starts.shape[0]
    assert feats.shape[1] == 6 + 8 + 16 + 256


def test_sliding_windows_validations() -> None:
    x = np.arange(8.0)
    with pytest.raises(ValueError):
        sliding_windows(np.zeros((2, 2, 2)), size=4)
    with pytest.raises(ValueError):
        sliding_windows(x, size=1)
    with pytest.raises(ValueError):
        sliding_windows(x, size=16)
    view = sliding_windows(x, size=4)
    assert view.shape[0] == 3
    multi = sliding_windows(np.stack([x, x], axis=1), size=4)
    assert multi.shape == (3, 4, 2)


def test_encode_sliding_invalid_and_spec() -> None:
    x = np.linspace(0.0, 2 * np.pi, 64)
    with pytest.raises(ValueError):
        encode_sliding(x[:10], size=20)
    imgs, starts = encode_sliding(x, encoder="spec", size=32, hop=16)
    assert imgs.ndim == 3
    assert starts.shape[0] == imgs.shape[0]


def test_encode_sliding_lazy() -> None:
    x = np.sin(np.linspace(0.0, 4 * np.pi, 128))
    eager, starts = encode_sliding(x, size=32, hop=16)
    gen = encode_sliding(x, size=32, hop=16, lazy=True)
    assert not isinstance(gen, np.ndarray)
    collected = list(gen)
    imgs_lazy, starts_lazy = zip(*collected)
    np.testing.assert_allclose(np.stack(imgs_lazy), eager)
    np.testing.assert_array_equal(np.array(starts_lazy), starts)


def test_encode_sliding_multichannel() -> None:
    t = np.linspace(0.0, 2 * np.pi, 128)
    x = np.column_stack([np.sin(t), np.cos(t)])
    imgs, _ = encode_sliding(x, size=32, hop=32, channel_fusion="stack")
    assert imgs.shape[-1] == 2
    imgs_mean, _ = encode_sliding(x, encoder="rp", size=32, channel_fusion="mean")
    assert imgs_mean.shape[1:] == (32, 32)
    imgs_concat, _ = encode_sliding(
        x,
        encoder="spec",
        size=64,
        hop=64,
        channel_fusion="concat",
        spec_win=32,
        spec_hop=16,
    )
    assert imgs_concat.shape[2] == 6


def test_features_for_sliding_lazy() -> None:
    t = np.linspace(0.0, 2 * np.pi, 128)
    x = np.sin(t)
    eager_feats, starts = features_for_sliding(x, size=32, hop=32)
    gen = features_for_sliding(x, size=32, hop=32, lazy=True)
    assert not isinstance(gen, np.ndarray)
    collected = list(gen)
    feats_lazy, starts_lazy = zip(*collected)
    np.testing.assert_allclose(np.vstack(feats_lazy), eager_feats)
    np.testing.assert_array_equal(np.array(starts_lazy), starts)

