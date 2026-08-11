import json
from pathlib import Path

import numpy as np

from tscv_vision import research


def test_track_experiment(tmp_path: Path) -> None:
    data = np.arange(5.0)
    data_path = tmp_path / "d.npy"
    np.save(data_path, data)
    log = research.track_experiment({"param": 1}, data_path, tmp_path)
    meta = json.loads(log.read_text())
    assert meta["hash"]
    assert meta["config"]["param"] == 1


def test_group_mean_disparity() -> None:
    feats = np.array([1.0, 2.0, 3.0])
    groups = np.array([0, 0, 1])
    report = research.group_mean_disparity(feats, groups)
    assert report["0"] == 1.5
    assert report["1"] == 3.0
    assert report["max_diff"] == 1.5
    assert research.group_mean_disparity(feats, np.zeros(3))["max_diff"] == 0.0


def test_group_mean_disparity_shape_mismatch() -> None:
    import pytest

    with pytest.raises(ValueError):
        research.group_mean_disparity(np.zeros(3), np.zeros(2))


def test_bias_report_alias_warns() -> None:
    import pytest

    feats = np.array([1.0, 2.0, 3.0])
    groups = np.array([0, 0, 1])
    with pytest.warns(DeprecationWarning, match="group_mean_disparity"):
        report = research.bias_report(feats, groups)
    assert report == research.group_mean_disparity(feats, groups)


def test_add_dp_noise_requires_sensitivity() -> None:
    import pytest

    feats = np.array([1.0, 2.0, 3.0])
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError):
        research.add_dp_noise(feats, 1.0, rng=rng)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="sensitivity"):
        research.add_dp_noise(feats, 1.0, sensitivity=0.0, rng=rng)
    with pytest.raises(ValueError, match="epsilon"):
        research.add_dp_noise(feats, 0.0, sensitivity=1.0, rng=rng)


def test_add_dp_noise_scales_with_sensitivity() -> None:
    feats = np.zeros(20000)
    small = research.add_dp_noise(
        feats, epsilon=1.0, sensitivity=1.0, rng=np.random.default_rng(0)
    )
    large = research.add_dp_noise(
        feats, epsilon=1.0, sensitivity=10.0, rng=np.random.default_rng(0)
    )
    assert small.shape == feats.shape
    # Laplace(b) has std sqrt(2) * b, so a 10x sensitivity means 10x the spread.
    assert 8.0 < large.std() / small.std() < 12.0


def test_add_laplace_noise_is_reproducible() -> None:
    import pytest

    feats = np.array([1.0, 2.0, 3.0])
    a = research.add_laplace_noise(feats, 0.5, rng=np.random.default_rng(7))
    b = research.add_laplace_noise(feats, 0.5, rng=np.random.default_rng(7))
    np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError, match="scale"):
        research.add_laplace_noise(feats, 0.0)


def test_plugin_and_report(tmp_path: Path) -> None:
    def _plugin(x: float) -> float:
        return x * 2

    research.register_plugin("double", _plugin)
    assert research.PLUGIN_REGISTRY["double"](3) == 6
    paper = research.generate_paper({"a": 1}, tmp_path / "r.md")
    assert paper.exists()
