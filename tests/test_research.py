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


def test_bias_report_and_dp() -> None:
    feats = np.array([1.0, 2.0, 3.0])
    groups = np.array([0, 0, 1])
    report = research.bias_report(feats, groups)
    assert "0" in report and "1" in report
    rng = np.random.default_rng(0)
    noisy = research.add_dp_noise(feats, 1.0, rng=rng)
    assert noisy.shape == feats.shape


def test_plugin_and_report(tmp_path: Path) -> None:
    def _plugin(x: float) -> float:
        return x * 2

    research.register_plugin("double", _plugin)
    assert research.PLUGIN_REGISTRY["double"](3) == 6
    paper = research.generate_paper({"a": 1}, tmp_path / "r.md")
    assert paper.exists()
