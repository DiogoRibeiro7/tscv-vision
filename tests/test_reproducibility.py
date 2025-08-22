from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tscv_vision import cli
from tscv_vision.sliding import encode_sliding, features_for_sliding


def test_reproducible_encode_sliding() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=128)
    imgs1, starts1 = encode_sliding(x, encoder="gaf", size=32, hop=16)
    imgs2, starts2 = encode_sliding(x, encoder="gaf", size=32, hop=16)
    np.testing.assert_allclose(imgs1, imgs2)
    np.testing.assert_array_equal(starts1, starts2)


def test_reproducible_pipeline_features() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=128)
    f1, s1 = features_for_sliding(x, encoder="gaf", size=32, hop=16, bins=8)
    f2, s2 = features_for_sliding(x, encoder="gaf", size=32, hop=16, bins=8)
    np.testing.assert_allclose(f1, f2)
    np.testing.assert_array_equal(s1, s2)


def test_cli_reproducibility(tmp_path: Path, monkeypatch) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=64)
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out1 = tmp_path / "a.npz"
    args = [
        "tscv-features",
        "--encoders",
        "gaf",
        "--input",
        str(in_path),
        "--output",
        str(out1),
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    f1 = np.load(out1)["features"]
    out2 = tmp_path / "b.npz"
    args[-1] = str(out2)
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    f2 = np.load(out2)["features"]
    np.testing.assert_allclose(f1, f2)
