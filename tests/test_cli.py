from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from pytest import CaptureFixture, MonkeyPatch

import tscv_vision
from tscv_vision import cli, features


def test_cli_single(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    x = np.sin(np.linspace(0.0, 2 * np.pi, 64))
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    args = [
        "tscv-features",
        "--encoders",
        "gaf",
        "--input",
        str(in_path),
        "--output",
        str(out_path),
        "--bins",
        "8",
        "--features",
        "intensity",
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    out = capsys.readouterr().out
    assert "Saved features" in out
    data = np.load(out_path)
    assert data["features"].shape[0] == 6
    assert "image" in data.files
    assert "metadata" in data.files
    meta = json.loads(str(data["metadata"]))
    assert meta["encoders"] == ["gaf"]
    assert meta["tscv_vision_version"] == tscv_vision.__version__
    assert meta["feature_layout"] == features.feature_layout(bins=8, selected=["intensity"])
    assert "pywavelets" in meta["optional_backends"]


def test_cli_gdf(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    x = np.sin(np.linspace(0.0, 2 * np.pi, 64))
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    args = [
        "tscv-features",
        "--encoders",
        "gdf",
        "--input",
        str(in_path),
        "--output",
        str(out_path),
        "--features",
        "intensity",
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    out = capsys.readouterr().out
    assert "Saved features" in out
    data = np.load(out_path)
    assert data["features"].shape[0] == 6


def test_cli_sliding_flags(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    x = np.sin(np.linspace(0.0, 4 * np.pi, 200))
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    args = [
        "tscv-features",
        "--encoders",
        "rp",
        "--sliding",
        "--win-len",
        "50",
        "--hop",
        "25",
        "--input",
        str(in_path),
        "--output",
        str(out_path),
        "--bins",
        "8",
        "--features",
        "intensity,hist",
        "--no-save-images",
        "--no-save-meta",
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    out = capsys.readouterr().out
    assert "Saved features matrix" in out
    data = np.load(out_path)
    assert "features" in data.files
    assert data["features"].shape[1] == 6 + 8  # intensity+hist
    assert "images" not in data.files
    assert "window_starts" not in data.files
    assert "metadata" in data.files
    meta = json.loads(str(data["metadata"]))
    assert meta["sliding"] is True


def test_cli_multichannel(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    t = np.linspace(0.0, 2 * np.pi, 128)
    x = np.column_stack([np.sin(t), np.cos(t)])
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    args = [
        "tscv-features",
        "--encoders",
        "gaf",
        "--input",
        str(in_path),
        "--output",
        str(out_path),
        "--channel-fusion",
        "mean",
        "--features",
        "intensity",
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    out = capsys.readouterr().out
    assert "Saved features" in out
    data = np.load(out_path)
    assert data["features"].shape[0] == 6
    assert "metadata" in data.files


def test_cli_encoder_fusion_and_aggregate(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    x = np.sin(np.linspace(0.0, 4 * np.pi, 200))
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    args = [
        "tscv-features",
        "--encoders",
        "gaf,gadf",
        "--fusion",
        "mean",
        "--sliding",
        "--win-len",
        "50",
        "--hop",
        "25",
        "--aggregate",
        "mean",
        "--input",
        str(in_path),
        "--output",
        str(out_path),
        "--no-save-images",
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    out = capsys.readouterr().out
    assert "Saved features" in out
    data = np.load(out_path)
    assert data["features"].ndim == 1


def test_cli_parallel(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    x = np.sin(np.linspace(0.0, 4 * np.pi, 200))
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    args = [
        "tscv-features",
        "--encoders",
        "gaf",
        "--sliding",
        "--win-len",
        "40",
        "--hop",
        "20",
        "--parallel",
        "2",
        "--input",
        str(in_path),
        "--output",
        str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    out = capsys.readouterr().out
    assert "Saved features matrix" in out
    data = np.load(out_path)
    assert data["features"].shape[0] > 0


def test_cli_dry_run(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    x = np.sin(np.linspace(0.0, 2 * np.pi, 64))
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    args = [
        "tscv-features",
        "--encoders",
        "gaf",
        "--input",
        str(in_path),
        "--output",
        str(out_path),
        "--dry-run",
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    captured = capsys.readouterr()
    assert "[DRY RUN]" in captured.out
    assert not out_path.exists()


def test_cli_batch_processing(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    x1 = np.sin(np.linspace(0.0, 2 * np.pi, 64))
    x2 = np.cos(np.linspace(0.0, 2 * np.pi, 64))
    in1 = tmp_path / "a.npy"
    in2 = tmp_path / "b.npy"
    np.save(in1, x1)
    np.save(in2, x2)
    out_dir = tmp_path / "out"
    args = [
        "tscv-features",
        "--encoders",
        "gaf",
        "--input",
        str(in1),
        str(in2),
        "--output",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    captured = capsys.readouterr()
    assert "Saved features" in captured.out
    assert (out_dir / "a.npz").exists()
    assert (out_dir / "b.npz").exists()


def test_cli_config_json(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    x = np.sin(np.linspace(0.0, 2 * np.pi, 64))
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    cfg = {"encoders": "gaf", "bins": 8, "features": "intensity"}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    args = [
        "tscv-features",
        "--config",
        str(cfg_path),
        "--input",
        str(in_path),
        "--output",
        str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    captured = capsys.readouterr()
    assert "Saved features" in captured.out
    data = np.load(out_path)
    assert data["features"].shape[0] == 6


def test_cli_invalid_bins(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    x = np.sin(np.linspace(0.0, 2 * np.pi, 64))
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    args = [
        "tscv-features",
        "--encoders",
        "gaf",
        "--input",
        str(in_path),
        "--output",
        str(out_path),
        "--bins",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):
        cli.main()


def test_cli_config_yaml(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    x = np.sin(np.linspace(0.0, 2 * np.pi, 64))
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("encoders: gaf\nbins: 8\nfeatures: intensity\n")
    args = [
        "tscv-features",
        "--config",
        str(cfg_path),
        "--input",
        str(in_path),
        "--output",
        str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    captured = capsys.readouterr()
    assert "Saved features" in captured.out
    data = np.load(out_path)
    assert data["features"].shape[0] == 6


def test_cli_weighted_fusion(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    x = np.sin(np.linspace(0.0, 4 * np.pi, 128))
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    args = [
        "tscv-features",
        "--encoders",
        "gaf,gadf",
        "--fusion",
        "weighted",
        "--fusion-weights",
        "0.7,0.3",
        "--input",
        str(in_path),
        "--output",
        str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    captured = capsys.readouterr()
    assert "Saved features" in captured.out
    data = np.load(out_path)
    assert data["features"].shape[0] > 0


def test_cli_rp_eps(tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    x = np.sin(np.linspace(0.0, 2 * np.pi, 64))
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    args = [
        "tscv-features",
        "--encoders",
        "rp",
        "--rp-eps",
        "0.2",
        "--input",
        str(in_path),
        "--output",
        str(out_path),
        "--features",
        "intensity",
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    captured = capsys.readouterr()
    assert "Saved features" in captured.out
    data = np.load(out_path)
    img = data["image"]
    assert set(np.unique(img)) <= {0.0, 1.0}


def test_cli_spec_params(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    x = np.sin(np.linspace(0.0, 4 * np.pi, 128))
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    args = [
        "tscv-features",
        "--encoders",
        "spec",
        "--spec-win",
        "16",
        "--spec-hop",
        "8",
        "--spec-window",
        "rect",
        "--input",
        str(in_path),
        "--output",
        str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    captured = capsys.readouterr()
    assert "Saved features" in captured.out
    data = np.load(out_path)
    assert data["image"].ndim == 2


def test_cli_log_level(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    x = np.sin(np.linspace(0.0, 2 * np.pi, 64))
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    args = [
        "tscv-features",
        "--encoders",
        "gaf",
        "--input",
        str(in_path),
        "--output",
        str(out_path),
        "--log-level",
        "ERROR",
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    captured = capsys.readouterr()
    assert "Processing" not in captured.err


def test_cli_progress_output(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    x = np.sin(np.linspace(0.0, 4 * np.pi, 200))
    in_path = tmp_path / "x.npy"
    np.save(in_path, x)
    out_path = tmp_path / "out.npz"
    args = [
        "tscv-features",
        "--encoders",
        "gaf",
        "--sliding",
        "--win-len",
        "50",
        "--hop",
        "25",
        "--input",
        str(in_path),
        "--output",
        str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    captured = capsys.readouterr()
    assert "Saved features matrix" in captured.out
    assert "Processing" in captured.err
