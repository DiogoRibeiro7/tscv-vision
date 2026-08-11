"""Tests for the leakage-safe benchmark harness."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tscv_vision import evaluation

sklearn = pytest.importorskip("sklearn")


def _tiny_dataset(name: str = "Tiny", seed: int = 0) -> evaluation.TSDataset:
    return evaluation.make_synthetic_dataset(name, n_per_class=6, length=32, seed=seed)


def test_synthetic_dataset_split_is_disjoint_and_balanced() -> None:
    ds = _tiny_dataset()
    assert ds.X_train.shape[1] == ds.X_test.shape[1] == 32
    assert ds.X_train.shape[0] + ds.X_test.shape[0] == 24
    assert ds.n_classes == 2
    assert ds.summary()["length"] == 32
    # No series appears in both splits.
    train = {row.tobytes() for row in ds.X_train}
    assert not train & {row.tobytes() for row in ds.X_test}


def test_load_ucr_tsv_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "Demo"
    root.mkdir()
    rng = np.random.default_rng(0)
    for split, n in (("TRAIN", 5), ("TEST", 4)):
        labels = rng.integers(1, 3, size=n)
        series = rng.normal(size=(n, 7))
        block = np.column_stack([labels, series])
        np.savetxt(root / f"Demo_{split}.tsv", block, delimiter="\t")
    ds = evaluation.load_ucr_tsv(tmp_path, "Demo")
    assert ds.name == "Demo"
    assert ds.X_train.shape == (5, 7)
    assert ds.X_test.shape == (4, 7)
    assert evaluation.list_ucr_datasets(tmp_path) == ["Demo"]


def test_load_ucr_tsv_reports_missing_and_bad_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        evaluation.load_ucr_tsv(tmp_path, "Absent")
    with pytest.raises(FileNotFoundError):
        evaluation.list_ucr_datasets(tmp_path / "nope")

    root = tmp_path / "Bad"
    root.mkdir()
    np.savetxt(root / "Bad_TRAIN.tsv", np.array([[1.0, np.nan, 2.0]]), delimiter="\t")
    np.savetxt(root / "Bad_TEST.tsv", np.array([[1.0, 0.0, 2.0]]), delimiter="\t")
    with pytest.raises(ValueError, match="NaN"):
        evaluation.load_ucr_tsv(tmp_path, "Bad")

    mismatched = tmp_path / "Len"
    mismatched.mkdir()
    np.savetxt(mismatched / "Len_TRAIN.tsv", np.array([[1.0, 0.0, 2.0]]), delimiter="\t")
    np.savetxt(mismatched / "Len_TEST.tsv", np.array([[1.0, 0.0]]), delimiter="\t")
    with pytest.raises(ValueError, match="lengths differ"):
        evaluation.load_ucr_tsv(tmp_path, "Len")


def test_encode_dataset_shapes() -> None:
    ds = _tiny_dataset()
    raw = evaluation.encode_dataset(ds.X_train, evaluation.Method("m", "raw"))
    assert raw.shape == ds.X_train.shape
    np.testing.assert_allclose(raw.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(raw.std(axis=1), 1.0, atol=1e-12)

    subset = evaluation.Method("m", "gaf", bins=8, features=("intensity",))
    feats = evaluation.encode_dataset(ds.X_train, subset)
    assert feats.shape == (ds.X_train.shape[0], 6)

    with pytest.raises(ValueError, match="ROCKET"):
        evaluation.encode_dataset(ds.X_train, evaluation.Method("m", "rocket"))


def test_evaluate_returns_measurements() -> None:
    ds = _tiny_dataset()
    result = evaluation.evaluate(ds, evaluation.Method("nn", "raw", "knn1"))
    assert result.error == ""
    assert 0.0 <= result.accuracy <= 1.0
    assert result.n_features == ds.length
    assert result.encode_seconds >= 0.0
    assert result.peak_mib > 0.0
    assert result.n_train == ds.X_train.shape[0]


def test_evaluate_records_failures_instead_of_raising() -> None:
    ds = _tiny_dataset()
    result = evaluation.evaluate(ds, evaluation.Method("bad", "raw", "does-not-exist"))
    assert np.isnan(result.accuracy)
    assert "unknown classifier" in result.error


def test_run_benchmark_freezes_raw_outputs(tmp_path: Path) -> None:
    datasets = [_tiny_dataset("A", seed=0), _tiny_dataset("B", seed=1)]
    methods = (
        evaluation.Method("nn", "raw", "knn1"),
        evaluation.Method("logreg", "raw", "logreg"),
    )
    results = evaluation.run_benchmark(
        datasets, methods, seeds=(0, 1), out_dir=tmp_path / "run"
    )
    assert len(results) == 2 * 2 * 2

    with (tmp_path / "run" / "results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(results)
    assert {row["dataset"] for row in rows} == {"A", "B"}

    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["n_rows"] == len(results)
    assert manifest["packages"]["numpy"] == np.__version__
    assert manifest["seeds"] == [0, 1]
    assert "python" in manifest and "git_commit" in manifest


def test_accuracy_matrix_averages_seeds_and_drops_incomplete() -> None:
    def result(dataset: str, method: str, seed: int, acc: float) -> evaluation.EvaluationResult:
        return evaluation.EvaluationResult(
            dataset=dataset,
            method=method,
            representation="raw",
            classifier="knn1",
            seed=seed,
            accuracy=acc,
            n_features=1,
            encode_seconds=0.0,
            fit_seconds=0.0,
            predict_seconds=0.0,
            peak_mib=0.0,
            n_train=1,
            n_test=1,
            length=1,
            n_classes=2,
        )

    results = [
        result("A", "m1", 0, 0.8),
        result("A", "m1", 1, 0.6),
        result("A", "m2", 0, 0.5),
        # Dataset B is missing m2, so it must be dropped from the block design.
        result("B", "m1", 0, 0.9),
        result("B", "m2", 0, float("nan")),
    ]
    datasets, methods, acc = evaluation.accuracy_matrix(results)
    assert datasets == ["A"]
    assert methods == ["m1", "m2"]
    np.testing.assert_allclose(acc, [[0.7, 0.5]])


def test_compare_methods_ranks_a_consistent_winner() -> None:
    def result(dataset: str, method: str, acc: float) -> evaluation.EvaluationResult:
        return evaluation.EvaluationResult(
            dataset=dataset,
            method=method,
            representation="raw",
            classifier="knn1",
            seed=0,
            accuracy=acc,
            n_features=1,
            encode_seconds=0.0,
            fit_seconds=0.0,
            predict_seconds=0.0,
            peak_mib=0.0,
            n_train=1,
            n_test=1,
            length=1,
            n_classes=2,
        )

    rng = np.random.default_rng(0)
    results = []
    for i in range(12):
        base = float(rng.uniform(0.4, 0.6))
        results.append(result(f"D{i}", "good", base + 0.2))
        results.append(result(f"D{i}", "mid", base + 0.1))
        results.append(result(f"D{i}", "poor", base))
    comparison = evaluation.compare_methods(results)
    assert comparison.methods == ["good", "mid", "poor"]
    np.testing.assert_allclose(comparison.average_ranks, [1.0, 2.0, 3.0])
    assert comparison.friedman_pvalue < 1e-4
    assert comparison.critical_difference > 0.0
    good_vs_poor = next(
        e for e in comparison.pairwise if {e["method_a"], e["method_b"]} == {"good", "poor"}
    )
    assert good_vs_poor["wins_a"] == 12
    assert good_vs_poor["significant"]

    report = evaluation.summary_markdown(comparison)
    assert "Friedman" in report and "good" in report
    assert report.index("| good ") < report.index("| poor ")


def test_compare_methods_requires_a_complete_block() -> None:
    results = evaluation.run_benchmark(
        [_tiny_dataset()], (evaluation.Method("nn", "raw", "knn1"),)
    )
    with pytest.raises(ValueError, match="at least two"):
        evaluation.compare_methods(results)


def test_cli_synthetic_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = evaluation.main(
        [
            "--synthetic",
            "--out",
            str(tmp_path / "out"),
            "--methods",
            "baseline-1nn-euclidean",
            "baseline-raw-logreg",
        ]
    )
    assert code == 0
    assert (tmp_path / "out" / "results.csv").is_file()
    assert (tmp_path / "out" / "summary.md").is_file()
    assert "Benchmark summary" in capsys.readouterr().out


def test_cli_rejects_unknown_method(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        evaluation.main(["--synthetic", "--methods", "not-a-method"])
    assert "unknown method" in capsys.readouterr().err


def test_cli_requires_archive(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        evaluation.main([])
    assert "--archive is required" in capsys.readouterr().err
