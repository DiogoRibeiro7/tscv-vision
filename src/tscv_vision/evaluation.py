"""Leakage-safe classification benchmark over UCR/UEA-style datasets.

The module provides the pieces a comparative study needs and nothing more:

* loading archive datasets with their **predefined train/test split**, so no
  choice is ever informed by test data;
* a small set of representations (raw series, any registered encoder plus image
  features, and optionally ROCKET via :mod:`pyts`) and classifiers;
* wall-clock and peak-memory measurement per fit/predict;
* frozen raw outputs — one CSV row per ``(dataset, method, seed)`` plus a
  manifest recording package versions and the git commit;
* multi-dataset comparison statistics (Friedman, average ranks, Nemenyi
  critical difference, Holm-corrected pairwise Wilcoxon) from
  :mod:`tscv_vision.stats`, following Demšar (2006).

Nothing here reports a number that any part of the pipeline has already seen.
Hyper-parameter choices, if any, must be made with
:meth:`tscv_vision.pipeline.AdaptivePipeline.nested_score` or on the training
split only.

Example
-------
Run the full sweep over a local copy of the UCR archive::

    python -m tscv_vision.evaluation --archive /data/UCRArchive_2018 \\
        --out results/ucr --datasets-file datasets.txt
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import __version__
from . import stats as ts_stats
from .encoders import get_encoder
from .features import extract_feature_vector

Array = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Datasets


@dataclass(frozen=True)
class TSDataset:
    """A univariate classification dataset with its archive train/test split."""

    name: str
    X_train: Array
    y_train: NDArray[np.int64]
    X_test: Array
    y_test: NDArray[np.int64]

    @property
    def n_classes(self) -> int:
        """Number of distinct labels in the training split."""
        return int(np.unique(self.y_train).size)

    @property
    def length(self) -> int:
        """Series length."""
        return int(self.X_train.shape[1])

    def summary(self) -> dict[str, int]:
        """Return the sizes used to describe the dataset in a results table."""
        return {
            "n_train": int(self.X_train.shape[0]),
            "n_test": int(self.X_test.shape[0]),
            "length": self.length,
            "n_classes": self.n_classes,
        }


def _read_ucr_tsv(path: Path) -> tuple[Array, NDArray[np.int64]]:
    raw = np.loadtxt(path, delimiter="\t")
    if raw.ndim == 1:
        raw = raw[None, :]
    labels = raw[:, 0].astype(np.int64)
    series = np.ascontiguousarray(raw[:, 1:], dtype=float)
    return series, labels


def load_ucr_tsv(archive: str | Path, name: str) -> TSDataset:
    """Load one dataset from a local ``UCRArchive_2018``-style directory.

    Expects ``<archive>/<name>/<name>_TRAIN.tsv`` and ``..._TEST.tsv``, the
    layout distributed by the UCR/UEA time-series classification archive.

    Parameters
    ----------
    archive:
        Root directory of the archive.
    name:
        Dataset name (directory name).

    Returns
    -------
    TSDataset
        The dataset with its predefined split.

    Raises
    ------
    FileNotFoundError
        If either split file is missing.
    ValueError
        If the two splits disagree on series length, or the data contain NaN.
    """

    root = Path(archive) / name
    train_path = root / f"{name}_TRAIN.tsv"
    test_path = root / f"{name}_TEST.tsv"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing split file: {path}")
    X_train, y_train = _read_ucr_tsv(train_path)
    X_test, y_test = _read_ucr_tsv(test_path)
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"{name}: train/test series lengths differ "
            f"({X_train.shape[1]} vs {X_test.shape[1]})"
        )
    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(X_test)):
        raise ValueError(
            f"{name}: contains NaN/inf; impute before benchmarking so the "
            "imputation strategy is an explicit, reported choice"
        )
    return TSDataset(name, X_train, y_train, X_test, y_test)


def list_ucr_datasets(archive: str | Path) -> list[str]:
    """Return the sorted names of every loadable dataset under ``archive``."""

    root = Path(archive)
    if not root.is_dir():
        raise FileNotFoundError(f"archive directory not found: {root}")
    names = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / f"{child.name}_TRAIN.tsv").is_file():
            names.append(child.name)
    return names


def make_synthetic_dataset(
    name: str = "Synthetic",
    *,
    n_per_class: int = 30,
    length: int = 64,
    n_classes: int = 2,
    seed: int = 0,
) -> TSDataset:
    """Build a small separable dataset for smoke-testing the harness.

    .. warning::
       This exists so the pipeline can be exercised without downloading the
       archive. Numbers obtained on it are **not** evidence of anything and
       must never appear in a results table.
    """

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, length)
    series, labels = [], []
    for cls in range(n_classes):
        freq = 1.0 + cls
        for _ in range(2 * n_per_class):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            noise = rng.normal(scale=0.1, size=length)
            series.append(np.sin(freq * t + phase) + noise)
            labels.append(cls)
    X = np.asarray(series, dtype=float)
    y = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(X.shape[0])
    X, y = X[order], y[order]
    half = X.shape[0] // 2
    return TSDataset(name, X[:half], y[:half], X[half:], y[half:])


# ---------------------------------------------------------------------------
# Methods


@dataclass(frozen=True)
class Method:
    """One representation + classifier combination to benchmark.

    Parameters
    ----------
    name:
        Identifier used in the results table.
    representation:
        ``"raw"`` for the z-normalised series, ``"rocket"`` for the ROCKET
        transform (requires :mod:`pyts`), or the name of any encoder in
        :data:`tscv_vision.encoders.ENCODER_REGISTRY`, whose image is described
        by :func:`~tscv_vision.features.extract_feature_vector`.
    classifier:
        ``"knn1"`` (the standard UCR baseline), ``"logreg"``, ``"ridge"`` or
        ``"rf"``.
    bins:
        Histogram bins for the image features.
    features:
        Optional subset of feature extractors; ``None`` uses all available.
    """

    name: str
    representation: str
    classifier: str = "knn1"
    bins: int = 16
    features: tuple[str, ...] | None = None


#: A defensible default sweep: the standard baseline, a raw-feature control,
#: the classic imaging encoders, and an ablation over feature subsets.
DEFAULT_METHODS: tuple[Method, ...] = (
    Method("baseline-1nn-euclidean", "raw", "knn1"),
    Method("baseline-raw-logreg", "raw", "logreg"),
    Method("gasf-features", "gaf", "logreg"),
    Method("gadf-features", "gadf", "logreg"),
    Method("mtf-features", "mtf", "logreg"),
    Method("rp-features", "rp", "logreg"),
    Method("ablation-gaf-intensity-only", "gaf", "logreg", features=("intensity",)),
    Method("ablation-gaf-texture-only", "gaf", "logreg", features=("lbp", "glcm")),
)


def _znorm(X: Array) -> Array:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return np.asarray((X - mean) / np.where(std > 0, std, 1.0), dtype=float)


def encode_dataset(
    X: Array,
    method: Method,
    *,
    rocket_transform: Any | None = None,
) -> Array:
    """Turn raw series into the feature matrix required by ``method``.

    Raises
    ------
    ValueError
        If ``method.representation`` is ``"rocket"`` but no fitted transform is
        supplied.
    """

    if method.representation == "raw":
        return _znorm(X)
    if method.representation == "rocket":
        if rocket_transform is None:
            raise ValueError("a fitted ROCKET transform is required")
        return np.asarray(rocket_transform.transform(_znorm(X)), dtype=float)
    encoder = get_encoder(method.representation)
    feats = [
        extract_feature_vector(
            encoder(row),
            bins=method.bins,
            selected=list(method.features) if method.features else None,
        )
        for row in _znorm(X)
    ]
    return np.vstack(feats)


def _build_classifier(kind: str, seed: int) -> Any:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("scikit-learn is required for the benchmark") from exc

    if kind == "knn1":
        return KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed),
        )
    if kind == "ridge":
        return make_pipeline(
            StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        )
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    raise ValueError(f"unknown classifier {kind!r}")


# ---------------------------------------------------------------------------
# Running


@dataclass
class EvaluationResult:
    """One ``(dataset, method, seed)`` measurement."""

    dataset: str
    method: str
    representation: str
    classifier: str
    seed: int
    accuracy: float
    n_features: int
    encode_seconds: float
    fit_seconds: float
    predict_seconds: float
    peak_mib: float
    n_train: int
    n_test: int
    length: int
    n_classes: int
    error: str = ""


def evaluate(dataset: TSDataset, method: Method, *, seed: int = 0) -> EvaluationResult:
    """Fit ``method`` on the training split and score it on the test split.

    The test split is touched exactly once, for scoring. Feature extraction is
    stateless per series, and the ROCKET transform (when used) is fitted on the
    training split only.

    Returns
    -------
    EvaluationResult
        Accuracy plus runtime and peak-memory measurements. If the run raises,
        the exception message is recorded in ``error`` and ``accuracy`` is NaN,
        so one broken dataset cannot abort a long sweep.
    """

    info = dataset.summary()
    base: dict[str, Any] = {
        "dataset": dataset.name,
        "method": method.name,
        "representation": method.representation,
        "classifier": method.classifier,
        "seed": seed,
        **info,
    }
    try:
        rocket = None
        if method.representation == "rocket":
            try:
                from pyts.transformation import ROCKET
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError("pyts is required for the ROCKET baseline") from exc
            rocket = ROCKET(random_state=seed).fit(_znorm(dataset.X_train))

        tracemalloc.start()
        t0 = time.perf_counter()
        train_feats = encode_dataset(dataset.X_train, method, rocket_transform=rocket)
        test_feats = encode_dataset(dataset.X_test, method, rocket_transform=rocket)
        encode_seconds = time.perf_counter() - t0

        clf = _build_classifier(method.classifier, seed)
        t0 = time.perf_counter()
        clf.fit(train_feats, dataset.y_train)
        fit_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = clf.predict(test_feats)
        predict_seconds = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return EvaluationResult(
            accuracy=float(np.mean(pred == dataset.y_test)),
            n_features=int(train_feats.shape[1]),
            encode_seconds=encode_seconds,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            peak_mib=peak / (1024**2),
            **base,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        return EvaluationResult(
            accuracy=float("nan"),
            n_features=0,
            encode_seconds=float("nan"),
            fit_seconds=float("nan"),
            predict_seconds=float("nan"),
            peak_mib=float("nan"),
            error=f"{type(exc).__name__}: {exc}",
            **base,
        )


def environment_manifest(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record everything needed to reproduce a run.

    Includes Python and OS versions, the versions of every package that can
    change a number, and the current git commit (marked dirty if the working
    tree has uncommitted changes).
    """

    import importlib.metadata as md

    versions: dict[str, str] = {}
    for pkg in (
        "tscv-vision",
        "numpy",
        "scikit-learn",
        "scipy",
        "pywavelets",
        "kymatio",
        "pyts",
        "numba",
        "scikit-image",
    ):
        try:
            versions[pkg] = md.version(pkg)
        except Exception:  # pragma: no cover - package absent
            versions[pkg] = __version__ if pkg == "tscv-vision" else "not installed"

    commit = "unknown"
    dirty = None
    try:  # pragma: no cover - depends on git availability
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except Exception:  # pragma: no cover - not a git checkout
        pass

    manifest: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": versions,
        "git_commit": commit,
        "git_dirty": dirty,
    }
    if extra:
        manifest.update(extra)
    return manifest


_RESULT_FIELDS = tuple(EvaluationResult.__dataclass_fields__)


def run_benchmark(
    datasets: Sequence[TSDataset],
    methods: Sequence[Method] | None = None,
    *,
    seeds: Sequence[int] = (0,),
    out_dir: str | Path | None = None,
) -> list[EvaluationResult]:
    """Evaluate every ``(dataset, method, seed)`` combination.

    Parameters
    ----------
    datasets:
        Datasets with predefined splits.
    methods:
        Methods to compare; ``None`` resolves :data:`DEFAULT_METHODS` at call
        time.
    seeds:
        Seeds for the classifiers; repeated seeds quantify run-to-run variance
        for the non-deterministic ones.
    out_dir:
        If given, ``results.csv`` and ``manifest.json`` are written there. The
        CSV is the frozen raw output — every row, including failures.

    Returns
    -------
    list[EvaluationResult]
        One entry per combination, in a deterministic order.
    """

    if methods is None:
        methods = DEFAULT_METHODS
    results: list[EvaluationResult] = []
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                results.append(evaluate(dataset, method, seed=seed))
    if out_dir is not None:
        write_results(results, out_dir, methods=methods, datasets=datasets, seeds=seeds)
    return results


def write_results(
    results: Sequence[EvaluationResult],
    out_dir: str | Path,
    *,
    methods: Sequence[Method] | None = None,
    datasets: Sequence[TSDataset] | None = None,
    seeds: Sequence[int] | None = None,
) -> Path:
    """Freeze ``results`` to ``out_dir`` and return the CSV path."""

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_RESULT_FIELDS))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    manifest = environment_manifest(
        {
            "methods": [asdict(m) for m in methods] if methods else None,
            "datasets": [d.name for d in datasets] if datasets else None,
            "seeds": list(seeds) if seeds else None,
            "n_rows": len(results),
        }
    )
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return csv_path


# ---------------------------------------------------------------------------
# Comparison


@dataclass
class Comparison:
    """Multi-dataset comparison of the benchmarked methods."""

    methods: list[str]
    datasets: list[str]
    accuracy: Array
    mean_accuracy: Array
    average_ranks: Array
    friedman_statistic: float
    friedman_pvalue: float
    critical_difference: float
    pairwise: list[dict[str, Any]] = field(default_factory=list)


def accuracy_matrix(
    results: Sequence[EvaluationResult],
) -> tuple[list[str], list[str], Array]:
    """Pivot ``results`` into a ``(n_datasets, n_methods)`` accuracy matrix.

    Seeds are averaged. Datasets where any method failed are dropped, because
    the multi-dataset tests need a complete block design.
    """

    methods = sorted({r.method for r in results})
    datasets = sorted({r.dataset for r in results})
    acc = np.full((len(datasets), len(methods)), np.nan)
    counts = np.zeros_like(acc)
    m_idx = {m: i for i, m in enumerate(methods)}
    d_idx = {d: i for i, d in enumerate(datasets)}
    for r in results:
        i, j = d_idx[r.dataset], m_idx[r.method]
        if np.isnan(r.accuracy):
            continue
        acc[i, j] = r.accuracy if np.isnan(acc[i, j]) else acc[i, j] + r.accuracy
        counts[i, j] += 1
    with np.errstate(invalid="ignore"):
        acc = np.where(counts > 0, acc / np.maximum(counts, 1), np.nan)
    complete = ~np.any(np.isnan(acc), axis=1)
    return [d for d, keep in zip(datasets, complete, strict=True) if keep], methods, acc[complete]


def compare_methods(
    results: Sequence[EvaluationResult], *, alpha: float = 0.05
) -> Comparison:
    """Rank methods across datasets using Demšar-style non-parametric checks.

    Runs a Friedman test over the complete block of datasets, reports average
    ranks and the Nemenyi critical difference, and adds Holm-corrected
    pairwise Wilcoxon signed-rank tests.

    Raises
    ------
    ValueError
        If fewer than two methods or two complete datasets remain.
    """

    datasets, methods, acc = accuracy_matrix(results)
    if acc.shape[0] < 2 or acc.shape[1] < 2:
        raise ValueError(
            "need at least two methods and two datasets with complete results; "
            f"got {acc.shape[0]} datasets x {acc.shape[1]} methods"
        )
    friedman = ts_stats.friedman_test(acc)
    try:
        cd = ts_stats.nemenyi_critical_difference(acc.shape[1], acc.shape[0], alpha=alpha)
    except ValueError:  # pragma: no cover - >20 methods
        cd = float("nan")

    pairwise: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            diff = acc[:, i] - acc[:, j]
            if np.allclose(diff, 0.0):
                pvalue = 1.0
            else:
                pvalue = ts_stats.wilcoxon_signed_rank(diff).pvalue
            raw_p.append(pvalue)
            pairwise.append(
                {
                    "method_a": methods[i],
                    "method_b": methods[j],
                    "mean_difference": float(np.mean(diff)),
                    "wins_a": int(np.sum(diff > 0)),
                    "wins_b": int(np.sum(diff < 0)),
                    "ties": int(np.sum(diff == 0)),
                    "pvalue": float(pvalue),
                }
            )
    if raw_p:
        for entry, adj in zip(
            pairwise, ts_stats.holm_bonferroni(np.asarray(raw_p)), strict=True
        ):
            entry["pvalue_holm"] = float(adj)
            entry["significant"] = bool(adj < alpha)

    return Comparison(
        methods=methods,
        datasets=datasets,
        accuracy=acc,
        mean_accuracy=np.asarray(acc.mean(axis=0), dtype=float),
        average_ranks=friedman.ranks,
        friedman_statistic=friedman.statistic,
        friedman_pvalue=friedman.pvalue,
        critical_difference=cd,
        pairwise=pairwise,
    )


def summary_markdown(comparison: Comparison) -> str:
    """Render ``comparison`` as a Markdown report."""

    lines = [
        "# Benchmark summary",
        "",
        f"{len(comparison.datasets)} datasets x {len(comparison.methods)} methods.",
        "",
        f"Friedman chi-square = {comparison.friedman_statistic:.3f}, "
        f"p = {comparison.friedman_pvalue:.4g}. "
        f"Nemenyi critical difference (alpha=0.05) = {comparison.critical_difference:.3f}.",
        "",
        "| Method | Mean accuracy | Average rank |",
        "| --- | ---: | ---: |",
    ]
    order = np.argsort(comparison.average_ranks)
    for idx in order:
        lines.append(
            f"| {comparison.methods[idx]} | {comparison.mean_accuracy[idx]:.4f} "
            f"| {comparison.average_ranks[idx]:.2f} |"
        )
    lines += [
        "",
        "## Pairwise Wilcoxon signed-rank (Holm-corrected)",
        "",
        "| A | B | mean diff | wins A | wins B | ties | p | p (Holm) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in sorted(comparison.pairwise, key=lambda e: e.get("pvalue_holm", 1.0)):
        lines.append(
            f"| {entry['method_a']} | {entry['method_b']} "
            f"| {entry['mean_difference']:+.4f} | {entry['wins_a']} | {entry['wins_b']} "
            f"| {entry['ties']} | {entry['pvalue']:.4g} "
            f"| {entry.get('pvalue_holm', float('nan')):.4g} |"
        )
    lines += [
        "",
        "Accuracies come from the archive's predefined train/test split; the "
        "test split is used only for the final score.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: run the sweep and write frozen outputs."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--archive", help="root of a local UCRArchive_2018-style directory"
    )
    parser.add_argument(
        "--datasets", nargs="*", help="dataset names (default: every dataset found)"
    )
    parser.add_argument(
        "--datasets-file", help="file with one dataset name per line"
    )
    parser.add_argument("--out", default="results/benchmark", help="output directory")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument(
        "--methods",
        nargs="*",
        help=f"subset of {', '.join(m.name for m in DEFAULT_METHODS)}",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="run on generated data to smoke-test the harness (not evidence)",
    )
    args = parser.parse_args(argv)

    if args.synthetic:
        # Deliberately tiny: this path exists to prove the harness runs, so it
        # must stay fast enough for CI.
        datasets = [
            make_synthetic_dataset(
                f"Synthetic{i}", seed=i, n_per_class=10, length=48, n_classes=2 + i % 2
            )
            for i in range(3)
        ]
    else:
        if not args.archive:
            parser.error("--archive is required unless --synthetic is given")
        names = list(args.datasets or [])
        if args.datasets_file:
            names += [
                line.strip()
                for line in Path(args.datasets_file).read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
        if not names:
            names = list_ucr_datasets(args.archive)
        datasets = [load_ucr_tsv(args.archive, name) for name in names]

    methods: Sequence[Method] = DEFAULT_METHODS
    if args.methods:
        by_name = {m.name: m for m in DEFAULT_METHODS}
        unknown = [name for name in args.methods if name not in by_name]
        if unknown:
            parser.error(f"unknown method(s): {', '.join(unknown)}")
        methods = [by_name[name] for name in args.methods]

    results = run_benchmark(
        datasets, methods, seeds=tuple(args.seeds), out_dir=args.out
    )
    failures = [r for r in results if r.error]
    for failure in failures:
        print(f"FAILED {failure.dataset}/{failure.method}: {failure.error}")
    comparison = compare_methods(results)
    report = summary_markdown(comparison)
    Path(args.out, "summary.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nRaw results: {Path(args.out, 'results.csv')}")
    return 1 if failures else 0


__all__ = [
    "TSDataset",
    "Method",
    "EvaluationResult",
    "Comparison",
    "DEFAULT_METHODS",
    "load_ucr_tsv",
    "list_ucr_datasets",
    "make_synthetic_dataset",
    "encode_dataset",
    "evaluate",
    "run_benchmark",
    "write_results",
    "environment_manifest",
    "accuracy_matrix",
    "compare_methods",
    "summary_markdown",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
