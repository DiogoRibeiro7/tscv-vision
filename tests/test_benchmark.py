import numpy as np
import pytest

pytest.importorskip("sklearn")

from tscv_vision.benchmark import benchmark_pipeline, benchmark_streaming  # noqa: E402
from tscv_vision.pipeline import AdaptivePipeline  # noqa: E402
from tscv_vision.streaming import StreamingEncoder  # noqa: E402

pytestmark = pytest.mark.optional


def test_benchmark_metrics() -> None:
    enc = StreamingEncoder(size=2)
    samples = np.arange(10, dtype=float)
    stats = benchmark_streaming(enc, samples)
    assert set(stats) == {"throughput", "latency", "memory"}
    assert stats["memory"] == samples.nbytes

    stats_mem = benchmark_streaming(enc, samples, track_mem=True)
    assert "peak_mem_mib" in stats_mem
    assert stats_mem["memory"] == samples.nbytes


def test_benchmark_pipeline() -> None:
    rng = np.random.default_rng(0)
    X = rng.random((30, 4))
    y = rng.integers(0, 2, size=30)
    pipe = AdaptivePipeline(encoders=["gaf"], cv=2, random_state=0).fit(X, y)
    stats = benchmark_pipeline(pipe, X, track_mem=True)
    assert {"throughput", "latency"}.issubset(stats)
    assert "peak_mem_mib" in stats


def test_benchmark_time_frequency_reports_all_transforms() -> None:
    from tscv_vision.benchmark import benchmark_time_frequency

    results = benchmark_time_frequency(repeats=1, frequencies=32)
    assert set(results) == {"spectrogram", "cwt", "synchrosqueezed_cwt"}
    for metrics in results.values():
        assert metrics["seconds"] > 0.0
        assert metrics["peak_mib"] > 0.0
        assert 0.0 <= metrics["sparsity"] <= 1.0
        assert 0.0 < metrics["concentration"] <= 1.0
    # Reassignment is the point: energy lands in fewer bins than the CWT.
    assert (
        results["synchrosqueezed_cwt"]["concentration"] > results["cwt"]["concentration"]
    )
