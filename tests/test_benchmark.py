import numpy as np
import pytest

pytest.importorskip("sklearn")

from tscv_vision.benchmark import benchmark_pipeline, benchmark_streaming
from tscv_vision.pipeline import AdaptivePipeline
from tscv_vision.streaming import StreamingEncoder

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
