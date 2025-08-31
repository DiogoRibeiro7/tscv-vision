"""Benchmark utilities for streaming encoders and pipelines."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from .pipeline import AdaptivePipeline
from .streaming import StreamingEncoder

Array = NDArray[np.float64]


def benchmark_streaming(
    encoder: StreamingEncoder,
    samples: Array,
    *,
    repeats: int = 1,
) -> dict[str, float]:
    """Measure throughput, latency, and memory usage for ``encoder``.

    Parameters
    ----------
    encoder:
        ``StreamingEncoder`` instance to benchmark. Its buffer is cleared before
        benchmarking.
    samples:
        Sample array to push through the encoder.
    repeats:
        Number of times to stream ``samples``; more repeats improve accuracy.

    Returns
    -------
    dict
        ``{"throughput": samples_per_sec, "latency": sec_per_sample, "memory": bytes}``
    """

    total = samples.size * repeats
    encoder._buf.clear()
    start = perf_counter()
    for _ in range(repeats):
        for s in samples:
            encoder.push(float(s))
    elapsed = perf_counter() - start
    throughput = total / elapsed if elapsed > 0 else float("inf")
    latency = elapsed / total if total > 0 else 0.0
    memory = float(samples.nbytes)
    return {"throughput": throughput, "latency": latency, "memory": memory}


def benchmark_pipeline(
    pipeline: AdaptivePipeline,
    X: Array,
    y: Array | None = None,
    *,
    repeats: int = 1,
) -> dict[str, float]:
    """Measure transform throughput and latency for ``pipeline``.

    Parameters
    ----------
    pipeline:
        Fitted :class:`AdaptivePipeline` instance.
    X:
        Input series ``(n_samples, series_len)``.
    y:
        Optional target values; if provided the pipeline is refit each repeat.
    repeats:
        Number of repeated runs.

    Returns
    -------
    dict
        ``{"throughput": samples_per_sec, "latency": sec_per_sample}``
    """

    X = np.asarray(X, dtype=float)
    total = X.shape[0] * repeats
    start = perf_counter()
    for _ in range(repeats):
        if y is not None:
            pipeline.fit(X, y)
        pipeline.transform(X)
    elapsed = perf_counter() - start
    throughput = total / elapsed if elapsed > 0 else float("inf")
    latency = elapsed / total if total > 0 else 0.0
    return {"throughput": throughput, "latency": latency}


__all__ = ["benchmark_streaming", "benchmark_pipeline"]

