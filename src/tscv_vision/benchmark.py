"""Benchmark utilities for streaming encoders and pipelines."""

from __future__ import annotations

import tracemalloc
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from . import encoders
from .pipeline import AdaptivePipeline
from .streaming import StreamingEncoder

Array = NDArray[np.float64]


def benchmark_streaming(
    encoder: StreamingEncoder,
    samples: Array,
    *,
    repeats: int = 1,
    track_mem: bool = False,
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
    track_mem:
        When ``True`` use :mod:`tracemalloc` to record peak memory in MiB.

    Returns
    -------
    dict
        ``{"throughput": samples_per_sec, "latency": sec_per_sample, "memory": bytes}``
    """

    total = samples.size * repeats
    encoder._buf.clear()
    if track_mem:
        tracemalloc.start()
    start = perf_counter()
    for _ in range(repeats):
        for s in samples:
            encoder.push(float(s))
    elapsed = perf_counter() - start
    throughput = total / elapsed if elapsed > 0 else float("inf")
    latency = elapsed / total if total > 0 else 0.0
    memory = float(samples.nbytes)
    stats = {"throughput": throughput, "latency": latency, "memory": memory}
    if track_mem:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats["peak_mem_mib"] = peak / (1024 ** 2)
    return stats


def benchmark_pipeline(
    pipeline: AdaptivePipeline,
    X: Array,
    y: Array | None = None,
    *,
    repeats: int = 1,
    track_mem: bool = False,
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
    track_mem:
        When ``True`` record peak memory in MiB via :mod:`tracemalloc`.

    Returns
    -------
    dict
        ``{"throughput": samples_per_sec, "latency": sec_per_sample}``
    """

    X = np.asarray(X, dtype=float)
    total = X.shape[0] * repeats
    if track_mem:
        tracemalloc.start()
    start = perf_counter()
    for _ in range(repeats):
        if y is not None:
            pipeline.fit(X, y)
        pipeline.transform(X)
    elapsed = perf_counter() - start
    throughput = total / elapsed if elapsed > 0 else float("inf")
    latency = elapsed / total if total > 0 else 0.0
    stats = {"throughput": throughput, "latency": latency}
    if track_mem:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats["peak_mem_mib"] = peak / (1024 ** 2)
    return stats


def benchmark_encoder(
    name: str,
    x: Array,
    *,
    use_numba: bool = False,
    use_gpu: bool = False,
    repeats: int = 3,
    track_mem: bool = False,
) -> dict[str, float]:
    """Benchmark an encoder before and after optimisation.

    Parameters
    ----------
    name:
        Name of encoder, e.g. ``"gaf"`` or ``"rp"``.
    x:
        Input 1D series.
    use_numba:
        Whether to run the optimised path (``use_numba=True``).
    use_gpu:
        Whether to benchmark the GPU implementation if available.
    repeats:
        Number of repetitions for timing.
    track_mem:
        When ``True`` record peak memory in MiB via :mod:`tracemalloc`.

    Returns
    -------
    dict
        ``{"baseline": seconds, "optimised": seconds}``
    """

    func = getattr(encoders, name)
    if track_mem:
        tracemalloc.start()
    start = perf_counter()
    for _ in range(repeats):
        func(x)
    base = (perf_counter() - start) / repeats

    if use_gpu:
        start = perf_counter()
        for _ in range(repeats):
            func(x, use_gpu=True)
        gpu_time = (perf_counter() - start) / repeats
    else:
        gpu_time = base

    if use_numba:
        start = perf_counter()
        for _ in range(repeats):
            func(x, use_numba=True)
        opt = (perf_counter() - start) / repeats
    else:
        opt = base
    stats = {"baseline": base, "optimised": opt, "gpu": gpu_time}
    if track_mem:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats["peak_mem_mib"] = peak / (1024 ** 2)
    return stats


def benchmark_sliding_gaf(
    length: int = 100_000,
    *,
    size: int = 256,
    hop: int = 32,
    repeats: int = 3,
    use_gpu: bool = True,
) -> dict[str, float | None]:
    """Benchmark ``encode_sliding`` with GAF on CPU vs batched GPU path.

    Parameters
    ----------
    length:
        Length of the input series.
    size, hop:
        Sliding window parameters.
    repeats:
        Number of timing repetitions per path.
    use_gpu:
        When ``False`` skip the GPU timing and return ``gpu=None``. Useful on
        machines without CuPy installed.

    Returns
    -------
    dict
        ``{"cpu": seconds, "gpu": seconds | None, "speedup": float | None,
        "n_windows": int}``.
    """

    from .sliding import encode_sliding

    rng = np.random.default_rng(0)
    x = rng.standard_normal(length)

    start = perf_counter()
    for _ in range(repeats):
        imgs, _ = encode_sliding(x, encoder="gaf", size=size, hop=hop)
    cpu_time = (perf_counter() - start) / repeats
    n_windows = imgs.shape[0]

    gpu_time: float | None = None
    if use_gpu:
        try:
            start = perf_counter()
            for _ in range(repeats):
                encode_sliding(
                    x, encoder="gaf", size=size, hop=hop, use_gpu=True
                )
            gpu_time = (perf_counter() - start) / repeats
        except RuntimeError:
            gpu_time = None

    speedup = cpu_time / gpu_time if gpu_time and gpu_time > 0 else None
    return {
        "cpu": cpu_time,
        "gpu": gpu_time,
        "speedup": speedup,
        "n_windows": float(n_windows),
    }


__all__ = [
    "benchmark_streaming",
    "benchmark_pipeline",
    "benchmark_encoder",
    "benchmark_sliding_gaf",
]


def main() -> None:
    """CLI entry point for benchmarking encoders.

    The command line interface is intentionally minimal and primarily used for
    manual profiling. Example usage::

        python -m tscv_vision.benchmark gaf 1024 --mem

    Parameters
    ----------
    Encoder name and series length are positional arguments. ``--mem`` enables
    peak memory reporting via :mod:`tracemalloc`.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Benchmark an encoder")
    parser.add_argument("encoder", help="encoder name, e.g. gaf or rp")
    parser.add_argument("length", type=int, help="input series length")
    parser.add_argument("--mem", action="store_true", help="record peak memory")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    x = rng.random(args.length)
    stats = benchmark_encoder(args.encoder, x, track_mem=args.mem)
    print(stats)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()

