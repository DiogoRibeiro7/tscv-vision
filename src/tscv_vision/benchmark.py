"""Benchmark utilities for streaming encoders and pipelines."""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable, Sequence
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


def _participation_ratio(image: Array) -> float:
    """Energy concentration: the share of total energy in the strongest bins.

    Larger means the same energy occupies fewer bins, which is exactly what
    reassignment is supposed to buy.
    """

    flat = np.abs(np.asarray(image, dtype=float)).ravel()
    total = flat.sum()
    if total <= 0:
        return 0.0
    normalised = flat / total
    return float(np.sum(normalised**2))


def benchmark_time_frequency(
    x: Array | None = None,
    *,
    fs: float = 200.0,
    repeats: int = 3,
    frequencies: int = 128,
    sparsity_threshold: float = 0.01,
) -> dict[str, dict[str, float]]:
    """Compare the spectrogram, CWT and synchrosqueezed CWT on one signal.

    Parameters
    ----------
    x:
        Input series. Defaults to a linear chirp, whose instantaneous
        frequency is known, so that concentration is measured on a signal the
        transforms are supposed to resolve.
    fs:
        Sampling frequency in Hz.
    repeats:
        Timing repetitions; the minimum is reported, being the least noisy
        estimator of the achievable time.
    frequencies:
        Output frequency bins for the transforms that take a grid.
    sparsity_threshold:
        Fraction of the peak below which a bin counts as empty.

    Returns
    -------
    dict
        ``{name: {"seconds", "peak_mib", "sparsity", "concentration"}}`` where
        ``sparsity`` is the fraction of bins below the threshold.
    """

    from . import encoders as _enc

    if x is None:
        t = np.arange(2048) / fs
        x = np.sin(2 * np.pi * (10.0 * t + 0.5 * 10.0 * t**2))
    series = np.asarray(x, dtype=float)
    scales = np.linspace(1.0, 64.0, frequencies)

    transforms: dict[str, Callable[[], Array]] = {
        "spectrogram": lambda: _enc.spectrogram(series, win=128, hop=16),
        "cwt": lambda: _enc.cwt(series, scales),
        "synchrosqueezed_cwt": lambda: _enc.synchrosqueezed_cwt(
            series, fs=fs, frequencies=frequencies
        ),
    }

    results: dict[str, dict[str, float]] = {}
    for name, func in transforms.items():
        tracemalloc.start()
        best = float("inf")
        image = None
        for _ in range(max(1, repeats)):
            start = perf_counter()
            image = func()
            best = min(best, perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert image is not None
        magnitude = np.abs(image)
        results[name] = {
            "seconds": best,
            "peak_mib": peak / (1024**2),
            "sparsity": float(np.mean(magnitude <= sparsity_threshold * magnitude.max())),
            "concentration": _participation_ratio(magnitude),
        }
    return results


def _time_best(func: Callable[[], object], repeats: int) -> tuple[float, object]:
    """Return the fastest of ``repeats` runs and the last result.

    The minimum is reported rather than the mean: the true cost is bounded
    below, and every source of noise on a shared machine adds time.
    """

    best = float("inf")
    out: object = None
    for _ in range(max(1, repeats)):
        start = perf_counter()
        out = func()
        best = min(best, perf_counter() - start)
    return best, out


def _peak_mib(func: Callable[[], object]) -> float:
    """Return the peak memory of one call, in MiB.

    Measured in its own pass. :mod:`tracemalloc` hooks every allocation, which
    inflates wall-clock substantially on allocation-heavy code, so timing it at
    the same time would report a number nobody can reproduce without the
    profiler attached.
    """

    already = tracemalloc.is_tracing()
    if not already:
        tracemalloc.start()
    try:
        func()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not already:
            tracemalloc.stop()
    return peak / (1024**2)


def benchmark_length_scaling(
    representations: Sequence[str] = ("gaf", "gadf", "mtf", "rp"),
    *,
    lengths: Sequence[int] = (128, 256, 512, 1024, 4096),
    repeats: int = 3,
    bins: int = 16,
    measure_features: bool = True,
    seed: int = 0,
) -> list[dict[str, float | str]]:
    """Measure encoder and feature cost as a function of input series length.

    Both stages of the benchmark pipeline are measured separately, because they
    do not scale alike: the encoders are ``O(N^2)`` in the length of the series,
    while feature extraction is linear in the *pixels* it is handed and so
    ``O(N^2)`` in the same variable, with a far larger constant.

    Timing and memory are measured in separate passes; see :func:`_peak_mib`.

    Parameters
    ----------
    representations:
        Registry keys to measure, resolved through
        :func:`~tscv_vision.encoders.get_encoder`.
    lengths:
        Input series lengths. Cost grows quadratically, so the largest entry
        dominates the total runtime of the sweep.
    repeats:
        Timing repetitions per cell; the fastest is reported.
    bins:
        Histogram bins passed to :func:`~tscv_vision.features.extract_feature_vector`.
    measure_features:
        When ``False``, only the encoder is measured and the feature columns are
        ``nan``. Useful when the image is too large to summarise.
    seed:
        Seed for the standard-normal input, so a rerun measures the same series.

    Returns
    -------
    list of dict
        One row per ``(representation, length)`` with encode and feature
        seconds, peak MiB, the number of values in the image and the length of
        the feature vector.
    """

    from .features import extract_feature_vector

    rows: list[dict[str, float | str]] = []
    for name in representations:
        encoder = encoders.get_encoder(name)
        for length in lengths:
            x = np.random.default_rng(seed).standard_normal(int(length))
            encode_seconds, image = _time_best(lambda x=x, enc=encoder: enc(x), repeats)
            assert isinstance(image, np.ndarray)
            row: dict[str, float | str] = {
                "representation": name,
                "length": float(length),
                "encode_seconds": encode_seconds,
                "encode_peak_mib": _peak_mib(lambda x=x, enc=encoder: enc(x)),
                "image_values": float(image.size),
                "feature_seconds": float("nan"),
                "feature_peak_mib": float("nan"),
                "n_features": float("nan"),
            }
            if measure_features:
                img = np.asarray(image, dtype=float)
                feature_seconds, vector = _time_best(
                    lambda img=img: extract_feature_vector(img, bins=bins), repeats
                )
                assert isinstance(vector, np.ndarray)
                row["feature_seconds"] = feature_seconds
                row["feature_peak_mib"] = _peak_mib(
                    lambda img=img: extract_feature_vector(img, bins=bins)
                )
                row["n_features"] = float(vector.size)
            rows.append(row)
    return rows


def scaling_exponent(lengths: Sequence[float], values: Sequence[float]) -> float:
    """Fit ``value ~ length**k`` on a log-log scale and return ``k``.

    This is the number to compare against the complexity string recorded in the
    representation metadata: an encoder documented as ``O(N^2)`` whose measured
    exponent is 1.0 is either mis-documented or not doing what it claims.

    Returns ``nan`` if fewer than two positive pairs are available.
    """

    pairs = [
        (float(n), float(v))
        for n, v in zip(lengths, values, strict=True)
        if n > 0 and v > 0 and np.isfinite(v)
    ]
    if len(pairs) < 2:
        return float("nan")
    log_n = np.log(np.array([p[0] for p in pairs]))
    log_v = np.log(np.array([p[1] for p in pairs]))
    slope, _ = np.polyfit(log_n, log_v, 1)
    return float(slope)


__all__ = [
    "benchmark_streaming",
    "benchmark_pipeline",
    "benchmark_encoder",
    "benchmark_sliding_gaf",
    "benchmark_time_frequency",
    "benchmark_length_scaling",
    "scaling_exponent",
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

