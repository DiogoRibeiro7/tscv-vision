#!/usr/bin/env python
"""Run reproducible encoder benchmark diagnostics.

The script deliberately writes raw JSON instead of formatted prose. Reports can
summarise it later, but the measured values remain machine-readable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tscv_vision import encoders  # noqa: E402
from tscv_vision.benchmark import benchmark_time_frequency  # noqa: E402

Array = NDArray[np.float64]


def _measure(func: Callable[[], Array], *, repeats: int) -> dict[str, float]:
    """Return best runtime, peak memory, output density, and shape area."""

    best = float("inf")
    image: Array | None = None
    tracemalloc.start()
    for _ in range(max(1, repeats)):
        start = perf_counter()
        image = func()
        best = min(best, perf_counter() - start)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if image is None:  # pragma: no cover - repeats is clamped above
        raise RuntimeError("benchmark did not run")
    magnitude = np.abs(image)
    scale = float(magnitude.max()) if magnitude.size else 0.0
    density = 0.0 if scale <= 0.0 else float(np.mean(magnitude > 0.01 * scale))
    return {
        "seconds": best,
        "peak_mib": peak / (1024**2),
        "density_gt_1pct_peak": density,
        "output_values": float(magnitude.size),
    }


def _hvg_scaling(sizes: list[int], *, repeats: int) -> list[dict[str, float]]:
    """Measure horizontal visibility graph scaling on deterministic noise."""

    rng = np.random.default_rng(0)
    rows: list[dict[str, float]] = []
    for size in sizes:
        x = rng.standard_normal(size)
        metrics = _measure(
            lambda x=x: encoders.horizontal_visibility_graph(x),
            repeats=repeats,
        )
        metrics["n"] = float(size)
        rows.append(metrics)
    return rows


def run_suite(*, repeats: int, sizes: list[int]) -> dict[str, object]:
    """Run the encoder benchmark suite and return JSON-serialisable results."""

    fs = 200.0
    t = np.arange(1024, dtype=float) / fs
    chirp = np.sin(2.0 * np.pi * (12.0 * t + 0.5 * 18.0 * t**2))
    return {
        "schema_version": 1,
        "created_unix": int(time.time()),
        "time_frequency": benchmark_time_frequency(
            chirp,
            fs=fs,
            repeats=repeats,
            frequencies=64,
        ),
        "horizontal_visibility_scaling": _hvg_scaling(sizes, repeats=repeats),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="JSON output path")
    parser.add_argument("--repeats", type=int, default=3, help="timing repeats")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048],
        help="HVG input lengths",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use a tiny configuration intended for CI and local checks",
    )
    args = parser.parse_args(argv)

    sizes = [32, 64] if args.smoke else args.sizes
    results = run_suite(repeats=max(1, args.repeats), sizes=sizes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
