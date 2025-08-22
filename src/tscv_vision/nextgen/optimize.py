"""Self-optimising wrapper for plugins."""
from __future__ import annotations

import time
from typing import Callable


class SelfOptimizer:
    """Selects the fastest implementation among registered variants."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def choose(self, *variants: Callable[..., object]) -> Callable[..., object]:
        """Return the fastest callable based on observed runtimes."""

        best = None
        best_t = float("inf")
        for fn in variants:
            start = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - start
            if elapsed < best_t:
                best_t = elapsed
                best = fn
        assert best is not None  # pragma: no cover - sanity check
        return best
