"""Test configuration for timeouts.

This fixture aborts tests that exceed the configured duration to avoid
hanging CI jobs. The timeout can be adjusted by setting the
``TSVISION_TEST_TIMEOUT`` environment variable (seconds).
"""

from __future__ import annotations

import os
import signal
from collections.abc import Iterator

import pytest

_DEFAULT_TIMEOUT = int(os.environ.get("TSVISION_TEST_TIMEOUT", "60"))


@pytest.fixture(autouse=True)
def timeout_guard() -> Iterator[None]:
    """Abort tests exceeding the global timeout."""

    if _DEFAULT_TIMEOUT <= 0:
        yield
        return

    def _handler(signum: int, frame: object) -> None:  # pragma: no cover - signal handler
        raise TimeoutError(f"test timed out after {_DEFAULT_TIMEOUT} s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(_DEFAULT_TIMEOUT)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
