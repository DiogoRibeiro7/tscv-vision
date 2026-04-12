"""Test configuration for timeouts.

This fixture aborts tests that exceed the configured duration to avoid
hanging CI jobs. The timeout can be adjusted by setting the
``TSVISION_TEST_TIMEOUT`` environment variable (seconds).
"""

from __future__ import annotations

import importlib.util
import os
import signal
from collections.abc import Iterator

import pytest

_DEFAULT_TIMEOUT = int(os.environ.get("TSVISION_TEST_TIMEOUT", "60"))


@pytest.fixture(autouse=True)
def timeout_guard() -> Iterator[None]:
    """Abort tests exceeding the global timeout."""

    if _DEFAULT_TIMEOUT <= 0 or not hasattr(signal, "SIGALRM"):
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


def _has_pkg(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip marked tests when optional deps are missing."""

    if "gpu" in item.keywords and not _has_pkg("cupy"):
        pytest.skip("requires CuPy")


@pytest.fixture
def torch_module() -> object:
    """Provide the torch module if installed."""

    return pytest.importorskip("torch")


@pytest.fixture
def cupy_module() -> object:
    """Provide the cupy module if installed."""

    return pytest.importorskip("cupy")


@pytest.fixture
def sklearn_module() -> object:
    """Provide the scikit-learn module if installed."""

    return pytest.importorskip("sklearn")
