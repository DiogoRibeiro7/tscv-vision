"""Optional quantum-enhanced encoders."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - optional dependency
    from qiskit import QuantumCircuit
except Exception:  # pragma: no cover - dependency missing
    QuantumCircuit = None


def quantum_fourier_encoder(
    signal: NDArray[np.floating[Any]]
) -> NDArray[np.floating[Any]]:
    """Encode a signal using the Quantum Fourier Transform.

    Requires ``qiskit``; otherwise raises ``ImportError``.
    """

    if QuantumCircuit is None:  # pragma: no cover - runtime guard
        raise ImportError("qiskit is required for quantum encoders")
    n = int(np.ceil(np.log2(len(signal))))
    qc = QuantumCircuit(n)
    # Placeholder: real implementation would load data into amplitudes
    qc.h(range(n))
    # Return deterministic dummy vector to avoid heavy deps
    return np.abs(np.fft.fft(signal))
