"""Generate example arrays for tscv-vision.

This script creates a small sine wave and saves it as ``samples/sine.npy``.
It is intended for documentation and CLI demonstrations and should not be
committed to version control.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def main() -> None:
    path = Path(__file__).parent
    path.mkdir(exist_ok=True)
    x = np.sin(np.linspace(0, 4 * np.pi, 128))
    out = path / "sine.npy"
    np.save(out, x)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
