#!/usr/bin/env python
"""Regenerate ``docs/encoder_validation.md`` from the representation metadata.

The matrix is generated so that it cannot drift from the code:
``tests/test_representations.py::test_validation_matrix_doc_is_current`` fails
if the committed file differs from what this script would write.

Usage::

    python scripts/generate_encoder_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tscv_vision.representations.metadata import validation_matrix_markdown  # noqa: E402


def main() -> int:
    """Write the matrix and report the path."""

    target = ROOT / "docs" / "encoder_validation.md"
    target.write_text(validation_matrix_markdown(), encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
