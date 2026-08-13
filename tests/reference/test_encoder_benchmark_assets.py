from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_encoder_benchmark_smoke_writes_expected_sections(tmp_path: Path) -> None:
    """The committed encoder benchmark entry point stays runnable."""

    out = tmp_path / "encoder-smoke.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    subprocess.run(
        [
            sys.executable,
            str(REPO / "benchmarks" / "encoders" / "run_encoder_suite.py"),
            "--smoke",
            "--repeats",
            "1",
            "--out",
            str(out),
        ],
        cwd=REPO,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert set(data["time_frequency"]) == {
        "spectrogram",
        "cwt",
        "synchrosqueezed_cwt",
    }
    assert [row["n"] for row in data["horizontal_visibility_scaling"]] == [32.0, 64.0]
