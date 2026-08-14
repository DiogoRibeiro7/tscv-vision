#!/usr/bin/env python
"""Freeze a runtime and peak-memory sweep over input series length.

Writes the same trio as the evaluation harness -- ``results.csv``,
``manifest.json`` and ``summary.md`` -- so a scaling claim is backed by raw rows
and a recorded environment rather than by a number quoted in prose.

The encoder and the feature extractor are timed separately. They are both
quadratic in the series length, but not with the same constant, and the table is
only useful if it says which stage the cost is actually in.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tscv_vision.benchmark import benchmark_length_scaling, scaling_exponent  # noqa: E402
from tscv_vision.evaluation import environment_manifest  # noqa: E402
from tscv_vision.representations import get_encoder_metadata  # noqa: E402

FIELDS = (
    "representation",
    "length",
    "encode_seconds",
    "encode_peak_mib",
    "image_values",
    "feature_seconds",
    "feature_peak_mib",
    "n_features",
)

DEFAULT_LENGTHS = (128, 256, 512, 1024, 4096)
DEFAULT_REPRESENTATIONS = ("gaf", "gadf", "mtf", "rp")


def _write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _grid(rows: list[dict[str, float | str]], column: str) -> dict[str, dict[float, float]]:
    grid: dict[str, dict[float, float]] = {}
    for row in rows:
        name = str(row["representation"])
        grid.setdefault(name, {})[float(row["length"])] = float(row[column])
    return grid


def _table(
    rows: list[dict[str, float | str]],
    column: str,
    lengths: list[float],
    *,
    fmt: str,
) -> list[str]:
    header = "| Representation | " + " | ".join(f"{int(n)}" for n in lengths) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in lengths) + " |"
    lines = [header, rule]
    for name, by_length in _grid(rows, column).items():
        cells = [format(by_length[n], fmt) if n in by_length else "-" for n in lengths]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return lines


def summary_markdown(rows: list[dict[str, float | str]], *, repeats: int) -> str:
    """Render the frozen rows as tables plus fitted scaling exponents."""

    lengths = sorted({float(r["length"]) for r in rows})
    out: list[str] = ["# Length-scaling summary", ""]
    out.append(
        f"{len(_grid(rows, 'encode_seconds'))} representations x {len(lengths)} "
        f"lengths, best of {repeats} timed runs per cell."
    )
    out.append("")
    out.append(
        "Timing and peak memory are measured in separate passes: `tracemalloc` "
        "hooks every allocation and inflates wall-clock on allocation-heavy "
        "code, so a number measured under it is not one anybody reproduces "
        "without the profiler attached."
    )
    out.append("")

    for title, column, fmt in (
        ("Encode time (seconds)", "encode_seconds", ".4f"),
        ("Encode peak memory (MiB)", "encode_peak_mib", ".1f"),
        ("Feature-extraction time (seconds)", "feature_seconds", ".4f"),
        ("Feature-extraction peak memory (MiB)", "feature_peak_mib", ".1f"),
    ):
        out.append(f"## {title}")
        out.append("")
        out.extend(_table(rows, column, lengths, fmt=fmt))
        out.append("")

    out.append("## Measured scaling exponents")
    out.append("")
    out.append(
        "Fitted as `value ~ length**k` on a log-log scale. Compare `k` against "
        "the complexity recorded in the representation metadata: that string is "
        "a claim, and this is the measurement of it."
    )
    out.append("")
    out.append(
        "| Representation | encode time | encode memory | feature time | "
        "feature memory | documented |"
    )
    out.append("| --- | ---: | ---: | ---: | ---: | --- |")
    grids = {c: _grid(rows, c) for c in FIELDS[2:]}
    for name in _grid(rows, "encode_seconds"):
        cells = []
        for column in ("encode_seconds", "encode_peak_mib", "feature_seconds", "feature_peak_mib"):
            by_length = grids[column][name]
            keys = sorted(by_length)
            cells.append(f"{scaling_exponent(keys, [by_length[k] for k in keys]):.2f}")
        try:
            documented = get_encoder_metadata(name).complexity
        except Exception:  # pragma: no cover - key absent from the registry
            documented = "-"
        out.append(f"| `{name}` | " + " | ".join(cells) + f" | `{documented}` |")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "length-scaling")
    parser.add_argument("--repeats", type=int, default=3, help="timing repeats per cell")
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument(
        "--representations", nargs="+", default=list(DEFAULT_REPRESENTATIONS)
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny configuration for CI; the output is not evidence",
    )
    args = parser.parse_args(argv)

    lengths = [32, 64] if args.smoke else args.lengths
    repeats = 1 if args.smoke else max(1, args.repeats)
    rows = benchmark_length_scaling(
        args.representations, lengths=lengths, repeats=repeats
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / "results.csv")
    manifest = environment_manifest(
        {
            "representations": list(args.representations),
            "lengths": list(lengths),
            "repeats": repeats,
            "n_rows": len(rows),
            "smoke": bool(args.smoke),
        },
        ignore_dirty_paths=(out_dir,),
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report = summary_markdown(rows, repeats=repeats)
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nRaw results: {out_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
