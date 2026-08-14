# Length-scaling run

This directory freezes a runtime and peak-memory sweep over input series length
for the four image-style encoders in `evaluation.DEFAULT_METHODS`. It is the
evidence behind the package's cost claims; it says nothing about accuracy.

Recreate it from the repository root:

```bash
python benchmarks/scaling/run_length_scaling.py --repeats 3
```

Grid:

- 4 representations: `gaf`, `gadf`, `mtf`, `rp`
- 5 lengths: 128, 256, 512, 1024, 4096
- best of 3 timed runs per cell, with peak memory measured in a separate pass

## What it shows

Three findings, all in `summary.md`:

- **The encoder is not the cost.** At length 4096 the encoders take 0.046–0.157 s
  while summarising their output into 662 features takes 25.9–30.3 s — between
  172x and 639x the encoder. Optimising the encoders, the part carrying the
  Cython, Numba and CuPy paths, attacks the smaller half of the bill.
- **Peak memory is the real limit.** Feature extraction peaks at 7.2 GiB on a
  4096-sample series, 56.3x the 128 MiB image it is handed, and is identical
  across all four encoders because it depends only on the image size. A 4096
  series is not a large input, and this is already past what a small machine has.
- **The documented complexity holds.** Feature-extraction memory scales as
  `N**2.00` for every encoder, and encode memory does too except for `mtf` at
  1.82, where per-bin overhead is still visible at the short lengths. Both match
  the `O(N^2) memory` recorded in the representation metadata.

Encode memory does separate the encoders: at 4096 `rp` peaks at 3x the image
size, `gaf` and `gadf` at 2x, and `mtf` at 1x.

The timing columns are best-of-three on a machine running other work, and they
moved by tens of percent between two runs of the same commit; the fitted time
exponents moved with them (1.44–2.43). Read the times as magnitudes. The memory
columns are byte-identical across runs.

Encode memory does separate the encoders: at 4096 `rp` peaks at 3x the image
size, `gaf` and `gadf` at 2x, and `mtf` at 1x.

## Scope

Per-series encoding and per-image feature extraction only. Two things in the
default method grid are deliberately absent:

- `baseline-rocket-ridge` is a transform fitted on a training batch, not a
  per-series encoder, so it does not fit this table's shape and would need its
  own batch-level sweep.
- The raw baselines are a z-normalisation, whose cost is linear in the series
  length and negligible against everything measured here.

The lengths follow the roadmap's requested set. Real UCR series in the committed
38-dataset run top out at 637 samples, so the 4096 column is a stress test of
the implementation rather than a description of the benchmark workload.
