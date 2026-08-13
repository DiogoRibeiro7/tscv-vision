# Encoder Benchmark Scripts

This directory contains lightweight scripts for measuring encoder runtime,
memory use, sparsity, and synthetic-signal concentration metrics. These runs
are development diagnostics, not publication evidence.

Run the smoke configuration from the repository root:

```bash
python benchmarks/encoders/run_encoder_suite.py --smoke --out results/encoder-smoke.json
```

Run a larger local scaling pass:

```bash
python benchmarks/encoders/run_encoder_suite.py --out results/encoder-local.json
```

Commit only benchmark outputs that should support a documented claim. Until a
full archive-scale run is committed, encoder metadata should remain below
`LEVEL 4`.
