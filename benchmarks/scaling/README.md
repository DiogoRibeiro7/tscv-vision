# Length-scaling benchmark script

Measures encoder and feature-extraction cost as a function of input series
length, and freezes the result as committed evidence rather than a development
diagnostic. Unlike `benchmarks/encoders/`, the output of this script *is*
intended to back documented claims, so it writes a manifest recording the
hardware and package versions that produced it.

Reproduce the committed run from the repository root:

```bash
python benchmarks/scaling/run_length_scaling.py --repeats 3
```

Smoke configuration for CI, whose output is not evidence:

```bash
python benchmarks/scaling/run_length_scaling.py --smoke --out results/scaling-smoke
```

The two stages are timed separately because they do not cost the same, and
timing and peak memory are measured in separate passes: `tracemalloc` hooks
every allocation, so a wall-clock number measured under it is not one anyone
reproduces without the profiler attached.

The `--lengths` sweep is quadratic in cost. Raising the largest entry is the
one change that dominates total runtime.
