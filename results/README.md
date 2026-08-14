# Frozen benchmark results

This directory holds the raw outputs of benchmark runs so that published
numbers can be traced back to the exact code and environment that produced
them. Nothing here is generated at import or test time — a run is a deliberate,
committed act.

## Layout

One directory per run:

```text
results/
  ucr-2026-08/
    results.csv      # one row per (dataset, method, seed) -- never edited
    manifest.json    # python/OS versions, package versions, git commit
    summary.md       # ranks, Friedman, Nemenyi CD, pairwise Wilcoxon
```

## Producing a run

```bash
python -m tscv_vision.evaluation \
    --archive /path/to/UCRArchive_2018 \
    --datasets-file results/<run>/datasets.txt \
    --seeds 0 1 2 \
    --out results/<run>
```

Commit `results.csv`, `manifest.json`, `summary.md` and the `datasets.txt` that
selected the datasets. Do not hand-edit `results.csv`: if a run is wrong, redo
it and commit the new one, keeping the old directory or deleting it in a commit
that says why.

`manifest.json` records `git_dirty`. A run made from a dirty tree is not
reproducible and should not back a published claim.

## Status

`pilot-synthetic/` is committed as a harness smoke artifact: 5 generated
datasets, all default methods, and 3 seeds. It proves the machinery writes the
expected files, but it is not benchmark evidence.

`ucr-thirty-eight/` is the first committed real-dataset run: 38 univariate UCR
datasets, all default methods, and 3 seeds. It is enough to ground claims about
that fixed method set on that fixed dataset subset, but it is not a full UCR/UEA
archive study and does not promote every encoder in the registry to benchmark
validation.

`length-scaling/` is a cost run, not an accuracy run: encoder and
feature-extraction runtime and peak memory for the four image-style encoders at
five series lengths. It backs the performance claims in
[docs/performance.md](../docs/performance.md) and the paper's runtime figure,
and is produced by `benchmarks/scaling/run_length_scaling.py` rather than by the
evaluation harness.

See [docs/benchmarks.md](../docs/benchmarks.md) for how to obtain the archive
and interpret the statistics.
