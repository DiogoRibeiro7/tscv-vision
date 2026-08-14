# UCR thirty-eight run

This directory freezes a 38-dataset univariate UCR benchmark subset. The raw
datasets are not vendored; they are downloaded one ZIP per dataset from
timeseriesclassification and cached under `.benchmarks/ucr-cache`.

Recreate it from the repository root:

```bash
python -m tscv_vision.evaluation --download-ucr \
    --datasets-file results/ucr-thirty-eight/datasets.txt \
    --ucr-cache .benchmarks/ucr-cache \
    --seeds 0 1 2 \
    --out results/ucr-thirty-eight --no-resume
```

Grid:

- 38 UCR datasets listed in `datasets.txt`
- 9 default methods from `evaluation.DEFAULT_METHODS`
- 3 seeds: `0`, `1`, `2`
- 1026 planned rows in `results.csv`

The `baseline-rocket-ridge` rows were added after the original eight methods,
by rerunning with `--resume` so only the missing rows were computed. The eight
original methods were produced at commit `6f0d5d9`, whose `encoders.py`,
`features.py` and `evaluation.py` are byte-identical to the commit recorded in
the current `manifest.json`, so every row in the file reflects the same numeric
code. ROCKET requires `pyts`; see `docs/benchmarks.md` for why the default grid
keeps it even where that package is absent.
