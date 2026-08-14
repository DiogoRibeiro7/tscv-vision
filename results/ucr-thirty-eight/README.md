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
- 8 default methods from `evaluation.DEFAULT_METHODS`
- 3 seeds: `0`, `1`, `2`
- 912 planned rows in `results.csv`
