# Benchmarking

`tscv_vision.evaluation` runs a comparative classification study over UCR/UEA
archive datasets. It exists so that claims about the library's encoders can be
checked rather than asserted.

## What the harness guarantees

- **Predefined splits.** Datasets are loaded with the archive's own
  `_TRAIN` / `_TEST` partition. The test split is touched once, to score.
- **No leakage in the comparison.** Each method is a fixed
  `representation + classifier` pair — nothing is tuned on the data being
  scored. If you need to tune, do it with
  `AdaptivePipeline.nested_score` or on the training split alone.
- **Frozen raw output.** Every `(dataset, method, seed)` row is appended to
  `results.csv` as soon as it completes, including failures. `manifest.json`
  records the Python and OS versions, the version of every package that can
  change a number, and the git commit (flagged if the tree was dirty).
- **Honest statistics.** Comparison uses Demšar-style non-parametric checks: a
  Friedman test over the complete block of datasets, average ranks, the Nemenyi
  critical difference, and Holm-corrected pairwise Wilcoxon signed-rank tests.
  Datasets where any method failed are dropped, because the tests need a
  complete block design.

## Getting the data

The UCR/UEA archive is not redistributable, so it is not vendored here.
Download `UCRArchive_2018` from
<https://www.cs.ucr.edu/~eamonn/time_series_data_2018/> and unpack it. The
expected layout is the archive's own:

```text
UCRArchive_2018/
  Adiac/
    Adiac_TRAIN.tsv
    Adiac_TEST.tsv
  Beef/
    ...
```

Datasets containing `NaN` (the archive's variable-length datasets encode
missing values this way) are rejected rather than silently imputed — impute
them yourself so the strategy is an explicit, reported choice.

## Running a sweep

```bash
# Everything the archive contains (long).
python -m tscv_vision.evaluation --archive /data/UCRArchive_2018 --out results/ucr

# A named subset, several seeds.
python -m tscv_vision.evaluation \
    --archive /data/UCRArchive_2018 \
    --datasets Adiac Beef Coffee ECG200 GunPoint \
    --seeds 0 1 2 \
    --n-jobs 4 \
    --out results/ucr-subset

# A file with one dataset name per line ('#' comments allowed).
python -m tscv_vision.evaluation --archive /data/UCRArchive_2018 \
    --datasets-file datasets.txt --out results/ucr

# A named subset downloaded one ZIP per dataset from timeseriesclassification.
python -m tscv_vision.evaluation --download-ucr \
    --datasets-file results/ucr-thirty-eight/datasets.txt \
    --ucr-cache .benchmarks/ucr-cache \
    --seeds 0 1 2 \
    --out results/ucr-thirty-eight

# Smoke-test the harness with no archive (generated data; not evidence).
python -m tscv_vision.evaluation --synthetic --out /tmp/bench

# Recreate the committed synthetic pilot run.
python -m tscv_vision.evaluation --synthetic \
    --synthetic-datasets 5 --synthetic-length 32 --synthetic-n-per-class 6 \
    --seeds 0 1 2 --out results/pilot-synthetic --no-resume
```

Outputs land in `--out`:

| File | Contents |
| --- | --- |
| `results.csv` | One row per `(dataset, method, seed)`: accuracy, feature count, encode/fit/predict seconds, peak MiB, dataset shape, error text |
| `manifest.json` | Environment, package versions, git commit, methods, seeds |
| `summary.md` | Rank table, Friedman result, critical difference, pairwise tests |

`run_benchmark()` and the CLI resume by default when `results.csv` already
exists. Rows whose `(dataset, method, seed)` key is already present are reused,
missing rows are appended, and stale rows outside the requested grid are
dropped when the file is normalized at startup. Pass `--no-resume` to recompute
from scratch. `--n-jobs` parallelizes independent combinations across worker
processes; keep it below the number of physical cores if the classifier or BLAS
library also uses threads.

## Default methods

`evaluation.DEFAULT_METHODS` covers the baselines and an ablation:

| Name | Representation | Classifier |
| --- | --- | --- |
| `baseline-1nn-euclidean` | z-normalised raw series | 1-NN Euclidean |
| `baseline-raw-logreg` | z-normalised raw series | logistic regression |
| `gasf-features` | GASF image → image features | logistic regression |
| `gadf-features` | GADF image → image features | logistic regression |
| `mtf-features` | MTF image → image features | logistic regression |
| `rp-features` | recurrence plot → image features | logistic regression |
| `ablation-gaf-intensity-only` | GASF, intensity features only | logistic regression |
| `ablation-gaf-texture-only` | GASF, LBP + GLCM only | logistic regression |

1-NN Euclidean on the z-normalised series is the standard UCR reference point;
any claim about an encoder should be made relative to it.

Define your own by building `Method` objects:

```python
from tscv_vision.evaluation import Method, load_ucr_tsv, run_benchmark, compare_methods

methods = [
    Method("1nn", "raw", "knn1"),
    Method("rocket", "rocket", "ridge"),          # needs pyts
    Method("cwt-ridge", "cwt", "ridge", bins=32),
]
datasets = [load_ucr_tsv("/data/UCRArchive_2018", n) for n in ("Coffee", "GunPoint")]
results = run_benchmark(datasets, methods, seeds=(0, 1, 2), out_dir="results/mine")
print(compare_methods(results).average_ranks)
```

`representation` accepts `"raw"`, `"rocket"`, or any key in
`encoders.ENCODER_REGISTRY`. `classifier` accepts `"knn1"`, `"logreg"`,
`"ridge"` and `"rf"`.

## Reading the summary

`summary.md` ranks methods by average rank (lower is better). Two methods
differ significantly under the Nemenyi test when their average ranks differ by
more than the reported critical difference. The pairwise table reports
per-dataset win counts and Holm-corrected Wilcoxon p-values; prefer these to
comparisons of mean accuracy, which is dominated by whichever datasets happen
to be hardest.

A non-significant Friedman result means the data do not support ranking the
methods at all — report that rather than the ordering. The Nemenyi
critical-difference helper uses the tabulated Demšar values for alpha 0.05 and
0.10 with 2-20 methods.

## Runtime and memory

Every row carries `encode_seconds`, `fit_seconds`, `predict_seconds` and
`peak_mib`. Encoding dominates for the image representations: an `(N, N)`
encoder is quadratic in series length, so cost grows sharply with `length`,
which is recorded per dataset. Report these alongside accuracy — a
representation that wins by 1% at 50× the cost is a different claim from one
that wins outright.

## Encoder diagnostics

For local encoder profiling without a dataset archive, run:

```bash
python benchmarks/encoders/run_encoder_suite.py --smoke --out results/encoder-smoke.json
```

The script records time-frequency concentration metrics for the STFT, CWT and
synchrosqueezed CWT, plus horizontal-visibility graph scaling rows. These
diagnostics are useful for regressions and profiling, but they do not raise an
encoder to `LEVEL 4`; only committed dataset-scale benchmark results can do
that.

## Reproducing a published run

`manifest.json` pins everything needed. To reproduce:

1. check out the recorded `git_commit` (and confirm `git_dirty` is `false`);
2. install the recorded package versions;
3. rerun with the same `--datasets`, `--seeds`, `--n-jobs` and `--out`;
4. diff the new `results.csv` against the frozen one.

Classifier seeds are threaded through, so the deterministic methods reproduce
exactly. `rf` and `rocket` vary across BLAS thread counts and platforms; run
several seeds and report the spread.
