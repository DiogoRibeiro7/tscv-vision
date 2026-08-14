# tscv-vision paper draft

Status: draft scaffold for a software paper. This is not a submitted
manuscript and it should not be cited as a peer-reviewed publication.

## Working Title

**tscv-vision: Validated image-style representations and feature pipelines for
univariate time-series classification**

## Abstract

`tscv-vision` is a NumPy-first Python package for constructing, validating and
evaluating structured representations of one-dimensional time series. It
implements image-style encoders such as Gramian Angular Fields, Markov
Transition Fields, recurrence plots, spectrograms and related project-defined
representations, together with classical image descriptors, sliding-window
processing, representation metadata and a leakage-safe benchmark harness.

The package's main contribution is not a new classifier or a claim that
time-series imaging is universally superior. Instead, it provides a disciplined
software surface for comparing representation choices: each encoder carries
machine-readable provenance and a validation level, tests record whether the
implementation is shape-tested, invariant-tested, formula-tested, compared to
an independent implementation, or benchmarked on real datasets, and benchmark
runs freeze raw row-level outputs plus an environment manifest.

In the current evidence run, nine fixed methods were evaluated on 38 univariate
UCR datasets with three classifier seeds. A ROCKET baseline is the strongest
method by a wide margin (mean accuracy 0.9005, average rank 1.14). The best
image-style pipeline is GADF image features (0.7706, average rank 4.43), which
does not improve on a raw-series logistic-regression control (0.8200, average
rank 3.95). Every pairwise difference between ROCKET and the other eight
methods is significant under a Holm-corrected Wilcoxon signed-rank test.

These results support a narrower claim than time-series imaging is usually
given: the package can reproduce a leakage-safe comparison, and on this subset
its default image-feature pipelines are clearly beaten by a cheap modern
baseline. We report this because the harness was built to be able to find it.

## Paper Claims

Supported claims:

- `tscv-vision` provides a unified Python API for deterministic time-series
  representations, image-feature extraction, sliding-window workflows and
  benchmark comparison.
- The provenance registry prevents silent overclaiming by separating canonical
  methods from project-defined variants and by recording the validation level
  actually backed by tests.
- The benchmark harness uses predefined train/test splits, writes one raw row
  per `(dataset, method, seed)`, records an environment manifest and computes
  Friedman, Nemenyi and Holm-corrected Wilcoxon comparisons.
- On the committed 38-dataset UCR subset, the ROCKET baseline has the best
  average rank among the default methods, beating every other method with a
  Holm-corrected Wilcoxon p below 1e-4. GADF features are the best-ranked
  image-feature method and do not improve on raw logistic regression.

Unsupported claims:

- The package does not establish that image-style representations are generally
  better than raw-series baselines. On this subset they are measurably worse
  than both a strong modern baseline and, for most of the grid, a raw-series
  control.
- The committed run is not a full UCR/UEA archive study.
- LEVEL 4 benchmark validation has not been assigned to individual encoders in
  the metadata registry.
- The software does not claim novelty for established methods such as GAF, MTF,
  recurrence plots, SAX, shapelets, persistence images or attention.

## Contribution Outline

1. **Representation surface.** Encoders convert a 1D series into square images,
   time-frequency images, symbolic matrices, graph matrices and topological
   summaries. Public functions validate shapes, numeric finiteness and
   `nan_policy` behavior.
2. **Feature extraction.** Classical descriptors summarize representation
   images, including intensity statistics, histograms, gradient/orientation
   features, local binary patterns, GLCM, Gabor, FFT, PSD and finite fractal
   descriptors.
3. **Metadata and validation.** `RepresentationInfo` records family,
   provenance, optional dependencies, complexity and validation level. Tests
   assert that documented validation claims name real backing tests.
4. **Leakage-safe evaluation.** `tscv_vision.evaluation` loads archive-style
   splits, evaluates fixed method definitions, appends rows incrementally and
   writes reproducibility manifests.
5. **Evidence artifact.** `results/ucr-thirty-eight/` freezes the first
   real-dataset run and provides the benchmark table used by this draft.

## Benchmark Design

Run directory: `results/ucr-thirty-eight/`

Reproduction command:

```bash
python -m tscv_vision.evaluation --download-ucr \
    --datasets-file results/ucr-thirty-eight/datasets.txt \
    --ucr-cache .benchmarks/ucr-cache \
    --seeds 0 1 2 \
    --out results/ucr-thirty-eight --no-resume --n-jobs 8
```

The command above reproduces the run in one pass. The committed artifact was
built in two: the eight non-ROCKET methods first, then the ROCKET rows added
with `--resume`. The source files that determine the numbers (`encoders.py`,
`features.py`, `evaluation.py`) are byte-identical across the two commits, so
every row reflects the same numeric code.

Design:

- Datasets: 38 fixed-length univariate UCR datasets listed in
  `results/ucr-thirty-eight/datasets.txt`.
- Methods: the nine entries in `evaluation.DEFAULT_METHODS`, including the
  ROCKET baseline (10 000 kernels, ridge with cross-validated alpha).
- Seeds: `0`, `1`, `2`.
- Rows: 1026 planned and completed rows, with zero failures.
- Data source:
  `https://timeseriesclassification.com/aeon-toolkit`, cached locally.
- Environment: Python 3.12.11, NumPy 2.5.2, scikit-learn 1.9.0, SciPy 1.16.3,
  pyts 0.13.0.

## Benchmark Results

From `results/ucr-thirty-eight/summary.md`:

| Method | Mean accuracy | Average rank |
| --- | ---: | ---: |
| baseline-rocket-ridge | 0.9005 | 1.14 |
| baseline-raw-logreg | 0.8200 | 3.95 |
| gadf-features | 0.7706 | 4.43 |
| baseline-1nn-euclidean | 0.7906 | 4.57 |
| gasf-features | 0.7612 | 4.83 |
| rp-features | 0.7611 | 4.83 |
| ablation-gaf-texture-only | 0.6924 | 6.95 |
| mtf-features | 0.6701 | 7.04 |
| ablation-gaf-intensity-only | 0.6230 | 7.26 |

Friedman chi-square = 151.932, p = 7.75e-29. Nemenyi critical difference at
alpha 0.05 = 1.949.

Interpretation:

- The fixed method grid supports ranking methods on this subset.
- ROCKET ranks first overall and its rank gap to the runner-up (2.81) exceeds
  the Nemenyi critical difference. Every one of its eight pairwise Wilcoxon
  comparisons survives Holm correction at p < 1e-4; against the next-best
  method it wins on 35 of 38 datasets.
- No image-feature pipeline improves on raw logistic regression. GADF is the
  best of them, but its rank gap to the raw control (0.48) is well inside the
  critical difference, so the honest reading is "not distinguishable", not
  "competitive".
- Texture-only and intensity-only GAF ablations trail the full image-feature
  pipelines, supporting the use of composed descriptors in the default grid.
- ROCKET's accuracy varies across seeds by 0.011 on average and by up to 0.078
  on one dataset, which is why it is reported over three seeds rather than one.

## Computational Cost

Run directory: `results/length-scaling/`, produced by
`benchmarks/scaling/run_length_scaling.py`. Encoder and feature-extraction cost
were measured separately for `gaf`, `gadf`, `mtf` and `rp` at series lengths
128, 256, 512, 1024 and 4096, best of three timed runs per cell, with peak
memory measured in a separate pass so the profiler's overhead does not enter the
reported times.

| Series length | Encode | Encode peak | Features | Feature peak |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.0001–0.0006 s | 0.3–0.4 MiB | 0.02–0.08 s | 7.1 MiB |
| 512 | 0.002–0.006 s | 2.1–6.0 MiB | 0.46–1.03 s | 112.6 MiB |
| 1024 | 0.004–0.024 s | 8.1–24.0 MiB | 1.79–2.47 s | 450.1 MiB |
| 4096 | 0.07–0.16 s | 128–384 MiB | 26.5–35.2 s | 7200.4 MiB |

Findings:

- Feature extraction, not encoding, dominates: at length 4096 it costs roughly
  200x the encoder. This is worth stating because the package's optimised paths
  (Cython, Numba, CuPy) target the encoders.
- Peak memory is the binding constraint. Feature extraction peaks at about 56x
  the size of the image it is handed, so a single 4096-sample series requires
  over 7 GiB.
- Measured memory exponents are 2.00 for every encoder, matching the `O(N^2)`
  complexity recorded in the representation metadata. This is a check on that
  metadata, not merely a timing report.

## Limitations

- The run covers 38 datasets, not the full UCR/UEA archive.
- The default method grid includes ROCKET but not MiniRocket, modern deep
  models or tuned pipelines. MiniRocket is absent because `pyts` 0.13 does not
  provide it, not because it was judged uninformative.
- ROCKET is not bit-reproducible across machines: its seed fixes the kernels,
  but the convolutions and the ridge solve go through BLAS, so results depend
  on thread count and platform. Its numbers should be read as a three-seed
  spread on the recorded hardware.
- Hyperparameters are fixed; the benchmark is a comparison of predefined
  methods, not a search for each method's best possible configuration.
- Runtime and memory are summarized for the four image-style encoders in
  `results/length-scaling/`, but not for ROCKET, which is a batch-fitted
  transform and needs its own sweep, and not across NumPy, Numba, Cython and
  GPU backends, which remains open.
- Some optional encoders depend on external packages that are not installed in
  every environment, so the paper should separate core claims from optional
  integration claims.

## Figures And Tables To Produce

- Table 1: package modules and responsibilities.
- Table 2: validation-level counts and examples.
- Table 3: benchmark method definitions.
- Table 4: 38-dataset average accuracy and rank table.
- Figure 1: representation pipeline diagram from series to image to feature
  vector to classifier.
- Figure 2: critical-difference diagram for the committed UCR subset.
- Figure 3: runtime and peak memory versus series length, on log-log axes with
  the fitted exponents annotated. Data in `results/length-scaling/results.csv`;
  the split between the encode and feature stages is the point of the figure.

## Reproducibility Checklist

- Source code: `https://github.com/DiogoRibeiro7/tscv-vision`
- Citation metadata: `CITATION.cff`
- Release metadata: `.zenodo.json`
- Raw results: `results/ucr-thirty-eight/results.csv`
- Manifest: `results/ucr-thirty-eight/manifest.json`
- Dataset selection: `results/ucr-thirty-eight/datasets.txt`
- Validation matrix: `docs/encoder_validation.md`
- Benchmark guide: `docs/benchmarks.md`

## Reference Notes

The final manuscript should cite the original methods rather than presenting
them as project inventions. At minimum, include references for Gramian Angular
Fields, Markov Transition Fields, recurrence plots, SAX/PAA, shapelets,
persistence images, ROCKET, Demsar-style classifier comparison and the UCR/UEA
archive.
