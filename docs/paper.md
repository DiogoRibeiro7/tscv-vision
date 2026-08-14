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

In the current evidence run, eight fixed methods were evaluated on 38
univariate UCR datasets with three classifier seeds. The strongest mean
accuracy in that fixed grid came from a raw-series logistic-regression baseline
(0.8200), while the best image-style method by average rank was GADF image
features (average rank 3.45). These results support a narrower claim: the
package can reproduce a leakage-safe comparison and, on this subset, its
default image-feature pipelines are competitive but do not dominate raw-series
baselines.

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
- On the committed 38-dataset UCR subset, the raw logistic-regression baseline
  has the best average rank among the default methods, and GADF features are
  the best-ranked image-feature method.

Unsupported claims:

- The package does not establish that image-style representations are generally
  better than raw-series baselines.
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

Design:

- Datasets: 38 fixed-length univariate UCR datasets listed in
  `results/ucr-thirty-eight/datasets.txt`.
- Methods: the eight entries in `evaluation.DEFAULT_METHODS`.
- Seeds: `0`, `1`, `2`.
- Rows: 912 planned and completed rows, with zero failures.
- Data source:
  `https://timeseriesclassification.com/aeon-toolkit`, cached locally.
- Environment: Python 3.12.11, NumPy 2.5.2, scikit-learn 1.9.0, SciPy 1.16.3.

## Benchmark Results

From `results/ucr-thirty-eight/summary.md`:

| Method | Mean accuracy | Average rank |
| --- | ---: | ---: |
| baseline-raw-logreg | 0.8200 | 3.01 |
| gadf-features | 0.7706 | 3.45 |
| baseline-1nn-euclidean | 0.7906 | 3.58 |
| rp-features | 0.7611 | 3.83 |
| gasf-features | 0.7612 | 3.84 |
| ablation-gaf-texture-only | 0.6924 | 5.96 |
| mtf-features | 0.6701 | 6.04 |
| ablation-gaf-intensity-only | 0.6230 | 6.29 |

Friedman chi-square = 82.045, p = 5.27e-15. Nemenyi critical difference at
alpha 0.05 = 1.703.

Interpretation:

- The fixed method grid supports ranking methods on this subset.
- Raw logistic regression ranks first overall.
- GADF features are the highest-ranked image-feature pipeline, but their
  average-rank difference from baseline raw logistic regression is below the
  Nemenyi critical difference.
- Texture-only and intensity-only GAF ablations trail the full image-feature
  pipelines, supporting the use of composed descriptors in the default grid.

## Limitations

- The run covers 38 datasets, not the full UCR/UEA archive.
- The default method grid does not include ROCKET, MiniRocket, modern deep
  models or tuned pipelines.
- Hyperparameters are fixed; the benchmark is a comparison of predefined
  methods, not a search for each method's best possible configuration.
- Runtime and memory are recorded per row but have not yet been summarized into
  paper-ready tables.
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
- Figure 3: runtime versus series length for image-style methods and raw
  baselines.

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
persistence images, Demsar-style classifier comparison and the UCR/UEA archive.
