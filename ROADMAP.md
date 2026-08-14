# Roadmap

> Last updated: 2026-08-14 (0.4.0 released)

This roadmap tracks what has shipped, what is next, and the long-term vision
for **tscv-vision**. Items are grouped by theme rather than strictly by
version so that priorities stay clear as scope shifts.

**Current stance: new work must arrive validated.** The 0.2.0 review found
that the API had outgrown its evidence. Rather than freeze the surface, the
rule is that nothing *new* lands at LEVEL 0: every encoder added since ships
with a provenance entry in the metadata registry, tests against its defining
formula or an independent implementation, and documentation. No encoder remains
at LEVEL 0; inherited project-defined encoders that still lack formula or
reference checks are listed below as LEVEL 1 gaps rather than quietly
overclaimed. The committed 38-dataset UCR run is the first evidence-bearing
artifact; broader archive coverage is still what broad empirical claims need.

**What the evidence says so far is not flattering, and that is the point.**
With a ROCKET baseline in the default grid (v0.4.0), no image-feature pipeline
improves on a raw logistic-regression control, and ROCKET beats all of them
with a Holm-corrected Wilcoxon p below 1e-4. The committed cost run adds that
feature extraction, not encoding, is 172x to 639x the bill at length 4096 and
peaks at 7.2 GiB — so the optimised paths in this package accelerate the
cheaper half. Both results came out of the harness the package exists to
provide, which is the strongest argument available that the harness works.
Neither is a reason to stop; they are the baseline any future representation
work has to beat.

## Where validation stands

29 encoders carry provenance metadata, 17 of them reproducing a published
method under its own name. Generated per-encoder detail lives in
[docs/encoder_validation.md](docs/encoder_validation.md).

| Level | Count | Meaning |
| --- | ---: | --- |
| 3 — reference | 8 | checked against an independent implementation |
| 2 — synthetic | 13 | checked against the published formula or an analytic answer |
| 1 — invariant | 8 | mathematical invariants only |
| 0 — smoke | 0 | no encoder is shape-only |
| 4 — benchmark | 0 | no encoder has been promoted from the subset run yet |

The empty LEVEL 4 row is the honest headline: two real runs now exist for the
default method set, but the registry has not yet promoted individual encoders
to benchmark-validated status. Note what promotion would now record — `gaf`,
`gadf`, `mtf` and `rp` have been benchmarked and *lost*. LEVEL 4 means measured
on real data, not vindicated by it, and the promotions should say so.

## Next up

In rough priority order, as of the v0.4.0 release:

1. **Raise the eight LEVEL 1 encoders, or retire them.** The largest remaining
   honesty gap in the registry, and the one the current stance was written for.
2. **Backend comparison** (NumPy vs Numba vs Cython vs GPU). The cost run
   showed the encoders are the cheaper half, which makes this less urgent for
   throughput than it looked, but it is still an unmeasured claim.
3. **Robustness sweeps** — noise, missingness per `nan_policy`, irregular
   lengths. The last unstarted item in the benchmark study.
4. **Promote the four benchmarked encoders to LEVEL 4**, recording that they
   were measured and beaten.
5. **Resolve the overdue deprecation aliases** (see API hygiene below).

Two known tooling problems, neither of which is a roadmap feature but both of
which will bite a contributor:

- `mypy` cannot run against this project's config in a current environment:
  `python_version = "3.11"` in `pyproject.toml` against numpy's stubs, which
  use 3.12 `type` statements. It aborts inside the stubs, so it reports success
  on code it never checked. The `.pre-commit-config.yaml` mypy hook depends on
  it.
- `tests/test_performance.py::test_extract_batch_speed` asserts a wall-clock
  ratio (`t_opt <= t_base * 1.25`) and fails intermittently under load.

---

## Completed

### Core encoders (v0.1.0 – present)

- [x] GAF / GADF (Gramian Angular Fields) — equivalence-tested against `pyts`
- [x] Recurrence Plot (euclidean / manhattan, binary threshold)
- [x] Spectrogram (STFT, Hann / rect windows)
- [x] CWT (Morlet / mexh / ricker wavelets, PyWavelets fallback)
- [x] Markov Transition Field (MTF) — equivalence-tested against `pyts`
- [x] DTW cost matrix
- [x] SAX (standard Gaussian breakpoints since 0.2.0; quantile variant opt-in)
- [x] Persistence diagram + persistence image — equivalence-tested against
      `ripser` / `persim` (0.2.0; the pre-0.2.0 heuristic is now
      `extrema_persistence_histogram`)
- [x] Gramian Difference Field (GDF)
- [x] Multi-scale Recurrence Plot (MSRP)
- [x] Multi-scale Convolutional encoder
- [x] Window self-attention (named `tpa` before 0.2.0; it is not the Temporal
      Pattern Attention architecture)
- [x] Visibility Graph adjacency matrix
- [x] Shapelet Transform distance maps
- [x] Matrix Profile — equivalence-tested against `stumpy`
- [x] Random Projection Image
- [x] Ensemble (stack / mean / weighted)

### Encoders added after 0.2.0 (shipped in v0.3.0)

Each shipped as its own PR with provenance metadata, validation tests and
documentation.

- [x] Synchrosqueezed CWT (`sst`) — reassignment to the instantaneous
      frequency from the phase derivative; ridge tracks the analytic law
- [x] Multitaper spectrogram (`mtspec`) — Thomson's estimator over DPSS
      tapers; variance falls from 0.93 to 0.18 going from 1 to 7 tapers.
      Requires SciPy
- [x] Chirplet transform (`chirplet`) — chirped atoms, recovers sweep rates
      to within one grid step; the `c=0` slice provably equals an STFT
- [x] Wavelet scattering (`scat`) — thin validated layer over Kymatio; the
      coefficients are the backend's, verified verbatim
- [x] Horizontal visibility graph (`hvg`) — `O(N)` stack algorithm checked
      against the quadratic definition
- [x] Ordinal pattern transition field (`otf`) — **TSCV-Vision
      representation**; reproduces the published forbidden-pattern result for
      the logistic map
- [x] Delay-embedding density (`ded`) — **TSCV-Vision representation**;
      recovers the logistic map's analytic parabola
- [x] Cross recurrence plot — two series of possibly different lengths;
      recovers a known lag as a diagonal offset
- [x] Joint recurrence plot — per-channel thresholds, so rescaling one
      channel by 1000x leaves the result unchanged
- [x] Wavelet coherence — with the degenerate unsmoothed case documented and
      tested rather than hidden

### Multivariate and backend modules (v0.3.0)

- [x] `tscv_vision.multivariate` — encoders taking more than one series, with
      shared recurrence machinery, plus `MULTIVARIATE_METADATA` so living
      outside the univariate registry does not mean living outside the
      provenance system
- [x] `tscv_vision.scattering` — Kymatio-backed, with the backend recorded as
      an `optional_dependency` in the metadata
- [x] `RepresentationInfo.optional_dependency` — registry consumers can tell
      "not installed" apart from "broken", and a test enforces that anything
      raising `ImportError` declares what it needs

### Feature extraction & pipeline

- [x] Intensity stats, histogram, gradient histogram, LBP / LBP-RI /
      LBP-uniform (circular sampling, matches scikit-image since 0.2.0)
- [x] GLCM, Gabor, orientation, edge/contour/fractal, FFT/PSD/wavelet
- [x] `feature_layout` / `feature_vector_length` for discoverable dimensionality
- [x] Sliding-window helpers (stride-trick views)
- [x] Multi-encoder fusion (concat, mean, median, weighted)
- [x] Temporal aggregation for sliding features
- [x] Custom encoder registry (`register_encoder` / `get_encoder`)
- [x] CLI (`tscv-features`) with batch, parallel, and dry-run modes

### Representation API (v0.2.0)

- [x] `tscv_vision.representations`: one interface over deterministic,
      fitted and pretrained representations, with adapters for every
      registered encoder and optional scikit-learn wrappers
- [x] `RepresentationInfo` provenance metadata — family, reference,
      complexity, canonical-vs-project-defined, validation level — with
      construction-time checks that it cannot over-claim
- [x] Registry filtering by provenance and validation level
- [x] `docs/encoder_validation.md` generated from the metadata, kept
      current by a test

### Evaluation & statistics (v0.2.0)

- [x] `tscv_vision.stats`: Welch t-test, Wilcoxon signed-rank, Friedman,
      Nemenyi critical difference, Holm correction — all validated against
      `scipy.stats`
- [x] `tscv_vision.evaluation`: UCR/UEA harness with predefined splits, frozen
      raw outputs, environment manifest and multi-dataset comparison
- [x] Leakage-safe model selection: `FeatureSelector` inside CV folds,
      `AdaptivePipeline.nested_score`, `AutoTSCV.nested_score`

### Integrations

- [x] scikit-learn `SklearnFeatureTransformer` (genuinely subclasses the
      sklearn bases since 0.2.0)
- [x] PyTorch `TorchFeatureDataset`
- [x] TensorFlow generator dataset
- [x] ONNX tensor export (`to_onnx_tensor` / `save_onnx`)
- [x] Arrow / Parquet / HDF5 I/O
- [x] Streaming & windowed dataset API (`StreamingEncoder`, `WindowedDataset`)
- [x] Dask-based distributed map (`map_dask`)

### Acceleration

- [x] Cython extensions (GAF, recurrence, STFT)
- [x] Optional Numba JIT path for GAF
- [x] CuPy GPU encoder (GAF) with automatic fallback, batched sliding-window path

### Evidence artifacts (v0.3.1 – v0.4.0)

- [x] `results/ucr-thirty-eight/` — 38 UCR datasets x 9 default methods x 3
      seeds, 1026 rows, no failures. ROCKET ranks first (mean accuracy 0.9005,
      average rank 1.14); no image-feature pipeline beats raw logistic
      regression
- [x] `results/length-scaling/` — encoder and feature cost at five series
      lengths, hardware in the manifest. Feature extraction dominates and peak
      memory, not time, is the binding constraint
- [x] `benchmark_length_scaling` / `scaling_exponent` — fits `value ~ N**k` so
      the complexity strings in the representation metadata are measured rather
      than trusted. Memory exponents came back at 2.00, matching `O(N^2)`
- [x] `results/pilot-synthetic/` — harness smoke artifact, explicitly not
      evidence
- [x] Every committed run re-produced against a clean tree, so all three
      manifests record `git_dirty: false` as `results/README.md` requires

### Quality & tooling

- [x] `nan_policy` on every public encoder (`raise` / `omit` / `interpolate` /
      `forward_fill`) with per-policy tests
- [x] Pre-commit hooks (ruff, mypy --strict)
- [x] GitHub Actions CI with separate jobs for core, optional integrations,
      reference equivalence and the benchmark harness
- [x] Property, regression and reproducibility tests alongside unit tests
- [x] Definition tests for every scientific encoder
- [x] Documentation-sync test (signatures, feature dimensions, registry keys,
      version consistency)
- [x] Version aligned across `pyproject.toml`, `setup.py` and `__version__`

---

## v0.3.0 – v0.4.0 – Evidence

**Theme:** Produce the experimental results the current API surface would need
to be defensible. The surface has grown since 0.2.0, which raises the bar
rather than lowering it: 29 encoders with no benchmark is a larger gap than
19 encoders with no benchmark.

This theme spans more than one release: the UCR run and the paper scaffold
shipped in 0.3.0/0.3.1, and the ROCKET baseline and the length-scaling run in
0.4.0. The unchecked items below are what the theme still owes, which is why
the later milestones are numbered from v0.5.0 onwards.

### Benchmark study

- [x] Run the harness over 30+ UCR/UEA datasets and commit the frozen
      `results.csv`, `manifest.json` and `summary.md` under `results/`
- [x] Add ROCKET as a strong baseline (via `pyts`) to the default method set,
      committed as part of the 38-dataset run. MiniRocket is still open: `pyts`
      0.13 ships `ROCKET` only, and adding it means taking a dependency on
      `sktime` or `aeon`, which is a bigger decision than this item implied
- [ ] Ablations: encoder alone, feature subsets, encoder + raw, ensemble,
      feature selection on/off
- [x] Runtime and peak-memory tables as a function of series length
      (128 / 256 / 512 / 1024 / 4096), frozen under `results/length-scaling/`
      with the hardware in the manifest. Covers the four image-style encoders
      and both pipeline stages; ROCKET needs its own batch-level sweep and is
      not in it
- [ ] NumPy vs Numba vs Cython vs GPU under controlled hardware, with the
      hardware recorded in the manifest
- [ ] Robustness sweeps: additive noise, missingness under each `nan_policy`,
      irregular lengths
- [x] Draft a software-paper scaffold grounded in the committed evidence run,
      with explicit unsupported-claim boundaries

### Remaining validation gaps

Read `docs/encoder_validation.md` for the current per-encoder status.

- [ ] Raise the LEVEL 1 encoders (`cwt`, `gdf`, `msrp`, `msc`, `randproj`,
      `shapelet`, `eph`, `ensemble`) to LEVEL 2 with tests against their
      defining formula, compare them against an independent implementation, or
      retire them
- [ ] Numerical validation for the domain modules, or demote them to examples
- [ ] Decide the fate of thin/unvalidated subsystems (`irregular.py`,
      parts of `multimodal.py`, the neural adapters that have no upstream
      package installed in CI): validate, document as experimental, or remove

### Deferred on a dependency

- [ ] Joint time-frequency scattering (Andén, Lostanlen & Mallat, 2019).
      Kymatio exposes `TimeFrequencyScattering` only on its development
      branch; no released version has it. Add it as a separate encoder under
      its own name once the backend ships it, rather than approximating it —
      `tscv_vision.scattering` provides time scattering in the meantime.
- [ ] Track Kymatio's SciPy compatibility. 0.3.0 imports
      `scipy.special.sph_harm`, removed in SciPy 1.17, so the `scattering`
      extra pins `scipy<1.17`. Drop the pin when upstream fixes it.

---

## Learned and pretrained representations

**Theme:** Extend `tscv_vision.representations` beyond deterministic encoders.
The abstractions exist — `LearnedRepresentation`, `PretrainedBackbone`,
`LearnedFusion` — and are deliberately still abstract.

The leakage rules already encoded in those base classes apply throughout: `fit`
sees training data only, fusion weights are parameters, and a pretrained
checkpoint whose corpus overlaps the evaluation data is contaminated in a way
no cross-validation scheme can detect.

- [ ] Pretrained vision encoder over encoder images (OpenCLIP / ViT), with the
      checkpoint recorded in the metadata as provenance
- [ ] Time-series foundation-model adapter
- [ ] Multi-view fusion, learned rather than concatenated
- [ ] Adaptive representation router
- [ ] Learnable tokenisers (time-series and visual)
- [ ] Self-supervised pretraining (JEPA, cross-view contrastive)
- [ ] Representation analysis: agreement/complementarity, probing, retrieval,
      caching
- [ ] Shape dictionary, shape transition, wavelet token, phase-aware and
      spectral-shift encoders

That baseline now exists, and it is a demanding one. Anything in this track has
to be measured against `baseline-rocket-ridge` on the committed subset — mean
accuracy 0.9005, average rank 1.14 — not against 1-NN Euclidean. A learned
representation that beats the image pipelines but not ROCKET has not cleared
the bar, and the harness will say so.

### API hygiene

- [ ] **Overdue.** Remove the 0.2.0 deprecation aliases (`tpa`,
      `TSHAPExplainer`, `cross_causal_lag`, `bias_report`). These were
      announced for removal *in 0.3.0*; 0.3.0, 0.3.1 and 0.4.0 have all shipped
      with them still present. Either cut them in the next release or withdraw
      the announcement — the docs currently promise something untrue
- [ ] `__all__` consistency check in CI (public API matches the docs)
- [x] `py.typed` marker for PEP 561 compliance
- [ ] Document minimum input length per encoder and enforce it uniformly
- [ ] Property-based tests (Hypothesis) for encoder shape invariants

---

## v0.5.0 – Documentation & usability

**Theme:** Lower the barrier to adoption, grounded in the v0.3.0 evidence.

- [ ] "When to use which encoder" decision table, backed by benchmark results
      rather than intuition
- [ ] Algorithm reference cards: one paragraph, key formula and citation per
      encoder
- [ ] End-to-end notebook: classification with GAF + scikit-learn, using
      nested CV
- [ ] End-to-end notebook: anomaly detection with recurrence plots
- [ ] CLI cookbook and a streaming tutorial
- [ ] API reference published (Sphinx or mkdocs)
- [ ] Platform notes: Windows / WSL / macOS Cython build caveats
- [ ] Python 3.12 in the CI matrix

---

## v0.6.0 – Performance & scale

**Theme:** Make large-scale and real-time workloads first-class, measured
against the v0.3.0 baselines.

- [ ] Extend CuPy encoders beyond GAF (recurrence plot, spectrogram, MTF)
- [ ] Benchmark CPU vs GPU crossover point by series length
- [ ] Chunked recurrence plots (memory O(N) instead of O(N²))
- [ ] Faster matrix profile (STOMP or SCRIMP++ instead of the O(N²) pairwise
      distance matrix)
- [ ] Buffer-size limits and an overflow policy for `StreamingEncoder`
- [ ] Publish p50 / p99 latency for `safe_encode`
- [ ] Async iterator interface for asyncio integration
- [ ] Benchmark regression gate in CI

---

## v0.7.0 – Extensibility

- [ ] Entry-point based encoder and feature-extractor discovery
- [ ] Validate plugin contracts at registration time (input/output shapes)
- [ ] Plugin template repository
- [ ] Stabilise the `domains/` API behind one consistent interface
- [ ] Ship domain modules as separate extras

---

## v1.0.0 – Production-ready release

### Release gates

- [ ] All v0.3–v0.7 items resolved or explicitly deferred with rationale
- [ ] Test coverage ≥ 90% on core (`encoders`, `features`, `sliding`, `stats`)
- [ ] Every public function has Parameters, Returns, Raises and an Example
- [ ] Every named scientific method has an equivalence test or an explicit
      "new / approximate variant" label
- [ ] Semantic versioning enforced; CHANGELOG maintained per release
- [ ] Security review: no `eval`, no arbitrary file writes, dependencies pinned

### Community & adoption

- [ ] CONTRIBUTING.md with DCO or CLA, issue templates, PR template
- [ ] Publish to PyPI with classifiers and project URLs
- [ ] Software paper built on the v0.3.0 evidence, claiming the framework and
      its measured properties — not novelty for GAF, MTF, RP, SAX, shapelets,
      persistence images or attention, all of which predate this project
- [ ] Interop with the time-series ecosystem (sktime, aeon, pyts)

---

## Future directions (post-v1.0)

- **Learnable encoders**: end-to-end differentiable image encoding (PyTorch)
- **Multi-variate native encoders**: cross-channel recurrence, cross-GAF
- **WebAssembly build**: run encoders in the browser for interactive demos
- **Pre-computed encoder galleries**: visual catalogue of encoder outputs on
  canonical datasets

---

## Feedback & contributions

Open an issue or discussion on GitHub to propose changes to this roadmap.
Priorities are driven by real-world use cases — if you have one, share it.
