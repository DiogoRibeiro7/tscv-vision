# Roadmap

> Last updated: 2026-08-10 (0.2.0)

This roadmap tracks what has shipped, what is next, and the long-term vision
for **tscv-vision**. Items are grouped by theme rather than strictly by
version so that priorities stay clear as scope shifts.

**Current stance: new work must arrive validated.** The 0.2.0 review found
that the API had outgrown its evidence. Rather than freeze the surface, the
rule is now that nothing lands at LEVEL 0: every new encoder ships with a
provenance entry in the metadata registry, tests against its defining formula
or an independent implementation, a benchmark, and documentation — see
`docs/encoder_validation.md` for where each one stands. The archive-scale
benchmark study remains the v0.3.0 gate for any empirical claim.

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

## v0.3.0 – Evidence

**Theme:** Produce the experimental results the current API surface would need
to be defensible. Nothing else ships until this does.

### Benchmark study

- [ ] Run the harness over 30+ UCR/UEA datasets and commit the frozen
      `results.csv`, `manifest.json` and `summary.md` under `results/`
- [ ] Add ROCKET / MiniRocket as a strong baseline (via `pyts`) to the default
      method set once it is part of a committed run
- [ ] Ablations: encoder alone, feature subsets, encoder + raw, ensemble,
      feature selection on/off
- [ ] Runtime and peak-memory tables as a function of series length
      (128 / 256 / 512 / 1024 / 4096)
- [ ] NumPy vs Numba vs Cython vs GPU under controlled hardware, with the
      hardware recorded in the manifest
- [ ] Robustness sweeps: additive noise, missingness under each `nan_policy`,
      irregular lengths

### Remaining validation gaps

Read `docs/encoder_validation.md` for the current per-encoder status.

- [ ] Raise `cwt` above LEVEL 0: the Morlet path is a bespoke FFT
      implementation with no numerical comparison against Torrence & Compo
      or PyWavelets
- [ ] Raise the LEVEL 1 encoders (`gdf`, `msrp`, `msc`, `randproj`,
      `shapelet`, `eph`) to LEVEL 2 with tests against their defining
      formula, or retire them
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

### API hygiene

- [ ] Remove the 0.2.0 deprecation aliases (`tpa`, `TSHAPExplainer`,
      `cross_causal_lag`, `bias_report`) in 0.3.0 as announced
- [ ] `__all__` consistency check in CI (public API matches the docs)
- [ ] `py.typed` marker for PEP 561 compliance
- [ ] Document minimum input length per encoder and enforce it uniformly
- [ ] Property-based tests (Hypothesis) for encoder shape invariants

---

## v0.4.0 – Documentation & usability

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

## v0.5.0 – Performance & scale

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

## v0.6.0 – Extensibility

- [ ] Entry-point based encoder and feature-extractor discovery
- [ ] Validate plugin contracts at registration time (input/output shapes)
- [ ] Plugin template repository
- [ ] Stabilise the `domains/` API behind one consistent interface
- [ ] Ship domain modules as separate extras

---

## v1.0.0 – Production-ready release

### Release gates

- [ ] All v0.3–v0.6 items resolved or explicitly deferred with rationale
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
