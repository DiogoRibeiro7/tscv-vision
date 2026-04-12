# Roadmap

> Last updated: 2026-04-12

This roadmap tracks what has shipped, what is next, and the long-term vision
for **tscv-vision**. Items are grouped by theme rather than strictly by
version so that priorities stay clear as scope shifts.

---

## Completed

Features that are implemented and available on `main`.

### Core encoders (v0.1.0 – present)

- [x] GAF / GADF (Gramian Angular Fields)
- [x] Recurrence Plot (euclidean / manhattan, binary threshold)
- [x] Spectrogram (STFT, Hann / rect windows)
- [x] CWT (Morlet / mexh / ricker wavelets, PyWavelets fallback)
- [x] Markov Transition Field (MTF)
- [x] DTW cost matrix
- [x] SAX similarity matrix
- [x] Persistence Image (topological approximation)
- [x] Gradient Difference Field (GDF)
- [x] Multi-scale Recurrence Plot (MSRP)
- [x] Multi-scale Convolutional encoder
- [x] Temporal Pattern Attention (TPA)
- [x] Visibility Graph adjacency matrix
- [x] Shapelet Transform distance maps
- [x] Matrix Profile
- [x] Random Projection Image
- [x] Ensemble (stack / mean / weighted)

### Feature extraction & pipeline

- [x] Intensity stats, histogram, gradient histogram, LBP
- [x] Sliding-window helpers (stride-trick views)
- [x] Multi-encoder fusion (concat, mean, median, weighted)
- [x] Temporal aggregation for sliding features
- [x] Custom encoder registry (`register_encoder` / `get_encoder`)
- [x] CLI (`tscv-features`) with batch, parallel, and dry-run modes

### Integrations

- [x] scikit-learn `SklearnFeatureTransformer`
- [x] PyTorch `TorchFeatureDataset`
- [x] TensorFlow generator dataset
- [x] ONNX tensor export (`to_onnx_tensor` / `save_onnx`)
- [x] Arrow / Parquet / HDF5 I/O
- [x] Streaming & windowed dataset API (`StreamingEncoder`, `WindowedDataset`)
- [x] Dask-based distributed map (`map_dask`)

### Acceleration

- [x] Cython extensions (GAF, recurrence, STFT)
- [x] Optional Numba JIT path for GAF
- [x] CuPy GPU encoder (GAF) with automatic fallback

### Quality & tooling

- [x] Pre-commit hooks (ruff, mypy --strict)
- [x] GitHub Actions CI (tests, linting, Codecov)
- [x] 43 test files with `gpu`, `slow`, `optional` markers

---

## v0.2.0 – Stability & correctness

**Theme:** Make the library robust enough for real-world data before adding
more features. Everything here is a prerequisite for a v1.0 claim.

### NaN / missing-data handling *(high priority)*

- [ ] Add a `nan_policy` parameter to `_validate_series`:
      `"raise"` (current default), `"omit"`, `"interpolate"`, `"fill"`
- [ ] Propagate `nan_policy` through all public encoder functions
- [ ] Add tests for each policy (constant NaN, leading/trailing, scattered)
- [ ] Document behaviour in the API reference and troubleshooting guide

### Version & packaging hygiene

- [ ] Align version between `pyproject.toml` (0.1.1) and `setup.py` (0.10.0)
      — use a single source of truth (`importlib.metadata` or `__version__`)
- [ ] Add `py.typed` marker for PEP 561 compliance
- [ ] Tag a proper release on GitHub with CHANGELOG entry

### Edge-case hardening

- [ ] Define and document minimum input length per encoder
- [ ] Handle zero-length arrays gracefully (return empty with correct shape)
- [ ] Validate multichannel inputs with mismatched lengths
- [ ] Add property-based tests (Hypothesis) for encoder shape invariants

### Platform & install notes

- [ ] Document Windows / WSL / macOS Cython build caveats
- [ ] Add a `--no-cython` install path for pure-Python fallback
- [ ] Test against Python 3.12 in CI matrix

---

## v0.3.0 – Documentation & usability

**Theme:** Lower the barrier to adoption. A library with 20 encoders but thin
examples is hard to evaluate.

### Encoder selection guide

- [ ] "When to use which encoder" decision table (signal type → encoder)
- [ ] Algorithm reference cards: one paragraph + key formula + citation per
      encoder (GAF → Zhiguang Wang 2015, RP → Eckmann 1987, etc.)

### Examples & tutorials

- [ ] End-to-end Jupyter notebook: ECG classification with GAF + sklearn
- [ ] End-to-end notebook: vibration anomaly detection with RP + PyTorch
- [ ] CLI cookbook: common one-liners for batch feature extraction
- [ ] Streaming tutorial: real-time encoding with `StreamingEncoder`

### API surface cleanup

- [ ] Audit thin / stub modules (`irregular.py` is 26 LOC) — either flesh out
      or fold into the module they extend
- [ ] Consolidate ONNX export: current `save_onnx` only serialises raw
      tensors; add a proper model graph export if feasible, or document the
      limitation clearly
- [ ] Add `__all__` consistency check in CI (ensure public API matches docs)

---

## v0.4.0 – Performance & scale

**Theme:** Make large-scale and real-time workloads first-class.

### Benchmarking

- [ ] Reproducible benchmark suite against UCR/UEA archive subsets
- [ ] Publish throughput & memory numbers for each encoder at common sizes
      (128, 256, 512, 1024, 4096)
- [ ] Compare against pyts, tslearn, and tsfresh on overlapping encoders
- [ ] Integrate `benchmark.py` results into CI as a regression gate

### GPU acceleration

- [ ] Extend CuPy encoders beyond GAF (recurrence plot, spectrogram, MTF)
- [ ] Add multi-GPU support for batch encoding
- [ ] Benchmark CPU vs GPU crossover point by series length

### Streaming & backpressure

- [ ] Add buffer-size limits and overflow policy to `StreamingEncoder`
- [ ] Quantify and publish latency guarantees (p50 / p99) for `safe_encode`
- [ ] Add async iterator interface for integration with asyncio event loops

### Algorithmic improvements

- [ ] Vectorise `visibility_graph` (currently O(N³) nested Python loops)
- [ ] Vectorise `matrix_profile` (currently O(N²) Python loop; consider
      STOMP or SCRIMP approach)
- [ ] Add chunked computation for large recurrence plots (memory O(N) instead
      of O(N²))

---

## v0.5.0 – Extensibility

**Theme:** Let users and downstream libraries plug in without forking.

### Plugin system

- [ ] Entry-point based encoder discovery (`tscv_vision.encoders` group)
- [ ] Entry-point based feature extractor discovery
- [ ] Plugin template repository / cookiecutter
- [ ] Validate plugin contracts at registration time (input/output shapes)

### Domain modules

- [ ] Stabilise `domains/` API — currently 8 domain modules with varying
      maturity; define a consistent interface per domain
- [ ] Add domain-specific encoder presets (e.g., `finance.default_pipeline()`)
- [ ] Allow domain modules to be installed as separate extras

---

## v1.0.0 – Production-ready release

**Theme:** Commitment to API stability, documentation completeness, and
community confidence.

### Release gates

- [ ] All v0.2–v0.5 items resolved or explicitly deferred with rationale
- [ ] Test coverage ≥ 90% on core (`encoders`, `features`, `sliding`)
- [ ] Every public function has a docstring with Parameters, Returns, Raises,
      and at least one Example
- [ ] API reference auto-generated and published (Sphinx or mkdocs)
- [ ] Semantic versioning enforced; CHANGELOG maintained per release
- [ ] Security review: no `eval`, no arbitrary file writes, dependencies pinned

### Community & adoption

- [ ] CONTRIBUTING.md with DCO or CLA, issue templates, PR template
- [ ] Publish to PyPI with classifiers and project URLs
- [ ] Blog post or short paper: "Computer-vision features for time series"
- [ ] Engage time-series community (TSDatasets, sktime, aeon interop)

---

## Future directions (post-v1.0)

Ideas that are worth exploring but not committed to a release.

- **Learnable encoders**: end-to-end differentiable image encoding (PyTorch)
- **Transformer-based features**: ViT patch embeddings as feature vectors
- **Multi-variate native encoders**: cross-channel recurrence, cross-GAF
- **Federated / privacy-preserving pipelines**: encode locally, aggregate
  features centrally
- **WebAssembly build**: run encoders in the browser for interactive demos
- **Pre-computed encoder galleries**: visual catalogue of encoder outputs on
  canonical datasets for quick reference

---

## Feedback & contributions

Open an issue or discussion on GitHub to propose changes to this roadmap.
Priorities are driven by real-world use cases — if you have one, share it.
