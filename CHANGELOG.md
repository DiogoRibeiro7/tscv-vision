# Changelog

## [Unreleased]

### Added

- **`encoders.synchrosqueezed_cwt`** (registry key `sst`) — synchrosqueezed
  continuous wavelet transform. Energy is reassigned along the frequency axis
  to the instantaneous frequency estimated from the phase derivative, computed
  exactly in the Fourier domain, so ridges are sharply localised rather than
  smeared across the wavelet's bandwidth. Analytic Morlet and bump wavelets,
  configurable frequency grid, optional log-magnitude and complex output.
  Validated against signals with closed-form instantaneous frequency: the
  ridge of a constant sinusoid and of a linear chirp sit within one frequency
  bin of the analytic law, and energy is at least 5x more concentrated than
  the plain `cwt` it is built on.
- **`encoders.chirplet_transform`** (registry key `chirplet`) — correlates
  the signal against Gaussian-windowed chirped atoms, resolving chirp rate as
  well as time and frequency. Computed by de-chirping each frame and taking
  one FFT per rate. Validated by recovering known sweep rates to within one
  grid step for rates from -60 to +60 Hz/s, by separating crossing chirps into
  their two rates, and by time reversal negating the recovered rate. The
  `chirp_rate=0` slice is asserted to equal a windowed Fourier transform
  exactly, which is what distinguishes the encoder from an STFT with a
  different window. The `(rates, frequencies, frames)` tensor is size-checked
  against `max_bytes` before allocation.
- **`tscv_vision.scattering`** (registry key `scat`) — wavelet scattering via
  Kymatio, added as a thin validated layer rather than a reimplementation.
  Contributes input validation, deterministic coefficient ordering, three
  documented image layouts and axis metadata; the transform itself is the
  backend's, and the tests assert the coefficients are returned verbatim and
  the image is exactly the metadata-described permutation of them. Validated
  against the property that motivates the second order: an amplitude-modulated
  carrier produces more than twice the order-2 energy of a pure tone.
  Requires the new `scattering` extra.

  Named `scattering_transform`, not `joint_time_frequency_scattering`: no
  released Kymatio exposes `TimeFrequencyScattering`, and approximating JTFS
  by hand would carry the name without the method. JTFS is deferred until the
  backend ships it.
- **`tscv_vision.multivariate`** — a new module for encoders that take more
  than one series, since the univariate validator in `encoders` is the wrong
  contract for them and their output may be rectangular. Provides the shared
  recurrence machinery (`delay_embed`, pairwise distances, threshold
  selection) used by every recurrence encoder in it.
- **`multivariate.cross_recurrence_plot`** — cross recurrence between two
  trajectories (Marwan & Kurths, 2002). The series need not have equal length.
  The automatic threshold is the recurrence-rate quantile of the observed
  distances, a stated rule rather than a constant, which makes the plot
  invariant to a common rescaling. Validated by recovering a known phase shift
  as the offset of the dominant diagonal, by hitting the target recurrence
  rate to within 0.01, and by agreeing exactly with `encoders.recurrence_plot`
  on `cross_recurrence_plot(x, x)`.
- **`multivariate.joint_recurrence_plot`** — joint recurrence across the
  channels of a `(n_samples, n_channels)` matrix (Romano et al., 2004),
  reusing the recurrence machinery introduced with the cross recurrence plot.
  Each channel is thresholded separately, so the result is unchanged by
  rescaling any single channel by a factor of 1000, which a shared threshold
  would not survive. Only `combination='and'` is the canonical definition;
  `product` and `mean` are labelled TSCV-Vision extensions. Validated by
  reduction to a plain recurrence plot on one channel, by equalling the
  product of its own per-channel plots, and by coupled logistic maps recurring
  jointly more than uncoupled ones.
- `RepresentationInfo` gained `optional_dependency`, and multivariate encoders
  get their own `MULTIVARIATE_METADATA` so that living outside the univariate
  registry does not mean living outside the provenance system.
- **`encoders.multitaper_spectrogram`** (registry key `mtspec`) — Thomson's
  multitaper spectral estimator, averaging periodograms over orthogonal DPSS
  tapers. Validated against the estimator's quantitative promise: the relative
  variance of the estimate on Gaussian noise falls from ~0.93 with one taper
  (the chi-square-with-2-degrees-of-freedom result) to ~0.18 with seven, and
  single- and multi-taper outputs match periodograms computed directly from
  `scipy.signal.windows.dpss`. Requires SciPy via the new `spectral` extra;
  substituting an easier window would quietly make it a different estimator
  under the same name.
- **`encoders.delay_embedding_density`** (registry key `ded`) — a
  TSCV-Vision representation: Takens' delay embedding, with a two-dimensional
  projection of the reconstructed state space rendered as an occupancy image,
  optionally Gaussian-smoothed. Explicitly not a recurrence plot, which is
  indexed by pairs of times rather than by state-space coordinates. Validated
  against known geometry: at delay 1 every occupied cell of the logistic map's
  embedding lies within three bin widths of its analytic parabola
  `y = 4x(1-x)`, and it occupies a quarter as much of the plane as noise.
  Chronological order is discarded, which the tests assert rather than assume.
- **`encoders.ordinal_transition_field`** (registry key `otf`) — a
  TSCV-Vision representation, labelled as such: it composes Bandt-Pompe
  ordinal patterns with an ordinal transition network, laid out as a field in
  the manner of the MTF. The ingredients are published; this composition is
  not, so it is not marked canonical. Patterns are labelled by exact Lehmer
  codes rather than float hashes, unobserved states get zero rows instead of an
  invented uniform distribution, and the factorial state space is capped at
  order 7. Validated against the published forbidden-pattern result: the
  logistic map at r=4 admits exactly five of the six order-3 patterns while
  i.i.d. noise admits all six.
- **`encoders.horizontal_visibility_graph`** (registry key `hvg`) — the
  horizontal visibility criterion, distinct from the natural visibility graph
  already provided by `visibility_graph`. Edges are found in `O(N)` with a
  monotonic stack rather than the quadratic scan the definition suggests, and
  the result is checked against that definition by brute force over hundreds of
  random series and against hand-computed examples. Being ordinal, it is
  invariant under any strictly increasing transformation of the values, which
  the tests use to demonstrate that it is genuinely a different graph from the
  NVG. Optional amplitude and distance weightings are labelled as TSCV-Vision
  extensions and must be requested explicitly.
- `benchmark.benchmark_time_frequency` — runtime, peak memory, sparsity and
  energy concentration for the spectrogram, CWT and synchrosqueezed CWT.

## [0.2.0] - 2026-08-10

Correctness and terminology release. Several routines were named after
published methods they did not implement, and several had defects that
silently produced wrong numbers. This release fixes the methods, constrains
the names, and adds the reference-equivalence and benchmark machinery that
should have accompanied them.

### Fixed — wrong results

- **`features.lbp` / `lbp_ri` / `lbp_uniform`**: `radius` changed only the
  padding, never the sampling geometry, so `radius=1` and `radius=2` produced
  identical codes. Neighbours are now sampled on a circle of the requested
  radius with bilinear interpolation (LBP<sub>8,R</sub>, Ojala et al. 2002) and
  match `skimage.feature.local_binary_pattern` exactly on the image interior.
- **`features.lbp_uniform`**: non-uniform patterns were mapped to code 255
  while the histogram only covered `[0, 59)`, so the documented shared
  "non-uniform" bin was silently discarded. They now land in bin 58 and the
  histogram accounts for every pixel.
- **`features._lbp_rotation_map`**: replaced with the conventional
  minimum-over-rotations canonical form (matches skimage's `ror`).
- **`analytics.group_significance`**: documented as a Welch t-test but took its
  p-value from the standard normal, making it valid only asymptotically. It now
  computes Welch–Satterthwaite degrees of freedom and evaluates the Student t
  distribution; verified against `scipy.stats.ttest_ind(equal_var=False)`.
- **`encoders.matrix_profile`**: `m` close to `len(x)` excluded every candidate
  match and returned `[nan]` (an infinite profile divided by infinity). Such
  inputs now raise with an explanation of the length requirement. Added
  `exclusion` and `normalize` parameters; verified against `stumpy.stump`.
- **`encoders.sax`**: accepted `segments > len(x)`, producing empty segments and
  "Mean of empty slice" warnings. Now rejected.
- **`automl.evolve_hyperparams`**: scored one generation but indexed the *next*
  population with `argmax` of the previous scores, so it could return an
  unevaluated individual unrelated to the reported best score. The best
  `(individual, score)` pair is now tracked as evaluations happen.
- **`analytics.cross_correlation_lag`**: the lag branches were swapped, so the
  returned sign was the opposite of the documented "x leads y" convention.
- **`ml_integration.SklearnFeatureTransformer`**: imported the sklearn base
  classes only under `TYPE_CHECKING`, so at runtime it always inherited empty
  stubs and never received `TransformerMixin.fit_transform`. It now subclasses
  the real bases when sklearn is installed, and composes with `Pipeline`,
  `clone` and `cross_val_score`. The stub fallback implements `get_params`,
  `set_params` and `fit_transform` so it stays usable without sklearn.
- **`neural.MambaEncoder`**: constructed `Mamba(d_model, n_layers=...)`, which
  is not the upstream block's signature, and fed it `(1, 1, N)` tensors —
  putting the series length in the model-dimension axis. Both encoders now use
  the upstream `(d_model, d_state, d_conv, expand)` signature, project scalar
  samples into the model dimension with a learned linear layer, consume
  `(batch, length, d_model)`, and mean-pool over time. `RetNetEncoder` probes
  the layer-count keyword instead of guessing.

### Changed — renamed to match what the code does

Old names keep working until 0.3.0 and emit `DeprecationWarning`.

- `encoders.persistence_image` → **`encoders.extrema_persistence_histogram`**.
  The old function paired consecutive extrema and histogrammed them; it never
  computed persistent homology. `persistence_image` now implements the real
  construction (Adams et al., 2017) on top of the new
  `encoders.persistence_diagram`, an exact 0-dimensional sublevel-set diagram.
  Verified against `ripser` and `persim`.
- `encoders.tpa` → **`encoders.window_attention`**. The implementation is plain
  scaled dot-product self-attention over sliding windows, not the Temporal
  Pattern Attention architecture; the docstring also misattributed TPA to
  "Li et al." rather than Shih, Sun & Lee (2019). The registry key `tpa` still
  resolves to `window_attention` without warning.
- `analytics.TSHAPExplainer` → **`analytics.OcclusionExplainer`**. It performs
  window/frequency occlusion; no Shapley axiom holds.
- `analytics.cross_causal_lag` → **`analytics.cross_correlation_lag`**. Maximum
  cross-correlation does not establish causality.
- `research.bias_report` → **`research.group_mean_disparity`**, documented as a
  screening statistic rather than a fairness analysis.
- `research.add_dp_noise` now **requires** a `sensitivity` keyword and uses
  scale `sensitivity / epsilon`. Previously it used `1 / epsilon`
  unconditionally, which does not establish ε-differential privacy for
  arbitrary features. `research.add_laplace_noise(features, scale)` exposes the
  uncalibrated mechanism with no privacy claim attached.
- **`encoders.sax`/`sax_symbols`** now default to standard SAX: z-normalise,
  PAA, then equiprobable Gaussian breakpoints (Lin et al., 2007). The previous
  data-adaptive behaviour is available as `breakpoints="quantile"` and is
  documented as non-standard.

### Fixed — evaluation methodology

- **`pipeline.AdaptivePipeline.optimize`** performed supervised feature
  selection on the full `X, y` and then scored with `cross_val_score`, leaking
  every validation fold into the selection step. Selection now happens inside
  each training fold via the new `pipeline.FeatureSelector` transformer.
- Added **`AdaptivePipeline.nested_score`** and **`AutoTSCV.nested_score`**:
  the entire selection procedure is repeated inside each outer fold, giving an
  unbiased estimate. `optimize` documents that its best-over-configurations
  score is not one.
- **`AutoTSCV.validate`** now warns when called on the data `fit` saw.

### Added

- **`tscv_vision.representations`** — one interface over every way a series is
  turned into model input, and a registry queryable by scientific provenance:

  - `Representation` / `FittedRepresentation` / `PretrainedRepresentation` keep
    the three leakage profiles apart. A fitted representation refuses to
    transform before `fit`; `as_sklearn()` wraps any of them so the fitting
    happens inside a `Pipeline` fold. None of them inherit from scikit-learn,
    which stays optional.
  - `RepresentationInfo` records family, reference, complexity, behavioural
    flags, and a 0–4 `ValidationLevel` naming the tests that back it.
    Construction rejects `canonical_method=True` without a reference, or a
    level above `SMOKE` without `validated_by` — so the metadata cannot claim
    more than the test suite delivers.
  - `list_representations(...)` / `list_encoders(...)` filter on that metadata,
    e.g. `list_representations(canonical_method=True, min_validation_level=3)`
    returns the five encoders checked against an independent implementation.
  - `DeterministicRepresentation` adapts every registered encoder, with named
    classes for the common ones and an `image_size` option that PAA-resamples
    the series so square-image encoders give a fixed shape for any input
    length. Shorter-than-requested series are rejected rather than
    interpolated.
  - `ConcatFusion` combines views deterministically via `fusion.fuse`;
    `LearnedFusion` is abstract because fusion weights are parameters.
  - `docs/encoder_validation.md` is generated from the metadata by
    `scripts/generate_encoder_validation.py`, with a test that fails when it
    goes stale. Every encoder now has a recorded provenance and validation
    level; `cwt` is honestly recorded at LEVEL 0, and no encoder claims
    LEVEL 4 because no benchmark run is committed.

- **`encoders.BUILTIN_ENCODERS`** — the registry snapshot taken before any user
  registration, so built-in encoders (which must carry metadata) can be told
  apart from ones added at runtime.

- `fusion.fuse` now accepts 1D feature vectors as well as `(N, D)` matrices;
  concatenation moved to the last axis, which is unchanged for 2D input.

- **`tscv_vision.stats`** — SciPy-free `welch_ttest`, `wilcoxon_signed_rank`
  (exact and normal branches), `friedman_test`, `average_ranks`,
  `nemenyi_critical_difference`, `holm_bonferroni`, and the underlying
  `betainc`, `gammainc_upper`, `student_t_sf`, `chi2_sf`, `normal_sf`. All
  validated against `scipy.stats`.
- **`tscv_vision.evaluation`** — leakage-safe benchmark harness over UCR/UEA
  archive datasets: predefined train/test splits, raw/encoder/ROCKET
  representations, 1-NN Euclidean and linear baselines, ablations, wall-clock
  and peak-memory measurement, frozen `results.csv` + environment `manifest.json`,
  and Demšar-style multi-dataset comparison. Run with
  `python -m tscv_vision.evaluation --archive <path> --out <dir>`.
- **`features.feature_layout` / `feature_vector_length`** — query the feature
  vector's composition instead of hard-coding a dimensionality that varies with
  `bins` and installed extras.
- `encoders.persistence_diagram`, `encoders.sax_symbols`,
  `ml_integration.SklearnFeatureTransformer.get_feature_names_out`.
- `tests/test_encoder_definitions.py` — every scientific encoder checked
  against its published formula, in the default suite.
- `tests/test_reference_equivalence.py` — equivalence with scikit-image, SciPy,
  pyts, ripser, persim and stumpy.
- `tests/test_docs_sync.py` — fails the build when documented signatures,
  feature dimensions, registry keys or versions drift from the code.
- CI jobs for optional integrations, reference equivalence and a benchmark
  smoke run. The default `pytest` invocation excludes `optional`, so these
  previously ran nowhere.

### Documentation

- `docs/api.md` rewritten against the actual signatures, with the naming policy
  and the real feature-dimension table (646 / 662 / 694 for `bins` 8 / 16 / 32
  with core dependencies only, not the 310 previously advertised).
- `docs/benchmarks.md` added: how to obtain the archive, run the sweep and read
  the statistics.
- `ROADMAP.md` corrected — items already shipped were still listed as TODO.
- Version aligned at 0.2.0 across `pyproject.toml`, `setup.py` and
  `tscv_vision.__version__`.
- Extras now cover every optional import. `numba`, `pyarrow`, `h5py`, `dask`,
  `redis`, `kafka-python`, `pika`, `onnx`, `torchvision` and `pyts` were used
  by the code but declared nowhere, so the only way to discover them was an
  `ImportError`. New extras: `ml`, `speed`, `io`, `streaming`, `distributed`,
  `onnx`; `research` was an empty extra and now installs what the benchmark
  harness needs. `pyproject.toml` and `setup.py` are checked against each other.

## [0.1.1] - 2026-08-10

- Document optional dependencies and provide sine-wave sample data
- Configure Cython build and clean up experimental modules

## [0.1.0] - 2025-08-13

- Initial release with GAF, recurrence plot and spectrogram encoders
- Feature extractors for intensity stats, histogram, gradient histogram and LBP
- Sliding-window pipeline and CLI
