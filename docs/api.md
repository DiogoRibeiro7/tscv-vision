# API Documentation

Public API of **tscv-vision**. All functions accept and return NumPy `ndarray`
objects and raise `ValueError` on invalid inputs.

Signatures below are kept in sync with the code; `tests/test_docs_sync.py`
fails the build if they drift.

## Naming policy

A function is named after a published method **only** when it implements that
method and has an equivalence test against a reference implementation
(`tests/test_reference_equivalence.py`) or against its published formula
(`tests/test_encoder_definitions.py`). Everything else is named descriptively
and says so in its docstring. Several names changed in 0.2.0 for this reason —
see the [changelog](../CHANGELOG.md#020). Old names still work for one release
and emit `DeprecationWarning`.

## Scope policy

The validated core and the experimental integration surface are tracked in
`tscv_vision.VALIDATED_CORE_MODULES` and `tscv_vision.CONTRIB_MODULES`. See
[scope.md](scope.md). Contrib modules remain importable for compatibility but
are candidates for extraction from the benchmark-critical package surface.

| Old name | New name | Why |
| --- | --- | --- |
| `encoders.persistence_image` (histogram) | `encoders.extrema_persistence_histogram` | Did not compute persistent homology. `persistence_image` now does. |
| `encoders.tpa` | `encoders.window_attention` | Plain self-attention, not Shih, Sun & Lee's TPA architecture. |
| `analytics.TSHAPExplainer` | `analytics.OcclusionExplainer` | Occlusion, not Shapley attribution. |
| `analytics.cross_causal_lag` | `analytics.cross_correlation_lag` | Cross-correlation is not causality. |
| `research.bias_report` | `research.group_mean_disparity` | Group-mean gap, not a fairness audit. |
| `research.add_dp_noise(x, eps)` | `research.add_dp_noise(x, eps, sensitivity=...)` | ε-DP requires sensitivity calibration. |

## Encoders

All encoders accept `nan_policy` (`"raise"` | `"omit"` | `"interpolate"` |
`"forward_fill"`).

### `gaf(x, method='summation', *, nan_policy='raise', use_numba=False, use_cython=False, use_gpu=False, gpu_device=None, gpu_mem_limit=None) -> Array`
Gramian Angular Field of shape `(N, N)`. `method` is `'summation'` (GASF) or
`'difference'` (GADF). Verified against `pyts.image.GramianAngularField`.

### `recurrence_plot(x, metric='euclidean', eps=None, *, nan_policy='raise', use_numba=False, use_cython=False) -> Array`
Recurrence matrix `(N, N)`. `eps` gives a binary plot; `None` returns
`1 - normalised distance`.

### `spectrogram(x, win=64, hop=None, window='hann', *, nan_policy='raise', use_numba=False, use_cython=False, use_gpu=False, gpu_device=None) -> Array`
STFT magnitude, normalised to `[0, 1]`. `hop` defaults to `win // 4`, `win >= 8`.
Returns `(win // 2 + 1, n_frames)`; the signal is zero-padded so every sample is
covered.

### `chirplet_transform(x, *, fs=1.0, frequencies=None, chirp_rates=None, window_size=64, hop_length=None, aggregate='max', log_scale=True, max_bytes=536870912, nan_policy='raise') -> Array`
Chirplet transform (Mann & Haykin, 1995): correlates against Gaussian-windowed
*chirped* atoms carrying a temporal centre, centre frequency, chirp rate and
complex phase. An STFT is the special case `chirp_rate = 0`, and a test asserts
exactly that agreement — sweeping the rate is what makes it a chirplet
transform. `aggregate='none'` returns the full
`(chirp_rates, frequencies, frames)` tensor, whose size is checked against
`max_bytes` before allocation.

### `multitaper_spectrogram(x, *, fs=1.0, window_size=128, hop_length=None, time_bandwidth=3.5, n_tapers=None, n_fft=None, scaling='log_power', dynamic_range=80.0, nan_policy='raise') -> Array`
Thomson's multitaper spectral estimate: average the periodograms taken with K
orthogonal DPSS (Slepian) tapers, reducing variance by roughly `1/K` at the
cost of widening resolution to `2NW/window_size`. A lower-variance alternative
to `spectrogram`. **Requires SciPy** (`pip install 'tscv-vision[spectral]'`) —
the DPSS tapers are the method, so no easier window is substituted.

### `cwt(x, scales, wavelet='morlet', *, nan_policy='raise') -> Array`
Continuous wavelet transform magnitude, `(len(scales), N)`. Non-Morlet wavelets
require PyWavelets.

### `synchrosqueezed_cwt(x, *, fs=1.0, scales=None, wavelet='morlet', frequencies=64, voices=32, threshold=None, magnitude=True, log_scale=False, nan_policy='raise') -> Array`
Synchrosqueezed CWT: a `(frequencies, N)` time-frequency image in which
coefficient energy is reassigned to the instantaneous frequency estimated from
the phase derivative, giving far sharper ridges than `cwt` or `spectrogram`.
Not a renamed CWT — the reassignment step is what makes it. `magnitude=False`
returns the complex transform, which keeps phase but cannot be fed to the image
feature extractors. Validated against analytically known instantaneous
frequencies (constant sinusoid, linear chirp, two-component signal).

### `mtf(x, bins=8, weighted=False, *, nan_policy='raise') -> Array`
Markov Transition Field `(N, N)` over `bins` quantile states. Verified against
`pyts.image.MarkovTransitionField`.

### `ordinal_transition_field(x, *, order=3, delay=1, image_size=None, tie_policy='stable', mode='field', seed=0, nan_policy='raise') -> Array`
**TSCV-Vision representation**, not a published image algorithm. Embeds the
series, replaces each window by its ordinal permutation (Bandt & Pompe, 2002)
labelled by an exact Lehmer code, estimates the pattern-to-pattern transition
matrix, and lays it out as a field in the manner of the MTF. Uses only the
*ordering* of values, so it is invariant under any strictly increasing
transformation and discards all amplitude information. The state space is
`order!`, hence the cap at `MAX_ORDINAL_ORDER = 7`. Unobserved states get
all-zero rows rather than an invented uniform distribution.

### `persistence_diagram(x, *, include_infinite=False, nan_policy='raise') -> Array`
Exact 0-dimensional sublevel-set persistence diagram, `(n_pairs, 2)` of
`(birth, death)`. Verified against Ripser.

### `persistence_image(x, bins=32, *, sigma=None, weight='persistence', birth_range=None, pers_range=None, nan_policy='raise') -> Array`
Persistence image (Adams et al., 2017) of shape `(bins, bins)`, indexed
`[persistence, birth]`. Pixels hold the exact integral of the weighted Gaussian
surface. Verified against `persim.PersistenceImager`. Ranges default to the
diagram's own extent, which makes images **series-relative** — pass explicit
`birth_range`/`pers_range` when comparing across series.

### `extrema_persistence_histogram(x, bins=32, *, nan_policy='raise') -> Array`
Histogram of consecutive-extrema `(birth, persistence)` pairs, `(bins, bins)`.
A heuristic descriptor, **not** persistent homology.

### `gdf(x, *, nan_policy='raise') -> Array`
Gramian Difference Field `(N, N)` in `[-1, 1]`.

### `delay_embedding_density(x, *, delay=1, dimension=2, bins=64, projection=(0, 1), density='histogram', sigma=1.0, normalize=True, nan_policy='raise') -> Array`
**TSCV-Vision representation**. Builds Takens' delay embedding and histograms a
two-dimensional projection of the reconstructed state space, optionally
Gaussian-smoothed. **Not a recurrence plot** — a recurrence plot is indexed by
pairs of times and records which states are close; this image is indexed by
state-space coordinates and records how often the trajectory visits each
region. Chronological order is discarded entirely, so direction of travel is
invisible. No automatic delay selection: choosing it from the data deserves its
own documented method rather than a silent default.

### `multi_scale_rp(x, scales, *, nan_policy='raise') -> Array`
Recurrence plots at several downsampling scales, `(len(scales), N, N)`.

### `dtw_matrix(x, *, nan_policy='raise') -> Array`
Self-DTW accumulated-cost matrix `(N, N)`, normalised and inverted.

### `sax_symbols(x, segments=8, alphabet=8, *, breakpoints='gaussian', nan_policy='raise') -> NDArray[np.int64]`
SAX word (Lin et al., 2007): z-normalise, PAA to `segments` means, map through
equiprobable Gaussian breakpoints. `breakpoints='quantile'` selects the
non-standard data-adaptive variant used before 0.2.0. `segments` must be
`<= len(x)`.

### `sax(x, segments=8, alphabet=8, *, breakpoints='gaussian', nan_policy='raise') -> Array`
Symbol-equality matrix `(segments, segments)` built from `sax_symbols`.

### `multi_scale_conv(x, kernels=(3, 5, 7), *, nan_policy='raise') -> Array`
Moving-average responses `(len(kernels), N)` scaled to `[0, 1]`.

### `window_attention(x, window=8, *, nan_policy='raise') -> Array`
Row-stochastic scaled dot-product self-attention between sliding windows,
`(N - window + 1, N - window + 1)`. Parameter-free; not a learned model.

### `visibility_graph(x, *, nan_policy='raise') -> Array`
Natural visibility graph adjacency `(N, N)` (Lacasa et al., 2008).

### `horizontal_visibility_graph(x, *, weighted=False, weight='binary', nan_policy='raise') -> Array`
Horizontal visibility graph adjacency `(N, N)` (Luque et al., 2009): `i` and
`j` are connected when every intermediate `x[k] < min(x[i], x[j])`. Distinct
from `visibility_graph` — the criterion is ordinal, so the HVG is a subgraph of
the natural visibility graph and is invariant under any strictly increasing
transformation of the values. Edges are found in `O(N)` with a monotonic stack.
The `amplitude` and `distance` weightings are TSCV-Vision extensions and
require `weighted=True`, which is enforced rather than silently ignored.

### `shapelet_transform(x, k=3, length=None, seed=None, *, nan_policy='raise') -> Array`
Distance maps to `k` randomly sampled subsequences, `(k, N - length + 1)`.

### `matrix_profile(x, m, *, exclusion=None, normalize=True, nan_policy='raise') -> Array`
Z-normalised matrix profile `(N - m + 1,)`. `exclusion` defaults to `m // 2`.
Requires `N - m + 1 >= exclusion + 2`; shorter inputs raise instead of
returning `nan`. Verified against `stumpy.stump`.

### `random_projection_image(x, size=32, seed=0, *, nan_policy='raise') -> Array`
`(size, size)` image from a seeded Gaussian projection.

### `ensemble(x, names=None, *, nan_policy='raise', weights=None, aggregate='stack') -> Array`
Stack or average several encoders that produce the same shape.

## Scattering

`tscv_vision.scattering` wraps Kymatio. **Requires** it
(`pip install 'tscv-vision[scattering]'`) — the cascade is the method, and a
short NumPy approximation would not be it.

### `scattering_transform(x, *, J=6, Q=8, format='image', log_scale=True, nan_policy='raise') -> Array`
Wavelet scattering coefficients (Mallat 2012; Andén & Mallat 2014). Three
layouts, all documented: `tensor` is the backend output verbatim; `image` is
`(n_coefficients, n_times)` with rows sorted by `(order, -xi1, -xi2)` and
described row-for-row by `scattering_meta`; `modulation` is the order-2
`(spectral band × modulation rate)` matrix averaged over time.

**This is time scattering, not joint time-frequency scattering.** No released
Kymatio exposes `TimeFrequencyScattering`, and approximating JTFS by hand
would carry the name without the method.

### `scattering_meta(n_samples, *, J=6, Q=8) -> dict[str, Any]`
Row labels for the `image` layout: `order`, `xi` and `j` per coefficient, in
image order, so axes can be labelled rather than guessed.

## Multivariate encoders

`tscv_vision.multivariate` holds encoders that take more than one series. They
are not in the univariate `ENCODER_REGISTRY`, whose contract is
`func(x) -> image`, but they do carry provenance metadata — see
`MULTIVARIATE_METADATA` and the validation matrix.

### `cross_recurrence_plot(x, y, *, dimension=1, delay=1, epsilon=None, recurrence_rate=0.1, metric='euclidean', binary=True, nan_policy='raise') -> Array`
Cross recurrence plot between two trajectories (Marwan & Kurths, 2002): when
does `x` visit a state of `y`. The two series **need not have equal length**,
so the output may be rectangular `(N_x, N_y)`. With `epsilon=None` the
threshold is the `recurrence_rate` quantile of the observed distances — a
stated rule, so the plot has approximately that fill by construction and is
invariant to a common rescaling. `binary=False` returns
`1 - distance / max(distance)`, matching `recurrence_plot`'s convention, and
the two agree exactly for `cross_recurrence_plot(x, x)`.

### `joint_recurrence_plot(X, *, dimension=1, delay=1, epsilon=None, recurrence_rate=0.1, metric='euclidean', combination='and', nan_policy='raise') -> Array`
Joint recurrence of a `(n_samples, n_channels)` matrix (Romano et al., 2004):
each channel gets its own recurrence plot with its **own** threshold, and the
canonical `combination='and'` keeps only where every channel recurs at once.
Per-channel thresholding is what makes it meaningful across channels with
different units — the result is unchanged by rescaling any single channel,
which a shared `epsilon` would not be. `product` and `mean` are TSCV-Vision
extensions combining continuous similarities rather than indicators.

### `wavelet_coherence(x, y, *, fs=1.0, scales=None, voices=12, wavelet='morlet', smoothing=True, return_phase=False, nan_policy='raise') -> Array | WaveletCoherenceResult`
Squared wavelet coherence (Torrence & Compo 1998; Grinsted et al. 2004):
localised, frequency-resolved coupling between two series of equal length,
bounded in `[0, 1]`.

**`smoothing=False` returns an all-ones image.** Without the time and scale
smoothing the ratio collapses to 1 at every point for *any* pair of signals.
The option exists to make that degeneracy inspectable and is tested; it is not
a faster approximation.

`return_phase=True` returns a `WaveletCoherenceResult` with `coherence`,
`phase`, `scales` and `frequencies`, so the return type follows the argument
rather than the data.

Coherence is normalised: it measures how *consistently* two series are related,
not how strong either is, and is numerically ill-conditioned in bands where
neither carries power.

### `delay_embed(x, dimension=1, delay=1) -> Array`
Delay-coordinate embedding, `(N - (m-1)tau, m)`. Shared by the recurrence
encoders in this module.

### Registry

`register_encoder(name, func)` / `get_encoder(name)` / `ENCODER_REGISTRY`.
Registered keys: `gaf`, `gadf`, `rp`, `spec`, `mtspec`, `chirplet`, `cwt`, `sst`, `ph`, `eph`, `mtf`, `otf`, `gdf`,
`msrp`, `ded`, `dtw`, `sax`, `msc`, `attn`, `vg`, `hvg`, `shapelet`, `mp`, `scat`,
`randproj`,
`ensemble`, plus the long-form aliases `visibility_graph`, `matrix_profile`,
`persistence_image`, `window_attention` and the legacy key `tpa`
(→ `window_attention`).

## Feature extraction

### `extract_feature_vector(img, bins=32, selected=None) -> Array`
Concatenate every registered descriptor for an image `(H, W)` or `(H, W, C)`.

**The output length is not a constant.** It depends on `bins` and on which
optional dependencies are installed, so query it instead of hard-coding it:

```python
from tscv_vision.features import feature_layout, feature_vector_length

feature_vector_length(bins=16)   # 662 with no optional extras installed
feature_layout(bins=16)          # {'intensity': 6, 'hist': 16, ...}
```

With only the core dependencies the layout is:

| Feature | bins=8 | bins=16 | bins=32 |
| --- | ---: | ---: | ---: |
| `intensity` | 6 | 6 | 6 |
| `hist` | 8 | 16 | 32 |
| `gradient` | 16 | 16 | 16 |
| `lbp` | 256 | 256 | 256 |
| `lbp_ri` | 256 | 256 | 256 |
| `lbp_uniform` | 59 | 59 | 59 |
| `glcm` | 12 | 12 | 12 |
| `gabor` | 16 | 16 | 16 |
| `edge_density` | 1 | 1 | 1 |
| `orientation` | 8 | 16 | 32 |
| `contour` | 1 | 1 | 1 |
| `fractal` | 1 | 1 | 1 |
| `fft` | 4 | 4 | 4 |
| `psd` | 2 | 2 | 2 |
| **total** | **646** | **662** | **694** |

Installing PyWavelets adds `wavelet` (6 values), giving 652 / 668 / 700. A
`(H, W, C)` input repeats the whole layout per channel.

### `feature_layout(bins=32, selected=None) -> dict[str, int]`
Per-extractor output sizes in concatenation order.

### `feature_vector_length(bins=32, selected=None) -> int`
Total length per channel.

### `extract_batch(images, bins=32, selected=None, *, lazy=False) -> Array | Iterator[Array]`
Features for a stack `(N, H, W[, C])`. `lazy=True` yields vectors one at a time
instead of allocating `(N, D)`.

### Local Binary Patterns

`lbp(img, radius=1)`, `lbp_ri(img, radius=1)` and `lbp_uniform(img, radius=1)`
implement LBP<sub>8,R</sub> with circular, bilinearly interpolated sampling
(Ojala et al., 2002) and match `skimage.feature.local_binary_pattern`
(`default` / `ror` / `nri_uniform`) on the image interior. `radius` genuinely
changes the sampling geometry. `lbp_uniform` returns 59 bins: 58 uniform
patterns plus a final shared bin for the non-uniform ones.

## Sliding windows

### `sliding_windows(x, size, hop=None, *, copy=False) -> Array`
Overlapping windows of length `size` every `hop` samples. `hop` defaults to
`size // 2`; `size` must be in `[2, N]`. Returns `(n_windows, size)`.

### `encode_sliding(x, encoder='gaf', *, size, hop=None, ...) -> (Array, Array)`
Encode each window. Returns `(encoded, starts)`. Accepts `metric`, `eps`,
`spec_win`, `spec_hop`, `spec_window`, `channel_fusion`, `cwt_scales`,
`use_gpu` and `lazy`.

### `features_for_sliding(x, *, encoder='gaf', size=128, hop=64, bins=32, feature_names=None, ...) -> (Array, Array)`
`encode_sliding` followed by `extract_batch`.

## Representations

`tscv_vision.representations` is one interface over every way the package turns
a series into something a model consumes, plus a registry queryable by
scientific provenance rather than by name.

```python
from tscv_vision.representations import get_representation, list_representations

rep = get_representation("gaf", image_size=32)
image = rep.transform(series)                  # (32, 32) for any series length

# Assemble an experiment from methods that are actually validated:
list_representations(canonical_method=True, min_validation_level=3)
# ['gadf', 'gaf', 'mp', 'mtf', 'mtspec', 'ph', 'scat']
```

### Interfaces

Three, kept distinct because conflating them is how test data reaches model
selection:

| Class | Contract | Leakage surface |
| --- | --- | --- |
| `Representation` | `transform(x)`, `info` | none — a pure function |
| `FittedRepresentation` | adds `fit(X, y=None)`, `fit_transform`, `check_fitted()` | `fit` must see training data only |
| `PretrainedRepresentation` | adds `encode(X)` | the pretraining corpus may overlap your evaluation data |

Batch helpers on every representation: `transform_many` (list, tolerates
ragged output), `transform_stack` (array, raises on ragged output),
`iter_transform` (lazy).

None inherit from scikit-learn. `as_sklearn(representation, *, stack=True)`
returns a transformer that fits the representation inside its own `fit`, so it
is safe in a `Pipeline` under cross-validation.

### `RepresentationInfo`

Every representation carries one, and the registry refuses an entry without it.

| Field | Meaning |
| --- | --- |
| `name`, `family` | registry key and grouping |
| `input_kind`, `output_kind` | shape contract |
| `trainable`, `pretrained`, `deterministic`, `differentiable` | behavioural flags |
| `dimension` | fixed output shape, `None` if input-dependent, `-1` marks one variable axis |
| `canonical_method` | `True` **only** when the code reproduces `reference` and a test pins it there |
| `reference` | citation, or `None` for project-defined transforms |
| `complexity` | time complexity in the series length |
| `validation_level` | `ValidationLevel`, 0–4 |
| `validated_by` | the tests backing that level |
| `notes` | how a project-defined variant differs from what its name suggests |

Construction is validated: `canonical_method=True` without a `reference`, or a
level above `SMOKE` without `validated_by`, raises.

### Registry

- `list_representations(*, family=None, input_kind=None, output_kind=None, trainable=None, pretrained=None, deterministic=None, canonical_method=None, min_validation_level=None, include_aliases=False) -> list[str]`
- `get_representation(name, **kwargs) -> Representation` — kwargs go to the encoder
- `get_representation_info(name) -> RepresentationInfo`
- `register_representation(name, factory, info, *, overwrite=False) -> None`
- `list_encoders(*, family=None, input_kind=None, output_kind=None, canonical_method=None, min_validation_level=None, include_aliases=False) -> list[str]`
- `get_encoder_metadata(name) -> RepresentationInfo`

### Deterministic adapters

`DeterministicRepresentation(name, *, image_size=None, nan_policy="raise", **params)`
wraps any registered encoder. Convenience classes with checked arguments:
`GAFRepresentation`, `RecurrencePlotRepresentation`, `SpectrogramRepresentation`,
`MTFRepresentation`, `PersistenceImageRepresentation`, `SAXRepresentation`.

`image_size` resamples the series with `paa` (Keogh et al., 2001) before
encoding, so a square-image encoder gives a fixed shape for any input length.
Series shorter than `image_size` are rejected rather than interpolated —
up-sampling would fabricate detail the encoder then reports as structure.
`SpectrogramRepresentation` does not accept it, because resampling before an
STFT changes what every frequency bin means; use `win`/`hop`.

### Fusion

`ConcatFusion(views, *, mode="concat", weights=None)` flattens each view and
reduces via `tscv_vision.fusion.fuse`. Its `info.validation_level` is the
weakest of the combined views. `LearnedFusion` is abstract: fusion weights are
parameters, so they must be estimated inside `fit`.

### Validation levels

See [encoder_validation.md](encoder_validation.md), generated from the
metadata by `python scripts/generate_encoder_validation.py` and checked for
staleness by `tests/test_representations.py`.

## Statistics

`tscv_vision.stats` implements the distribution functions and tests the package
needs without pulling in SciPy. Every routine is checked against `scipy.stats`.

- `welch_ttest(a, b) -> (statistic, pvalue, df)` — unequal-variance t-test with
  Welch–Satterthwaite degrees of freedom evaluated on the Student t
  distribution (not the normal approximation used before 0.2.0).
- `wilcoxon_signed_rank(x, y=None)` — exact for `n <= 25` without ties,
  normal approximation with tie correction otherwise.
- `friedman_test(scores, *, higher_is_better=True)` and `average_ranks(...)`
  over a `(n_datasets, n_methods)` matrix.
- `nemenyi_critical_difference(n_methods, n_datasets, alpha=0.05)`.
- `holm_bonferroni(pvalues)`.
- Building blocks: `betainc`, `gammainc_upper`, `normal_sf`, `student_t_sf`,
  `chi2_sf`.

## Benchmarking

`tscv_vision.evaluation` runs a leakage-safe comparison over UCR/UEA-style
datasets. See [benchmarks.md](benchmarks.md).

- `load_ucr_tsv(archive, name)` / `list_ucr_datasets(archive)`
- `Method(name, representation, classifier='knn1', bins=16, features=None)`
- `evaluate(dataset, method, *, seed=0) -> EvaluationResult`
- `run_benchmark(datasets, methods=None, *, seeds=(0,), out_dir=None, resume=True, n_jobs=1)`
- `compare_methods(results, *, alpha=0.05) -> Comparison`
- `summary_markdown(comparison) -> str`

## Model selection

`tscv_vision.pipeline`:

- `pipeline.select_features(X, y, *, method='mutual_info', k=10, cv=3, random_state=None)`
- `FeatureSelector(*, method, k, cv, random_state)` — sklearn transformer, so
  selection can sit **inside** a `Pipeline` and be re-fitted per fold.
- `AdaptivePipeline.optimize(X, y, *, n_iter=10)` — searches encoder and
  feature count with fold-local selection. The returned score is the best over
  searched configurations and is therefore optimistic.
- `AdaptivePipeline.nested_score(X, y, *, n_iter=10, outer_cv=5)` — unbiased
  estimate via nested cross-validation. **This is the number to report.**

`tscv_vision.automl.AutoTSCV` mirrors this: `validate(X, y)` warns when called
on the training data, and `nested_score(X, y, *, outer_splits=3)` repeats the
whole `fit` inside each outer split.

## Analytics (optional)

### `OcclusionExplainer(model, baseline=0.0)`
Window and frequency occlusion. `explain(series, window)` returns time and
frequency importances. These are **not** SHAP values — no coalitions are
sampled and no Shapley axiom holds. Use `shap_values` (which delegates to the
`shap` package) for Shapley attribution.

### `gaf_attribution(importance)` / `rp_attribution(importance)` / `spectrogram_attribution(importance)`
Collapse an importance matrix to a 1D time-domain signal.

### `group_significance(a, b) -> tuple[float, float]`
Two-sided Welch t-test; delegates to `stats.welch_ttest`.

### `cross_correlation_lag(x, y, max_lag=10) -> int`
Lag maximising the Pearson cross-correlation, pairing `x[t]` with `y[t + k]`,
so a positive result means `x` leads `y`. Associational only.

### `plot_importance(series, importance, title=None)`
Plot importance alongside the series (requires Matplotlib).

### `generate_counterfactual(series, model, target, step=0.1, max_iter=100) -> Array`
Gradient-based search for a nearby series moving the output toward `target`.

## Research utilities

### `group_mean_disparity(features, groups) -> dict[str, float]`
Per-group means plus `max_diff`. A screening statistic, not a fairness audit.

### `add_laplace_noise(features, scale, *, rng=None) -> Array`
Zero-mean Laplace noise of the given scale. No privacy semantics.

### `add_dp_noise(features, epsilon, *, sensitivity, rng=None) -> Array`
Laplace mechanism with scale `sensitivity / epsilon`. Gives ε-differential
privacy **only if** `sensitivity` is a correct upper bound on the L1 sensitivity
of the query that produced `features`; that bound follows from the query and
any clipping applied beforehand, never from the data. Composition across
releases is the caller's responsibility. `sensitivity` is a required keyword —
there is no safe default.
