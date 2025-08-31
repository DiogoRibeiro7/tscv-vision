# API Documentation

## Encoders

### gaf(x, method='summation') -> Array
Converts a 1D series to a Gramian Angular Field. `method` may be `'summation'` or `'difference'`. Returns a square image of shape `(N, N)`.

### recurrence_plot(x, metric='euclidean', threshold=None, binary=True) -> Array
Creates a recurrence plot using the specified distance metric. Returns a square image `(N, N)`.

### spectrogram(x, win=128, hop=64, pad=True) -> Array
Computes a short-time Fourier transform magnitude spectrogram with optional zero padding. Returns an array of shape `(win//2 + 1, ceil((N - win)/hop) + 1)`.

### visibility_graph(x) -> Array
Encodes the series as a natural visibility graph and returns a binary adjacency matrix `(N, N)`.

### shapelet_transform(x, k=3, length=None, seed=None) -> Array
Randomly samples `k` shapelets of length `length` and returns their distance maps `(k, N - length + 1)` scaled to `[0, 1]`.

### matrix_profile(x, m) -> Array
Computes the naive matrix profile for subsequence length `m`, returning a profile vector `(N - m + 1,)` scaled to `[0, 1]`.

## Streaming

### StreamingEncoder(size, hop=None, precision='high', use_gpu=False, incremental=None)
Encodes streaming samples into images with optional incremental updates, GPU
acceleration via CuPy, and adaptive precision control.

### benchmark_streaming(encoder, samples, repeats=1) -> dict
Benchmarks a `StreamingEncoder`, returning throughput, latency, and memory
usage statistics.

## Feature Extraction

### extract_feature_vector(img, bins=32, selected=None) -> Array
Combines intensity statistics, histogram, gradient histogram, and LBP into a single feature vector.

### extract_batch(images, bins=32) -> Array
Vectorises multiple images at once, returning an array of shape `(N, D)`.

## Analytics

### TSHAPExplainer(model, baseline=0.0)
Explains a model by occluding time windows and frequency components. Use
`explain(series, window)` to obtain time and frequency importance arrays.

### gaf_attribution(importance) -> Array
Collapses a Gramian Angular Field importance matrix back to 1D time
importance.

### rp_attribution(importance) -> Array
Equivalent to `gaf_attribution` for recurrence plot images.

### spectrogram_attribution(importance) -> Array
Sums over the frequency axis to obtain time-window importance.

### plot_importance(series, importance, title=None)
Interactive Matplotlib plot that overlays importance values on a series.

### generate_counterfactual(series, model, target, step=0.1, max_iter=100) -> Array
Uses gradient-based updates to find a nearby time series that moves the
model prediction toward `target`.

## Domains

### DomainAdapter(extractor, base_estimator=None)
Fine-tunes a pre-trained scikit-learn estimator on features extracted from domain-specific series.

### PrototypicalClassifier()
Few-shot classifier that predicts by nearest class prototype.

### uncertainty_sampling(model, series, n_samples) -> Array
Return indices of the most uncertain samples for active learning.

### classification_metrics(y_true, y_pred) -> dict
Return accuracy, precision, recall, and F1 scores.

Each domain module (e.g., `finance`, `healthcare`) provides `generate_*` utilities and
`augment_*` functions to synthesise and augment data for transfer learning pipelines.

## MLOps

### validate_features(features) -> None
Raise ``ValueError`` if any feature is NaN or infinite.

### DriftDetector(bins=32, threshold=0.1)
Detect distribution drift via histogram KL divergence.

### assign_variant(key) -> str
Deterministically map a key to variant ``"A"`` or ``"B"``.

### ResourceScaler(max_replicas=100)
``required_replicas(throughput, target)`` estimates the worker count.

### ModelRegistry()
Thread-safe registry for model versions, metrics, and deployment status.

### ABTester()
Collect variant metrics and ``compare()`` for p-value and lift.

### safe_encode(series, primary, fallback, timeout=0.1) -> Array
Use ``primary`` encoder with a ``fallback`` on errors or timeouts.

### batch_process(data, func, batch_size=64, start=0, progress=None) -> List[Array]
Process large datasets in batches with optional resumption and progress callback.

### create_feature_service() -> FastAPI
HTTP service exposing feature extraction; Prometheus metrics at ``/metrics``.

### create_monitoring_app(detector=None) -> FastAPI
Health and drift endpoints with optional Prometheus metrics.

### FeastWriter(repo_path=None)
Light-weight wrapper around ``feast`` for pushing features.

## Pipelines

### AdaptivePipeline(encoders=None, feature_select='mutual_info', k=10, cv=3, random_state=None)
Automatically choose the best encoder and feature subset via cross-validated scoring. Use
``fit(X, y)`` then ``transform(X)`` to obtain selected features. ``optimize`` performs
Bayesian search over encoder and feature-count combinations.

### FeatureEnsemble(encoders, cv=3, random_state=None)
Learns weights for multiple encoders based on cross-validated scores and returns a weighted
concatenation of their features.

### select_features(X, y, method='mutual_info', k=10, cv=3, random_state=None) -> Array
Utility to select the top ``k`` features using mutual information, correlation, or
stability-based ranking.

## Sliding

### sliding_windows(x, size, hop) -> Array
Provides a read-only view over sliding windows of the series.

### encode_sliding(x, size, encoder='gaf', hop=None, **kw) -> Tuple[Array, Array]
Encodes each window using the selected encoder. Returns the encoded stack and window start indices.

### features_for_sliding(x, encoder='gaf', size=128, hop=64, bins=32) -> Tuple[Array, Array]
Convenience wrapper that encodes and extracts features for each window.

## CLI

See `tscv-features --help` for the full command-line interface.
