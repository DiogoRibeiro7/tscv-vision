# API Documentation

## Encoders

### `gaf(x, method='summation', *, use_numba=False, use_cython=False) -> Array`
Return a Gramian Angular Field image of shape `(N, N)`.

### `recurrence_plot(x, metric='euclidean', eps=None, *, use_numba=False, use_cython=False) -> Array`
Return a recurrence matrix. If `eps` is provided the result is binary; otherwise values are scaled to `[0, 1]`.

### `spectrogram(x, win=128, hop=64, pad=True, *, use_numba=False, use_cython=False) -> Array`
Compute a magnitude spectrogram with windows of length `win` and step `hop`. The result has shape `(win//2 + 1, n_frames)`.

## Feature extraction

### `extract_feature_vector(img, bins=32, selected=None) -> Array`
Combine intensity statistics, histogram, gradient histogram and LBP into a single feature vector.

### `extract_batch(images, bins=32, selected=None) -> Array`
Vectorise a batch of images `images` with shape `(N, H, W)` and return features of shape `(N, D)`.

## Sliding windows

### `sliding_windows(x, size, hop=None, *, copy=False) -> Array`
Return a view of overlapping windows of length `size` taken every `hop` samples.

### `encode_sliding(x, *, size, encoder='gaf', hop=None, **kw) -> (Array, Array)`
Encode each window using the specified encoder. Returns the encoded stack and the window start indices.

### `features_for_sliding(x, encoder='gaf', size=128, hop=64, bins=32) -> (Array, Array)`
Convenience wrapper that encodes windows and extracts features for each.

## Analytics (optional)

### `TSHAPExplainer(model, baseline=0.0)`
Explain model predictions via window and frequency occlusion. Call `explain(series, window)` to obtain time and frequency importances.

### `gaf_attribution(importance) -> Array`
Collapse a GAF importance matrix to a 1D signal.

### `rp_attribution(importance) -> Array`
Collapse a recurrence plot importance matrix to a 1D signal.

### `spectrogram_attribution(importance) -> Array`
Sum along the frequency axis of a spectrogram importance map to obtain time importances.

### `plot_importance(series, importance, title=None)`
Plot importance values alongside the original series (requires Matplotlib).

### `generate_counterfactual(series, model, target, step=0.1, max_iter=100) -> Array`
Search for a nearby time series that moves the model output toward `target` using gradient-based updates.

