# API Documentation

High-level overview of the public functions exposed by **tscv-vision**. All
functions accept and return NumPy `ndarray` objects and raise `ValueError` on
invalid inputs.

## Encoders

### `gaf(x, method='summation', *, use_numba=False, use_cython=False) -> Array`
Generate a Gramian Angular Field of shape `(N, N)` from a 1D signal `x`.

- `x`: 1D array `(N,)`
- `method`: `'summation'` or `'difference'`
- returns: image `(N, N)`

### `recurrence_plot(x, metric='euclidean', eps=None, *, use_numba=False, use_cython=False) -> Array`
Compute a recurrence matrix for `x`.

- `metric`: `'euclidean'` or `'manhattan'`
- `eps`: threshold for binary output; when `None` values are scaled to `[0, 1]`
- returns: `(N, N)` matrix

### `spectrogram(x, win=128, hop=64, pad=True, *, use_numba=False, use_cython=False) -> Array`
Short-time Fourier transform with magnitude output.

- `win`: window length
- `hop`: step between windows
- `pad`: pad signal so that `n_frames = ceil((N - win)/hop) + 1`
- returns: array `(win//2 + 1, n_frames)`

## Feature extraction

### `extract_feature_vector(img, bins=32, selected=None) -> Array`
Concatenate intensity stats `(6,)`, histogram `(bins,)`, gradient histogram
`(16,)` and LBP `(256,)` into a feature vector of length `6 + bins + 16 + 256`.

- `img`: image `(H, W)`
- `bins`: histogram bin count
- `selected`: optional subset of feature names

### `extract_batch(images, bins=32, selected=None) -> Array`
Vectorise a stack of images `(N, H, W)` and return features `(N, D)` using the
same options as `extract_feature_vector`.

## Sliding windows

### `sliding_windows(x, size, hop=None, *, copy=False) -> Array`
Return a view of overlapping windows of length `size` every `hop` samples.

- `size`: window length
- `hop`: step size (defaults to `size`)
- `copy`: force a copy instead of a view
- returns: array `(n_windows, size)`

### `encode_sliding(x, *, size, encoder='gaf', hop=None, **kw) -> (Array, Array)`
Encode each window with `encoder`.

- returns: `(encoded, starts)` where `encoded` has shape
  `(n_windows, H, W)` and `starts` holds window start indices

### `features_for_sliding(x, encoder='gaf', size=128, hop=64, bins=32) -> (Array, Array)`
Convenience wrapper combining `encode_sliding` and `extract_batch`.

## Analytics (optional)

### `TSHAPExplainer(model, baseline=0.0)`
Explain model predictions via window and frequency occlusion. Call
`explain(series, window)` to obtain time and frequency importances.

### `gaf_attribution(importance) -> Array`
Collapse a GAF importance matrix to a 1D signal.

### `rp_attribution(importance) -> Array`
Collapse a recurrence plot importance matrix to a 1D signal.

### `spectrogram_attribution(importance) -> Array`
Sum along the frequency axis of a spectrogram importance map to obtain time
importances.

### `plot_importance(series, importance, title=None)`
Plot importance values alongside the original series (requires Matplotlib).

### `generate_counterfactual(series, model, target, step=0.1, max_iter=100) -> Array`
Search for a nearby series that moves the model output toward `target` using
gradient-based updates.

