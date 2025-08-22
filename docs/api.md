# API Documentation

## Encoders

### gaf(x, method='summation') -> Array
Converts a 1D series to a Gramian Angular Field. `method` may be `'summation'` or `'difference'`. Returns a square image of shape `(N, N)`.

### recurrence_plot(x, metric='euclidean', threshold=None, binary=True) -> Array
Creates a recurrence plot using the specified distance metric. Returns a square image `(N, N)`.

### spectrogram(x, win=128, hop=64, pad=True) -> Array
Computes a short-time Fourier transform magnitude spectrogram with optional zero padding. Returns an array of shape `(win//2 + 1, ceil((N - win)/hop) + 1)`.

## Feature Extraction

### extract_feature_vector(img, bins=32, selected=None) -> Array
Combines intensity statistics, histogram, gradient histogram, and LBP into a single feature vector.

### extract_batch(images, bins=32) -> Array
Vectorises multiple images at once, returning an array of shape `(N, D)`.

## Sliding

### sliding_windows(x, size, hop) -> Array
Provides a read-only view over sliding windows of the series.

### encode_sliding(x, size, encoder='gaf', hop=None, **kw) -> Tuple[Array, Array]
Encodes each window using the selected encoder. Returns the encoded stack and window start indices.

### features_for_sliding(x, encoder='gaf', size=128, hop=64, bins=32) -> Tuple[Array, Array]
Convenience wrapper that encodes and extracts features for each window.

## CLI

See `tscv-features --help` for the full command-line interface.
