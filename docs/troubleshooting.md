# Troubleshooting

## Common issues

### NaN or infinite values
Ensure the input series contains only finite numbers. Use `np.nan_to_num` or filter invalid values before encoding.

### Window size larger than series
`sliding_windows` requires `size <= len(series)`. Adjust the window or pad the series.

### Spectrogram memory usage
Large windows and small hops increase memory. Use `encode_sliding(..., lazy=True)` to stream results for huge datasets.

### Unsupported encoder name
Check available encoders with `tscv_vision.encoders.list_encoders()` and ensure names match.
