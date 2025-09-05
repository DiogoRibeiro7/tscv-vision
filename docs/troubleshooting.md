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

### Missing optional dependency
Errors like `ImportError: No module named 'cupy'` indicate that an extra is
required. Install the appropriate extra, e.g. `pip install tscv-vision[gpu]`.

### Shape mismatch
When concatenating features ensure arrays have matching leading dimensions. The
`extract_batch` helper guarantees consistent shapes for batches.

### GPU not detected
Some containers hide GPUs by default. Verify access with `nvidia-smi` and ensure
the `NVIDIA_VISIBLE_DEVICES` environment variable is set.

### CLI cannot find input file
Paths are resolved relative to the working directory. Pass absolute paths or run
`pwd` to confirm the current directory.
