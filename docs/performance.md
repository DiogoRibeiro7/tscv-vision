# Performance Guidelines

- Prefer vectorised encoders (`gaf`, `recurrence_plot`, `spectrogram`) and avoid Python loops in tight paths.
- Use `extract_batch` to preallocate feature arrays for large image sets.
- For long signals, enable `lazy=True` in `encode_sliding` to stream windows instead of materialising them.
- Benchmark with `%timeit` in your environment to choose window sizes and hops.
