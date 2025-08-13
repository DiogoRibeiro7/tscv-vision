# Roadmap

## v0.1.x (Patch releases)
- Pre-commit hooks and Codecov integration.
- High-priority: handle NaNs in encoders.
- Document platform-specific installation notes.

## v0.2.0
- [x] Multichannel/Multivariate support (per-channel and fused images).
- [x] Save/load feature configurations via feature registry.
- [x] Streaming-friendly WindowedDataset API.
- Medium-priority: streaming API for online features.

## v0.3.0
- [x] Time-frequency fusion across multiple encoders.
- [x] Temporal aggregation functions for sliding features.
- [x] Custom encoder registry for user-defined encoders.

## v0.4.0
- [x] Out-of-core processing for large datasets.
- [x] Parallel and distributed execution helpers.
- [x] Interoperability with Arrow/Parquet/HDF5 formats.

## v0.5.0
- [x] ML framework wrappers for scikit-learn, PyTorch and TensorFlow.
- [x] ONNX feature export utility.
- [x] Metadata-rich feature outputs for reproducibility.

## v1.0.0
- Plugin system for optional encoders/features.
- Benchmark suite on public datasets; reproducible scripts.
- Long-term: add Continuous Wavelet Transform encoder.

## Feedback
1. Handle NaNs gracefully in encoders (**high priority**, next patch release).
2. Provide streaming API for real-time extraction (**medium priority**, next minor release).
3. Support additional encoder types such as CWT (**long-term**, major release).
