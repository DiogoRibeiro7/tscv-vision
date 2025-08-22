# Changelog

## [0.10.0] - 2025-08-13
- Add domain-specific modules for finance, healthcare, IoT, audio, astronomy,
  climate and manufacturing, each with toy pre-trained models

## [0.9.0] - 2025-08-13
- Introduce research utilities for experiment tracking, fairness reports and
  differential privacy
- Add random projection encoder and Mamba/RetNet neural stubs

## [0.8.0] - 2025-08-13
- Add analytics module with SHAP/LIME wrappers, saliency maps, counterfactuals,
  causal lag analysis, and simple reporting utilities

## [0.7.0] - 2025-08-13
- Add optional MLOps utilities for FastAPI feature services, drift detection and
  feature validation

## [0.6.0] - 2025-08-13
- Introduce optional PyTorch-based neural encoders (CNN and Vision Transformer)
- Add contrastive learning utilities, attention fusion and simple VAE generator
- Provide style transfer helper and lightweight NAS search stub

## [0.5.0] - 2025-08-13
- Provide scikit-learn, PyTorch and TensorFlow integration helpers
- Allow exporting feature arrays as ONNX ``TensorProto`` objects
- Save metadata about encoders, features and sliding windows in outputs

## [0.4.0] - 2025-08-13
- Support memory-mapped ``.npy`` files and chunked iteration for ``.npz``/Parquet
- Add multiprocessing utilities and ``--parallel`` CLI flag
- Introduce Arrow/Parquet/HDF5 interoperability helpers

## [0.3.0] - 2025-08-13
- Add fusion utilities for combining outputs from multiple encoders
- Introduce temporal aggregation functions for sliding features
- Provide encoder registry with user-defined registration API
- Expand CLI with multi-encoder fusion and aggregation options

## [0.2.0] - 2025-08-13
- Add multichannel encoding with fusion strategies
- Introduce WindowedDataset streaming API and feature registry
- Support selective feature extraction via CLI `--features`

## [0.1.1] - 2025-08-13
- Add example notebooks and CLI walkthrough
- Integrate Codecov and CLI smoke tests in CI
- Fix deprecation warning by moving ruff settings under `[tool.ruff.lint]`

## [0.1.0] - 2025-08-13
- Initial pre-release with GAF/GADF, recurrence plot and spectrogram encoders
- Feature extractors for intensity stats, histograms, gradients and LBP
- Sliding-window batch pipeline and CLI with image/metadata saving
