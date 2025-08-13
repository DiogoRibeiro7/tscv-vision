# Roadmap

## v0.1 (MVP)
- Encoders: GAF (sum/diff), Recurrence Plot, STFT-based spectrogram.
- Features: stats, histogram, gradient histogram, LBP.
- CLI + simple dataset stubs + tests + CI.

## v0.2
- Multichannel/Multivariate support (per-channel and fused images).
- Sliding-window encoding (rolling images) with batch API.
- Save/load feature configurations.

## v0.3
- Optional extras: OpenCV keypoints (ORB/SIFT if available), Torch CNN embeddings (timm).
- Pluggable feature registry.

## v0.4
- Benchmark suite on public datasets; reproducible scripts.
- Docs site with examples and comparisons.
