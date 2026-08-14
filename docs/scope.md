# Package scope

`tscv-vision` ships two surfaces today:

- **Validated core and research surface.** NumPy-first encoders, feature
  extraction, sliding windows, representation metadata, leakage-safe
  evaluation, statistics, and benchmark tooling. This is the surface the
  package claims as the supported framework for time-series representation.
- **Experimental integration surface.** Domain adapters, neural helpers,
  multimodal utilities, irregular-series helpers, MLOps helpers, and ONNX/Torch
  integration glue. These modules remain importable for compatibility, but they
  are not evidence for the package's scientific core and are candidates for a
  future `tscv-vision-contrib` package or examples tree.

The split is encoded in `tscv_vision.VALIDATED_CORE_MODULES` and
`tscv_vision.CONTRIB_MODULES` so tooling and docs can stay aligned. New modules
should be added deliberately to one of those sets.

## Validated core

- `aggregation`
- `analysis`
- `analytics`
- `automl`
- `dataset`
- `encoders`
- `evaluation`
- `features`
- `fusion`
- `gpu`
- `io`
- `multivariate`
- `parallel`
- `pipeline`
- `representations`
- `research`
- `scattering`
- `sliding`
- `stats`
- `streaming`

## Experimental integrations

- `domains`
- `irregular`
- `ml_integration`
- `mlops`
- `multimodal`
- `neural`
