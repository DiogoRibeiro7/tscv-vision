# tscv-vision

NumPy-first computer‑vision feature engineering for 1D time series. Encode sequences as
images, describe them with classical CV features, and process long signals efficiently.

## Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Command line interface](#command-line-interface)
5. [Development](#development)
6. [Optional dependencies](#optional-dependencies)
7. [Sample data](#sample-data)
8. [License](#license)

## Features

### Encoders

Turn a 1D sequence `x` of shape `(N,)` into a 2D image `(N, N)` or `(F, T)`:

- `encoders.gaf` / `encoders.gadf` – Gramian Angular Fields
- `encoders.recurrence_plot` – distance matrix as an image
- `encoders.spectrogram` – STFT magnitude spectrogram

### Feature extraction

Operate on grayscale images `img` of shape `(H, W)`:

- `features.intensity_stats(img) -> (6,)`
- `features.histogram(img, bins=32) -> (bins,)`
- `features.gradient_histogram(img, bins=16) -> (bins,)`
- `features.lbp(img, radius=1) -> (256,)`
- `features.extract_feature_vector(img, bins=32) -> (310,)`
- `features.extract_batch(images, bins=32) -> (N, 310)`

### Sliding‑window helpers

Efficiently process long series:

- `sliding.sliding_windows(x, win_len, hop)` returns overlapping windows using stride tricks
- `sliding.encode_sliding(x, encoder, win_len, hop)` encodes every window

### CLI

`tscv-features` reads `.npy` files, encodes them, and exports features.

## Installation

The core package only depends on NumPy:

```bash
pip install tscv-vision
```

### Optional extras

Install extras to enable additional capabilities:

- CLI & YAML config files: `pip install tscv-vision[cli]`
- Analytics and visualization: `pip install tscv-vision[analytics]`
- CuPy-accelerated encoders: `pip install tscv-vision[gpu]`

For local development use Poetry:

```bash
poetry install
```

## Quick start

```python
import numpy as np
from tscv_vision import encoders, features, sliding

x = np.sin(np.linspace(0, 4*np.pi, 128))

# Encode a series to an image (128x128)
img = encoders.gaf(x)

# Extract a feature vector (310 dims when bins=32)
vec = features.extract_feature_vector(img, bins=16)
print(vec.shape)

# Multiple images → stacked features
batch = features.extract_batch(np.stack([img, img]), bins=16)
print(batch.shape)

# Sliding-window pipeline
windows = sliding.sliding_windows(x, win_len=64, hop=32)
encoded = sliding.encode_sliding(x, encoders.gaf, win_len=64, hop=32)
print(encoded.shape)
```

## Command line interface

`tscv-features` loads time-series from `.npy` files, encodes them, and extracts features.

```bash
# create a sample sine wave
python samples/generate.py

# inspect available flags
tscv-features --help

# single image features
tscv-features --encoders gaf --input samples/sine.npy --output out.npz --features intensity,hist

# sliding-window batch
tscv-features --encoders gaf,spec --fusion mean --sliding --win-len 128 --input samples/sine.npy --output out_sliding.npz --aggregate mean --parallel 2 --no-save-images
```

The output `.npz` always contains a `features` array and JSON `metadata`. Sliding runs add `window_starts`, `win_len`, and `hop` unless `--no-save-meta` is specified. Images can be skipped via `--no-save-images`.

## Development

Run linters and tests before contributing:

```bash
poetry run ruff check .
poetry run mypy src
poetry run pytest -q
```

## Optional dependencies

- `cupy` for GPU acceleration
- `torch` for neural encoders
- `scikit-learn` for ML integration
- `pyyaml` for CLI config files
- `matplotlib`, `seaborn`, and `pywavelets` for analytics

Install them via extras, for example: `pip install tscv-vision[gpu,cli]`.

## Documentation

Further guides live under the `docs/` directory:

- [API reference](docs/api.md) – function signatures, shapes, and optional analytics tools
- [Deployment](docs/deployment.md) – container builds, Kubernetes manifests, and runtime safety
- [Performance](docs/performance.md) – streaming tips, GPU usage, and benchmarking helpers
- [Release checklist](docs/release-checklist.md) – versioning and packaging steps
- [Test matrix](docs/test-matrix.md) – pytest markers and dependency combinations
- [Troubleshooting](docs/troubleshooting.md) – common errors and their fixes

## Sample data

Generate a demo sine wave with `python samples/generate.py`. See `samples/README.md` for details.

## Maintainer

Diogo Ribeiro  
ESMAD – Instituto Politécnico do Porto  
GitHub: [@DiogoRibeiro7](https://github.com/DiogoRibeiro7)  
ORCID: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)  
Email: dfr@esmad.ipp.pt

## License

MIT

