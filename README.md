# tscv-vision

[![CI](https://github.com/DiogoRibeiro7/tscv-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/tscv-vision/actions/workflows/ci.yml)
[![Publish](https://github.com/DiogoRibeiro7/tscv-vision/actions/workflows/publish.yml/badge.svg)](https://github.com/DiogoRibeiro7/tscv-vision/actions/workflows/publish.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)

`tscv-vision` is a NumPy-first Python package for computer-vision feature
engineering of one-dimensional time series. It converts signals into image
representations, extracts classical image descriptors, and scales the workflow
to long signals with sliding-window processing.

The project is designed for research, reproducible feature extraction, and
lightweight production pipelines where the core dependency footprint should
stay small.

## Highlights

- Time-series imaging encoders: Gramian Angular Field, Gramian Angular
  Difference Field, recurrence plots, spectrograms, and additional registry
  encoders.
- Classical feature extractors: intensity statistics, normalized histograms,
  gradient histograms, Local Binary Patterns, and batch extraction.
- Sliding-window pipelines for long 1D or multichannel signals.
- CLI for `.npy` inputs with metadata-rich `.npz` outputs.
- Optional extras for analytics, GPU acceleration, neural integrations, MLOps,
  and domain adapters.
- Release-ready packaging with PyPI Trusted Publishing and Zenodo metadata.

## Installation

Install the core package:

```bash
pip install tscv-vision
```

Install optional extras only when needed:

| Extra | Command | Purpose |
| --- | --- | --- |
| `cli` | `pip install "tscv-vision[cli]"` | YAML configuration files |
| `analytics` | `pip install "tscv-vision[analytics]"` | SHAP, LIME, UMAP, plotting, wavelets |
| `domains` | `pip install "tscv-vision[domains]"` | Domain adapters backed by scikit-learn |
| `gpu` | `pip install "tscv-vision[gpu]"` | CuPy-accelerated encoder paths |
| `mlops` | `pip install "tscv-vision[mlops]"` | FastAPI, Prometheus, Feast integrations |
| `torch` | `pip install "tscv-vision[torch]"` | Torch-based neural components |

For local development:

```bash
git clone https://github.com/DiogoRibeiro7/tscv-vision.git
cd tscv-vision
poetry install
```

## Quick Start

```python
import numpy as np
from tscv_vision import encoders, features, sliding

x = np.sin(np.linspace(0, 4 * np.pi, 128))

# Encode a series as a 2D image.
img = encoders.gaf(x)
print(img.shape)  # (128, 128)

# Extract a feature vector from one image.
vec = features.extract_feature_vector(img, bins=32)
print(vec.shape)

# Extract a stacked feature matrix from multiple images.
batch = features.extract_batch(np.stack([img, img]), bins=32)
print(batch.shape)

# Encode overlapping windows and keep their start indices.
images, starts = sliding.encode_sliding(x, encoder="gaf", size=64, hop=32)
print(images.shape, starts)

# Extract sliding-window feature vectors directly.
matrix, starts = sliding.features_for_sliding(x, encoder="gaf", size=64, hop=32)
print(matrix.shape, starts)
```

## Command Line

Generate sample data:

```bash
python samples/generate.py
```

Extract features from one signal:

```bash
tscv-features \
  --encoders gaf \
  --input samples/sine.npy \
  --output out.npz \
  --features all \
  --bins 32
```

Run a sliding-window, multi-encoder pipeline:

```bash
tscv-features \
  --encoders gaf,spec \
  --fusion concat \
  --sliding \
  --win-len 128 \
  --hop 64 \
  --input samples/sine.npy \
  --output out_sliding.npz \
  --save-images \
  --save-meta
```

The output `.npz` contains feature arrays plus JSON metadata. Sliding runs can
include `window_starts`, `win_len`, `hop`, and encoded image stacks.

## Feature Surface

### Encoders

| Encoder | Function or name | Output |
| --- | --- | --- |
| GAF | `encoders.gaf(x, method="summation")` or `gaf` | `(N, N)` |
| GADF | `encoders.gaf(x, method="difference")` or `gadf` | `(N, N)` |
| Recurrence plot | `encoders.recurrence_plot(x)` or `rp` | `(N, N)` |
| Spectrogram | `encoders.spectrogram(x)` or `spec` | `(F, T)` |
| Continuous wavelet | `cwt` | `(scales, N)` |

### Feature Extractors

| Function | Description |
| --- | --- |
| `features.intensity_stats(img)` | mean, std, min, max, skewness, kurtosis |
| `features.histogram(img, bins=32)` | normalized intensity histogram |
| `features.gradient_histogram(img, bins=16)` | Sobel-like gradient magnitude histogram |
| `features.lbp(img, radius=1)` | Local Binary Pattern histogram |
| `features.extract_feature_vector(img, bins=32)` | unified feature vector |
| `features.extract_batch(images, bins=32)` | stacked feature matrix |

## Documentation

- [API reference](docs/api.md)
- [Deployment guide](docs/deployment.md)
- [Performance guide](docs/performance.md)
- [Release checklist](docs/release-checklist.md)
- [Test matrix](docs/test-matrix.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Roadmap](ROADMAP.md)

## Development

The project uses Poetry, Ruff, mypy, pytest, and pre-commit.

```bash
poetry install
poetry run pre-commit run --all-files
poetry run ruff check .
poetry run mypy src
poetry run pytest -q
```

Build and check distribution artifacts:

```bash
python -m build
python -m twine check dist/*
```

The default package build is pure Python. To opt into building the optional
Cython extension locally, set:

```bash
TSCV_BUILD_EXT=1 python -m build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution workflow, coding
standards, and pull request expectations.

## Releases, PyPI, and Zenodo

Tagged releases are published to PyPI through GitHub Actions using PyPI Trusted
Publishing with OpenID Connect. No PyPI API token is stored in the repository.

Zenodo archiving is configured through [.zenodo.json](.zenodo.json). After the
repository is enabled in Zenodo's GitHub integration, each GitHub Release can be
archived with a version DOI. Add the DOI badge here after the first archive is
created.

## Citation

If you use `tscv-vision` in academic work, cite the archived Zenodo release
once available, or use the metadata in [CITATION.cff](CITATION.cff).

```text
Diogo Ribeiro. tscv-vision: Computer-vision feature engineering for 1D time series.
https://github.com/DiogoRibeiro7/tscv-vision
```

## Support and Security

- General questions and usage problems: open a GitHub Discussion or issue.
- Bug reports and feature requests: use the issue templates.
- Security concerns: follow [SECURITY.md](SECURITY.md).

## Maintainer

- Diogo Ribeiro
- School of Media Arts and Design, Polytechnic of Porto, Portugal
- GitHub: [@DiogoRibeiro7](https://github.com/DiogoRibeiro7)
- ORCID: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)
- Email: dfr@esmad.ipp.pt

## License

`tscv-vision` is distributed under the [MIT License](LICENSE).
