# tscv-vision

[![CI](https://github.com/DiogoRibeiro7/tscv-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/tscv-vision/actions/workflows/ci.yml)
[![Publish](https://github.com/DiogoRibeiro7/tscv-vision/actions/workflows/publish.yml/badge.svg)](https://github.com/DiogoRibeiro7/tscv-vision/actions/workflows/publish.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)

`tscv-vision` is a framework for constructing, learning, combining, and
evaluating structured representations of time-series data. It includes
NumPy-first image encoders, classical feature descriptors, representation
metadata, fusion utilities, leakage-safe evaluation, and sliding-window
processing for long signals.

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
- Leakage-safe evaluation: nested cross-validation helpers, SciPy-free
  statistical tests, and a UCR/UEA benchmark harness that freezes its raw
  outputs.
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
| `ml` | `pip install "tscv-vision[ml]"` | scikit-learn transformer, pipelines, model selection |
| `research` | `pip install "tscv-vision[research]"` | Benchmark harness, including the ROCKET baseline |
| `analytics` | `pip install "tscv-vision[analytics]"` | SHAP, LIME, UMAP, plotting, wavelets |
| `domains` | `pip install "tscv-vision[domains]"` | Domain adapters backed by scikit-learn |
| `speed` | `pip install "tscv-vision[speed]"` | Numba JIT encoder paths |
| `spectral` | `pip install "tscv-vision[spectral]"` | DPSS tapers for the multitaper spectrogram |
| `scattering` | `pip install "tscv-vision[scattering]"` | Kymatio wavelet scattering |
| `gpu` | `pip install "tscv-vision[gpu]"` | CuPy-accelerated encoder paths |
| `io` | `pip install "tscv-vision[io]"` | Arrow / Parquet / HDF5 readers and writers |
| `streaming` | `pip install "tscv-vision[streaming]"` | Redis, Kafka and RabbitMQ stream sources |
| `distributed` | `pip install "tscv-vision[distributed]"` | Dask-backed parallel map |
| `mlops` | `pip install "tscv-vision[mlops]"` | FastAPI, Prometheus, Feast integrations |
| `torch` | `pip install "tscv-vision[torch]"` | Torch-based neural components |
| `onnx` | `pip install "tscv-vision[onnx]"` | ONNX tensor export |

Every optional import in the package belongs to one of these extras;
`tests/test_docs_sync.py` fails the build if a new one appears without a
documented install route.

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

Every encoder accepts a `nan_policy` and is reachable by registry name through
`encoders.get_encoder(name)`.

| Encoder | Function or name | Output |
| --- | --- | --- |
| GAF | `encoders.gaf(x, method="summation")` or `gaf` | `(N, N)` |
| GADF | `encoders.gaf(x, method="difference")` or `gadf` | `(N, N)` |
| Recurrence plot | `encoders.recurrence_plot(x)` or `rp` | `(N, N)` |
| Spectrogram | `encoders.spectrogram(x)` or `spec` | `(F, T)` |
| Multitaper spectrogram | `encoders.multitaper_spectrogram(x)` or `mtspec` | `(F, T)` |
| Chirplet transform | `encoders.chirplet_transform(x)` or `chirplet` | `(F, T)` |
| Continuous wavelet | `encoders.cwt(x, scales)` or `cwt` | `(scales, N)` |
| Synchrosqueezed CWT | `encoders.synchrosqueezed_cwt(x, fs=...)` or `sst` | `(frequencies, N)` |
| Markov Transition Field | `encoders.mtf(x)` or `mtf` | `(N, N)` |
| Ordinal transition field | `encoders.ordinal_transition_field(x)` or `otf` | `(W, W)` |
| Gramian Difference Field | `encoders.gdf(x)` or `gdf` | `(N, N)` |
| Persistence diagram | `encoders.persistence_diagram(x)` | `(n_pairs, 2)` |
| Persistence image | `encoders.persistence_image(x, bins)` or `ph` | `(bins, bins)` |
| Extrema persistence histogram | `encoders.extrema_persistence_histogram(x)` or `eph` | `(bins, bins)` |
| SAX image | `encoders.sax(x)` or `sax` | `(segments, segments)` |
| DTW cost matrix | `encoders.dtw_matrix(x)` or `dtw` | `(N, N)` |
| Visibility graph | `encoders.visibility_graph(x)` or `vg` | `(N, N)` |
| Horizontal visibility graph | `encoders.horizontal_visibility_graph(x)` or `hvg` | `(N, N)` |
| Matrix profile | `encoders.matrix_profile(x, m)` or `mp` | `(N - m + 1,)` |
| Shapelet transform | `encoders.shapelet_transform(x, k)` or `shapelet` | `(k, N - L + 1)` |
| Window attention | `encoders.window_attention(x, window)` or `attn` | `(W, W)` |
| Delay-embedding density | `encoders.delay_embedding_density(x)` or `ded` | `(bins, bins)` |
| Multi-scale RP / conv | `msrp`, `msc` | stacked |
| Random projection | `encoders.random_projection_image(x)` or `randproj` | `(size, size)` |
| Scattering | `scattering.scattering_transform(x)` or `scat` | `(paths, T)` |
| Ensemble | `encoders.ensemble(x, names)` or `ensemble` | stacked or averaged |

Encoders taking more than one series live in `tscv_vision.multivariate`:

| Encoder | Function | Output |
| --- | --- | --- |
| Cross recurrence plot | `multivariate.cross_recurrence_plot(x, y)` | `(N_x, N_y)` |
| Joint recurrence plot | `multivariate.joint_recurrence_plot(X)` | `(W, W)` |
| Wavelet coherence | `multivariate.wavelet_coherence(x, y)` | `(scales, N)` |

### Feature Extractors

| Function | Description |
| --- | --- |
| `features.intensity_stats(img)` | mean, std, min, max, skewness, kurtosis |
| `features.histogram(img, bins=32)` | normalized intensity histogram |
| `features.gradient_histogram(img, bins=16)` | Sobel-like gradient magnitude histogram |
| `features.lbp(img, radius=1)` | LBP<sub>8,R</sub>, circular sampling, matches scikit-image |
| `features.lbp_ri` / `features.lbp_uniform` | rotation-invariant and uniform variants |
| `features.glcm_features`, `gabor_features`, `orientation_histogram` | texture and orientation |
| `features.edge_density`, `contour_ratio`, `fractal_dimension` | shape descriptors |
| `features.fft_features`, `power_spectral_density`, `wavelet_stats` | spectral descriptors |
| `features.extract_feature_vector(img, bins=32)` | unified feature vector |
| `features.extract_batch(images, bins=32)` | stacked feature matrix |

The unified vector's length depends on `bins` **and** on which optional
packages are installed, so query it rather than hard-coding it:

```python
from tscv_vision.features import feature_layout, feature_vector_length

feature_vector_length(bins=32)   # 694 with core dependencies only
feature_layout(bins=32)          # {'intensity': 6, 'hist': 32, 'lbp': 256, ...}
```

## Benchmarking

`tscv_vision.evaluation` compares encoders against standard baselines
(1-NN Euclidean, raw features, optionally ROCKET) on UCR/UEA datasets using
their predefined train/test splits. It writes one CSV row per
`(dataset, method, seed)`, a manifest pinning package versions and the git
commit, and a summary applying Demšar-style checks: Friedman test,
average ranks, Nemenyi critical difference and Holm-corrected pairwise
Wilcoxon tests.

```bash
python -m tscv_vision.evaluation --archive /data/UCRArchive_2018 --out results/ucr
```

The harness appends each completed row, resumes existing `results.csv` files by
default, and accepts `--n-jobs` for independent dataset/method/seed
combinations. The archive is not redistributable and is not vendored here. See
[docs/benchmarks.md](docs/benchmarks.md).

For model selection, prefer the nested-CV entry points — they re-run the whole
selection procedure inside each outer fold, so the reported score is not
contaminated by the choices it evaluates:

```python
pipe.nested_score(X, y)      # AdaptivePipeline
auto.nested_score(X, y)      # AutoTSCV
```

## Representations

`tscv_vision.representations` puts one interface over every encoder and lets
you select them by scientific provenance rather than by name:

```python
from tscv_vision.representations import get_representation, list_representations

rep = get_representation("gaf", image_size=32)
image = rep.transform(series)          # (32, 32) whatever the series length

rep.info.canonical_method              # True — reproduces Wang & Oates (2015)
rep.info.validation_level.label        # 'LEVEL 3 — reference'

# Build an experiment from methods that are actually validated:
list_representations(canonical_method=True, min_validation_level=3)
# ['gadf', 'gaf', 'mp', 'mtf', 'mtspec', 'ph', 'scat']
```

Every representation carries a `RepresentationInfo` with its family, reference,
complexity, and a validation level from 0 (smoke-tested) to 4 (benchmarked on
real data). The dataclass refuses to claim more than the tests deliver: marking
something canonical without a reference, or above smoke level without naming
the tests that back it, raises at construction.

The three interfaces — `Representation`, `FittedRepresentation`,
`PretrainedRepresentation` — are kept apart because they have different leakage
profiles. A fitted representation refuses to transform before `fit`, and
`as_sklearn()` wraps any of them so the fitting happens inside a cross-validation
fold. scikit-learn is not required to use them.

See [docs/encoder_validation.md](docs/encoder_validation.md) for the per-encoder
matrix, generated from the metadata.

### Representation analysis

`tscv_vision.analysis` includes NumPy-only tools for comparing representation
spaces before combining them. Start with linear CKA:

```python
from tscv_vision.analysis import representation_similarity

similarity = representation_similarity({
    "gaf": Z_gaf,
    "cwt": Z_cwt,
    "rp": Z_rp,
})
```

The same module reports mean redundancy, pairwise fusion gains over the best
single representation, and entropy-based effective rank for collapse checks.

## Scientific Naming Policy

A function carries the name of a published method only when it implements that
method **and** has a test pinning it to a reference implementation
(`tests/test_reference_equivalence.py`, checked against scikit-image, SciPy,
pyts, ripser, persim and stumpy) or to its published formula
(`tests/test_encoder_definitions.py`). Everything else is named descriptively,
and its docstring states what it is and what it is not.

Six names changed in 0.2.0 under this policy — `persistence_image`, `tpa`,
`TSHAPExplainer`, `cross_causal_lag`, `bias_report` and `add_dp_noise`. The old
names keep working until 0.3.0 and emit `DeprecationWarning`. See the
[changelog](CHANGELOG.md) for the reasoning behind each.

## Documentation

- [API reference](docs/api.md)
- [Encoder validation matrix](docs/encoder_validation.md)
- [Benchmarks](docs/benchmarks.md)
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
poetry run pytest -q                   # core suite
poetry run pytest -m optional          # optional integrations
poetry run pytest -m optional tests/test_reference_equivalence.py
```

The default `pytest` invocation excludes the `optional`, `slow` and `gpu`
markers; CI runs each in its own job so a regression in the optional
integrations cannot hide behind the default marker expression. See
[docs/test-matrix.md](docs/test-matrix.md).

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
