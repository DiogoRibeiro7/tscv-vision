# tscv-vision

NumPy-first computer-vision feature engineering for 1D time series.

Encode series as images (Gramian Angular Fields, Recurrence Plots, simple
spectrograms) and extract visual features such as intensity statistics,
histograms, gradient histograms and Local Binary Patterns.

See `examples/03_cli_workflow.md` for an end-to-end CLI walkthrough.

## Neural encoders (optional)

Install the package with the `torch` extra to enable learnable CNN and
Vision Transformer adapters:

```bash
poetry install -E torch
```

```python
from tscv_vision.neural import TorchCNNEncoder

encoder = TorchCNNEncoder()
features = encoder.encode(img.astype("float32"))
```

## Install

```bash
poetry install
```

## Analytics & interpretability

Install with the `analytics` extra to enable optional SHAP/LIME wrappers and
projection utilities:

```bash
poetry install -E analytics
```

```python
from tscv_vision.analytics import saliency_map

grad = saliency_map(lambda x: x.sum(), np.arange(10.0))
```

Install optional MLOps helpers with:

```bash
poetry install -E mlops
```

## Research utilities

Install with the `research` extra to enable experiment tracking, fairness checks
and differential-privacy noise:

```bash
poetry install -E research
```

```python
import numpy as np
from tscv_vision.research import track_experiment, bias_report

log = track_experiment({"encoder": "randproj"}, "series.npy", "runs")
report = bias_report(np.array([0.1, 0.2, 0.4]), np.array([0, 0, 1]))
```

## Domain-specific models

Install with the `domains` extra to enable lightweight transfer-learning helpers:

```bash
poetry install -E domains
```

```python
import numpy as np
from tscv_vision.domains import DomainAdapter, finance

series = [finance.generate_price_series(50, drift=d) for d in (-0.01, 0.01)]
labels = np.array([0, 1])
adapter = DomainAdapter(finance.microstructure_features)
adapter.fit(series, labels)
```

## Next-generation architecture (experimental)

Version 2.0 introduces an optional plugin-based architecture for advanced
research workflows.  Register custom encoders, build graph-based feature
pipelines and even generate Python code from high-level specifications:

```python
from tscv_vision.nextgen import Node, PipelineGraph, registry

def scale(x: float) -> float:
    return 2 * x

registry.register("scale", scale)
g = PipelineGraph()
g.add_node(Node(name="out", op="scale", deps=["inp"]))
result = g.run({"inp": 3.0})
```

The graph can be visualised (`tscv_vision.nextgen.visual.to_dot`) or executed in
parallel (`tscv_vision.nextgen.distribute.execute_distributed`).

## Quick start

```python
import numpy as np
from tscv_vision import encoders, features

x = np.sin(np.linspace(0, 8*np.pi, 256))
img = encoders.gaf(x, method="summation")
vec = features.extract_feature_vector(img, bins=16)
print(vec.shape)

# Batch from multiple images
batch = features.extract_batch(np.stack([img, img]), bins=16)
print(batch.shape)

# Select specific feature extractors
vec_small = features.extract_feature_vector(img, selected=["intensity", "hist"])
print(vec_small.shape)
```

## Built-in encoders

- `gaf` / `gadf` – Gramian Angular Fields
- `rp` – Recurrence Plot
- `spec` – STFT spectrogram
- `vg` – Visibility Graph adjacency
- `shapelet` – Shapelet Transform distance maps
- `mp` – Matrix Profile

## Multi-modal utilities

The ``multimodal`` module provides lightweight helpers for combining
heterogeneous inputs and adapting features across domains.  Examples
include weighted fusion of multi-variate series, simple metadata
concatenation, CORAL domain alignment and graph-based propagation for
related signals:

```python
from tscv_vision.multimodal import fuse_series, coral_align

series = np.vstack([x, x * 2])
fused = fuse_series(series, method="pca", n_components=1)

aligned = coral_align(np.random.randn(10, 3), np.random.randn(8, 3))
```

The spectrogram encoder pads the tail with zeros so all samples are covered and
returns an array of shape ``(win//2 + 1, ceil((N - win)/hop) + 1)``.

### Sliding windows

```python
from tscv_vision.sliding import features_for_sliding, encode_sliding

feats, starts = features_for_sliding(x, encoder="rp", size=64, hop=32, bins=16)
print(feats.shape, starts.shape)

# Multichannel encoding
t = np.linspace(0, 8*np.pi, 256)
x_multi = np.column_stack([np.sin(t), np.cos(t)])
imgs, starts = encode_sliding(x_multi, size=128, channel_fusion="stack")
print(imgs.shape)

# Streaming dataset
from tscv_vision import WindowedDataset
ds = WindowedDataset(x, size=64, hop=32)
feat, meta = next(iter(ds))
```

### Real-time streaming

Process samples as they arrive with adaptive buffering and anomaly triggers. The
encoder supports incremental updates, optional GPU acceleration via CuPy, and
adaptive precision scaling for throughput-sensitive deployments:

```python
from tscv_vision.streaming import StreamingEncoder

stream = StreamingEncoder(size=64, hop=16, anomaly_threshold=0.9)
for sample in sensor():
    for img in stream.push(sample):
        ...  # use encoded image
```

See ``docs/performance.md`` for tuning tips and benchmarking helpers.

## Domain-specific modules

Utilities in :mod:`tscv_vision.domains` provide encoders and lightweight models
tailored for particular application areas.  For example, market microstructure
analysis for finance or heart-rate variability for healthcare:

```python
from tscv_vision.domains import finance, healthcare

fin_feats = finance.microstructure_features(prices)
hr, sdnn = healthcare.ecg_features(ecg)
```

## MLOps & production deployment

Optional utilities under ``mlops`` help deploy the library at scale.  A FastAPI
service can expose feature extraction over HTTP, Prometheus metrics track
requests, a model registry versions encoders and A/B tests compare variants.
Monitoring endpoints expose health and drift metrics while ``safe_encode`` and
``batch_process`` add graceful degradation and large-batch processing:

```python
from tscv_vision.mlops import (
    ABTester,
    DriftDetector,
    ModelRegistry,
    batch_process,
    create_feature_service,
    create_monitoring_app,
    safe_encode,
)

app = create_feature_service()
monitor = create_monitoring_app(DriftDetector())
registry = ModelRegistry()
tester = ABTester()

# A/B test metrics
tester.add("A", 0.6)
tester.add("B", 0.7)
result = tester.compare()
```

Feature vectors may be validated and pushed to a Feast feature store for
versioned storage:

```python
from tscv_vision.mlops import FeastWriter, validate_features

validate_features(vec)
store = FeastWriter(repo_path="./feature_repo")
store.push("entity", {"gaf_feat": vec})
```

## CLI

```bash
# Single encoder
poetry run tscv-features --encoders gaf --input series.npy --output out.npz \
    --features intensity,hist

# Multiple encoders with fusion and temporal aggregation
poetry run tscv-features --encoders gaf,spec --fusion mean --sliding --win-len 128 \
    --input series.npy --output out_sliding.npz --aggregate mean --parallel 2 \
    --no-save-images
```

The output `.npz` always contains a `features` array and a JSON `metadata`
field describing encoders, feature extractors and parameters. For sliding runs
it also stores `window_starts`, `win_len` and `hop` unless `--no-save-meta` is
supplied. Use `--no-save-images` to disable saving encoded images. ``--parallel``
controls the number of worker processes for encoding.

### ML integration

```
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tscv_vision.ml_integration import SklearnFeatureTransformer

pipe = Pipeline([
    ("feat", SklearnFeatureTransformer("gaf")),
    ("clf", LogisticRegression()),
])

pipe.fit(X_train, y_train)
```

`TorchFeatureDataset` and `tf_feature_dataset` provide similar wrappers for
PyTorch and TensorFlow. Use :func:`tscv_vision.ml_integration.to_onnx_tensor` to
export feature arrays as ONNX tensors for deployment.

### Custom encoders

Register a new encoder at runtime:

```python
from tscv_vision import encoders

def my_encoder(x):
    return encoders.gaf(x)  # toy example

encoders.register_encoder("my_gaf", my_encoder)
```

### AutoML utilities

Basic helpers under :mod:`tscv_vision.automl` can analyse a time series,
rank feature importance and perform lightweight evolutionary search over
hyper-parameters:

```python
from tscv_vision import automl

cfg = automl.suggest_encoders(x)
order = automl.rank_feature_importance(features, labels)
```

### Adaptive feature engineering pipeline

Automatically select the most informative encoder and features:

```python
from tscv_vision.pipeline import AdaptivePipeline
import numpy as np

X = np.random.rand(32, 64)
y = (X.mean(axis=1) > 0.5).astype(int)
pipe = AdaptivePipeline(random_state=0)
feats = pipe.fit_transform(X, y)
```

Use :func:`tscv_vision.benchmark.benchmark_pipeline` to compare pipeline
configurations.

### Out-of-core and data formats

`WindowedDataset` can stream directly from a memory-mapped `.npy` file:

```python
ds = WindowedDataset("large.npy", size=64, hop=32)
```

Utilities in :mod:`tscv_vision.io` provide optional helpers for Arrow,
Parquet, and HDF5 formats (requires extra dependencies).

## Documentation

API and troubleshooting guides live under `docs/`:

- `docs/api.md` – function signatures and return shapes
- `docs/troubleshooting.md` – common errors and fixes
- `docs/performance.md` – tuning tips and benchmarks

## License

MIT

