# tscv-vision

NumPy-first computer-vision feature engineering for 1D time series.

Encode series as images (Gramian Angular Fields, Recurrence Plots, simple
spectrograms) and extract visual features such as intensity statistics,
histograms, gradient histograms and Local Binary Patterns.

Example notebooks are provided under `examples/`:

- `01_basic_usage.ipynb` – encode a signal and extract features
- `02_sliding_window.ipynb` – batch features via sliding windows
- `03_cli_workflow.md` – end-to-end CLI walkthrough

## Install

```bash
poetry install
```

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

### Out-of-core and data formats

`WindowedDataset` can stream directly from a memory-mapped `.npy` file:

```python
ds = WindowedDataset("large.npy", size=64, hop=32)
```

Utilities in :mod:`tscv_vision.io` provide optional helpers for Arrow,
Parquet, and HDF5 formats (requires extra dependencies).

## License

MIT

