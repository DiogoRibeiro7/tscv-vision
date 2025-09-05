# tscv-vision

NumPy-first computer-vision feature engineering for 1D time series.

## Install

Minimal setup only requires NumPy:

```bash
pip install tscv-vision
```

Optional extras add specific features:

- `pip install tscv-vision[cli]` – YAML config support for the CLI
- `pip install tscv-vision[analytics]` – advanced analytics and visualization
- `pip install tscv-vision[gpu]` – CuPy-accelerated encoders

For development use Poetry:

```bash
poetry install
```

## Quick start

```python
import numpy as np
from tscv_vision import encoders, features

x = np.sin(np.linspace(0, 4*np.pi, 128))
img = encoders.gaf(x)
vec = features.extract_feature_vector(img, bins=16)
print(vec.shape)

# multiple images
batch = features.extract_batch(np.stack([img, img]), bins=16)
print(batch.shape)
```

## CLI

`tscv-features` encodes `.npy` files and extracts features.

```bash
# create a sample sine wave
python samples/generate.py

# single image features
tscv-features --encoders gaf --input samples/sine.npy --output out.npz --features intensity,hist

# sliding-window batch
tscv-features --encoders gaf,spec --fusion mean --sliding --win-len 128 --input samples/sine.npy --output out_sliding.npz --aggregate mean --parallel 2 --no-save-images
```

Outputs contain a `features` array and JSON `metadata`. Sliding runs add `window_starts`, `win_len` and `hop` unless `--no-save-meta` is used.

## Optional dependencies

- `cupy` for GPU acceleration
- `torch` for neural encoders
- `scikit-learn` for ML integration
- `pyyaml` for CLI config files
- `matplotlib`, `seaborn` and `pywavelets` for analytics

Install them via extras, for example: `pip install tscv-vision[gpu,cli]`.

Core functionality requires only NumPy.

## Sample data

Generate a demo sine wave with `python samples/generate.py`. See `samples/README.md` for details.

## License

MIT

