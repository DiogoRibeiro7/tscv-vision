# Repository: `tscv-vision`

Computer-vision feature engineering for 1D time series. Convert sequences into images (e.g., Gramian Angular Fields, Recurrence Plots, Spectrogram-like maps) and extract robust visual features for downstream ML.

---

## Files

### `README.md`

````markdown
# tscv-vision

Computer-vision feature engineering for 1D time series.

## Why
Turning 1D signals into images lets you leverage decades of CV features and powerful CNN embeddings. This repo provides:
- **Image encoders**: Gramian Angular Fields (GAF), Recurrence Plots (RP), simple Spectrogram-like maps (STFT).
- **Feature extractors**: intensity stats, histograms, gradient-based texture, Local Binary Patterns (LBP).
- **Simple API** with type hints and docstrings.
- **Zero heavy deps** by default: NumPy only. (Torch/OpenCV are optional extras you may add later.)

## Install (Poetry)
```bash
poetry install
poetry run pytest -q
````

## Quick start

```python
import numpy as np
from tscv_vision import encoders, features

x = np.sin(np.linspace(0, 12*np.pi, 512)) + 0.1*np.random.randn(512)
img = encoders.gaf(x, method="summation")  # (H,W) float64 in [-1,1]
vec = features.extract_feature_vector(img, bins=32)
print(vec.shape)  # 1D feature vector
```

## CLI

```bash
poetry run tscv-features --encoder gaf --bins 32 --input samples/sine.npy --output features.npz
```

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## License

MIT
