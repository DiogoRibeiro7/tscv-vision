# Release checklist

## Testing
- `poetry run ruff check .`
- `poetry run mypy src`
- `poetry run pytest -q`
- `./scripts/integration.sh`

## Optional dependencies
- `cli`: YAML configuration (`pyyaml`)
- `analytics`: advanced analytics and visualization (`scikit-learn`, `shap`, `lime`, `umap-learn`, `matplotlib`, `seaborn`, `pywavelets`)
- `gpu`: CuPy-accelerated encoders (`cupy`)
- `torch`: neural encoders (`torch`)
- `mlops`: model serving and monitoring (`fastapi`, `prometheus-client`, `feast`)
- `domains`: domain adapters (`scikit-learn`)

## Installation
- Minimal: `pip install tscv-vision`
- With CLI support: `pip install tscv-vision[cli]`
- Full analytics stack: `pip install tscv-vision[analytics]`
- GPU acceleration: `pip install tscv-vision[gpu]`
