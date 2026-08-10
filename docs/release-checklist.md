# Release checklist

## Testing
- `poetry run ruff check .`
- `poetry run mypy src`
- `poetry run pytest -q`
- `./scripts/integration.sh`
- `python -m build`
- `python -m twine check dist/*`

## PyPI trusted publishing
- Create or claim the `tscv-vision` project on PyPI.
- Configure a trusted publisher for:
  - Owner: `DiogoRibeiro7`
  - Repository name: `tscv-vision`
  - Workflow filename: `publish.yml`
- Do not create or store a PyPI API token for GitHub Actions. The publish workflow uses OpenID Connect.

## Versioning
- Update `pyproject.toml` and `CHANGELOG.md`
- Update `setup.py` and `src/tscv_vision/__init__.py`
- Commit with `build: release vX.Y.Z`
- Tag the commit: `git tag vX.Y.Z && git push origin vX.Y.Z`
- The `Publish` GitHub Actions workflow publishes tagged releases to PyPI.

## Documentation
- Regenerate README examples and ensure links are valid
- Proofread files under `docs/`
- Ensure new flags are documented in `README.md` and `docs/api.md`

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
