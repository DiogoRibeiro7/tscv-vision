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

## Zenodo archiving
- Enable the `DiogoRibeiro7/tscv-vision` repository in Zenodo's GitHub integration.
- Keep `.zenodo.json` metadata current before each release.
- Create a GitHub Release for the pushed version tag so Zenodo archives the release and mints a DOI.
- Add the Zenodo DOI badge to `README.md` after the first archive exists.

## Versioning
- Update `pyproject.toml` and `CHANGELOG.md`
- Update `setup.py` and `src/tscv_vision/__init__.py`
- Update `.zenodo.json` and `CITATION.cff`
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
- Experimental/contrib extras: `torch`, `mlops`, `domains`, `onnx`

## Installation
- Minimal: `pip install tscv-vision`
- With CLI support: `pip install tscv-vision[cli]`
- Full analytics stack: `pip install tscv-vision[analytics]`
- GPU acceleration: `pip install tscv-vision[gpu]`
