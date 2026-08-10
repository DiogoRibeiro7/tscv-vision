# Contributing

Thank you for helping improve `tscv-vision`. This project values focused
changes, reproducible tests, and a small dependency footprint.

## Development Setup

Use Python 3.11 for local development unless you are explicitly testing another
supported version.

```bash
git clone https://github.com/DiogoRibeiro7/tscv-vision.git
cd tscv-vision
poetry install
poetry run pre-commit install
```

## Workflow

1. Open an issue for non-trivial bugs or feature proposals.
2. Create a branch from `main`.
3. Keep changes scoped to one behavior or documentation improvement.
4. Add or update tests for behavior changes.
5. Run the validation commands before opening a pull request.

```bash
poetry run pre-commit run --all-files
poetry run ruff check .
poetry run mypy src
poetry run pytest -q
```

## Coding Standards

- Public functions and classes need type hints and docstrings.
- Core logic should stay NumPy-first and avoid unnecessary heavy dependencies.
- Optional integrations must be gated behind extras or lazy imports.
- Prefer vectorized NumPy operations for hot paths.
- Raise `ValueError` with precise messages for invalid user inputs.
- Do not introduce generated binaries or sample datasets into git.

## Tests

Tests live under `tests/` and use pytest markers for optional environments.

```bash
poetry run pytest -q
poetry run pytest -m gpu
poetry run pytest -m optional
```

Only run GPU and optional tests when the relevant dependencies and hardware are
available.

## Pull Requests

A complete pull request should include:

- A clear summary of what changed and why.
- Tests for new behavior or a note explaining why tests are not applicable.
- Documentation updates when public APIs, CLI flags, or release behavior change.
- Passing lint, type checks, and tests.

## Release Changes

Release-related pull requests should also update:

- `pyproject.toml`
- `setup.py`
- `src/tscv_vision/__init__.py`
- `CHANGELOG.md`
- `.zenodo.json`

Follow [docs/release-checklist.md](docs/release-checklist.md) for the full
release process.
