# Test Matrix

The test suite uses markers to manage optional dependencies and runtime requirements.

| Marker | Description | Dependencies |
| ------ | ----------- | ------------ |
| `gpu` | GPU-accelerated tests; skipped when CuPy is missing | `cupy` |
| `slow` | Long-running stress or performance tests | none |
| `optional` | Tests that rely on optional third-party packages | varies (e.g. `torch`, `sklearn`, `matplotlib`) |

## Running Tests

Run only the core tests (default):

```bash
poetry run pytest -q
```

Include optional tests:

```bash
poetry run pytest -m optional
```

Run GPU tests when CuPy is available:

```bash
poetry run pytest -m gpu
```

Skip slow tests explicitly:

```bash
poetry run pytest -m "not slow"
```

Combine markers with logical expressions, e.g. run optional GPU tests:

```bash
poetry run pytest -m "gpu and optional"
```

Collect coverage information and show slowest tests:

```bash
poetry run pytest --cov=tscv_vision --durations=5
```
