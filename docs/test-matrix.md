# Test Matrix

The test suite uses markers to manage optional dependencies and runtime requirements.

| Marker | Description | Dependencies |
| ------ | ----------- | ------------ |
| `gpu` | GPU-accelerated tests; skipped when CuPy is missing | `cupy` |
| `slow` | Long-running stress or performance tests | none |
| `optional` | Tests that rely on optional third-party packages | varies (e.g. `torch`, `sklearn`, `matplotlib`) |

`pytest`'s default `addopts` is `-m 'not slow and not gpu and not optional'`.
That is a convenience for local runs, **not** a statement that the excluded
tests do not matter: CI runs each marker in its own job so nothing hides behind
the default expression.

| CI job | Command | Covers |
| ------ | ------- | ------ |
| `core` | `pytest` | NumPy-only surface, plus the encoder definition tests and the docs-sync guard |
| `optional` | `pytest -m optional` | scikit-learn, torch, TensorFlow, ONNX, matplotlib integrations |
| `reference` | `pytest -m optional tests/test_reference_equivalence.py` | numerical equivalence with scikit-image, SciPy, pyts, ripser, persim, stumpy |
| `benchmark-smoke` | `python -m tscv_vision.evaluation --synthetic` | the benchmark harness stays runnable |

## Validation layers

| Layer | File | Runs by default |
| ----- | ---- | --------------- |
| Definition checks (published formula, re-implemented naively) | `tests/test_encoder_definitions.py` | yes |
| Reference equivalence (third-party implementations) | `tests/test_reference_equivalence.py` | no (`optional`) |
| Documentation sync (signatures, dimensions, registry, versions) | `tests/test_docs_sync.py` | yes |
| Statistical behaviour without SciPy | `tests/test_stats.py` | yes |
| Benchmark harness | `tests/test_evaluation.py` | yes (skipped without scikit-learn) |

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
