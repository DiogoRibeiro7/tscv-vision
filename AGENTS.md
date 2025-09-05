# AGENTS.md -- Development Instructions for tscv-vision

## Goal

Develop and maintain a Python package for computer-vision feature engineering of 1D time series, including:

- Time-series → image encoders (GAF, GADF, Recurrence Plot, Spectrogram)
- Feature extractors (intensity stats, histogram, gradient histogram, LBP)
- Sliding-window processing for batch encoding and feature extraction
- CLI for extracting features from `.npy` time series files

## Environment Setup

1. Use Python 3.11.
2. Install dependencies via Poetry:

```bash
poetry install
```

1. Run linters and tests:

```bash
poetry run ruff check .
poetry run mypy src
poetry run pytest -q
```

## Coding Standards

- Follow PEP 8.
- All public functions and classes must have type hints and docstrings.
- Keep dependencies minimal -- NumPy only for core logic. Optional extras (OpenCV, Torch) must be gated.
- Use NumPy vectorization where possible, avoid Python loops unless necessary.
- Maintain strict typing (`mypy --strict`).
- Ensure CI passes before merging.

## Repository Structure

- `src/tscv_vision/encoders.py`: Implement GAF, GADF, Recurrence Plot, Spectrogram.
- `src/tscv_vision/features.py`: Implement feature extractors.
- `src/tscv_vision/sliding.py`: Implement `sliding_windows` and `encode_sliding`.
- `src/tscv_vision/cli.py`: CLI entry point.
- `tests/`: Unit tests for each module.
- `samples/`: Scripts to generate example `.npy` files (no binaries tracked).

## Implementation Tasks

1. **Encoders**

  - GAF: summation and difference modes.
  - Recurrence Plot: support Euclidean and Manhattan distances, binary or continuous output.
  - Spectrogram: basic NumPy STFT.

2. **Feature Extraction**

  - Implement: intensity statistics, normalized histogram, gradient histogram (Sobel-like), LBP.
  - Compose into a unified feature vector.

3. **Sliding-Window Support**

  - Implement `sliding_windows` using stride tricks.
  - Implement `encode_sliding` to batch-process windows.

4. **CLI**

  - Add options for encoder type, histogram bins.
  - Add sliding-window parameters (`--sliding`, `--win-len`, `--hop`).
  - Output `.npz` with features and metadata.

5. **Testing**

  - Add smoke tests for all encoders.
  - Add shape/consistency tests for sliding-window outputs.
  - Test CLI via subprocess.

## PR Process

1. Create a feature branch from `develop`.
2. Implement changes and add tests.
3. Run `poetry run pytest -q` and ensure all tests pass.
4. Lint and type-check:

```bash
poetry run ruff check .
poetry run mypy src
```

1. Open a PR to `develop` with a clear description of changes.
2. After review and approval, merge into `develop`.
3. For releases, merge `develop` into `main` and tag the version.

## Additional Notes

- Keep modules small and cohesive.
- Avoid duplication; use helper functions where possible.
- Document all public APIs in the README.
- Ensure reproducibility of tests by setting NumPy random seeds when applicable.

--------------------------------------------------------------------------------

### `AGENTS.md`

```markdown
# AGENTS.md — Build Plan for a Coding Agent

This document instructs an autonomous coding agent how to develop and maintain **tscv-vision**.
The agent must follow these rules strictly.

---

## 1) Mission & Scope
**Mission:** Provide NumPy-first computer-vision feature engineering for 1D time series.
- Encode series → images: GAF (sum/diff), Recurrence Plot, STFT spectrogram.
- Extract features: intensity stats, histogram, gradient histogram, LBP.
- Support **sliding-window** pipelines at scale.
- Keep dependencies minimal (hard dep: `numpy`). Optional extras may be gated behind extras in later versions.

**Out of scope (v0.x):** Training CNNs, heavy CV deps, data labeling UIs.

---

## 2) Tech Constraints
- Python: `>=3.9,<3.13` (CI uses 3.11).
- Packaging: Poetry (`pyproject.toml` already provided).
- Lint: ruff; Types: mypy (strict); Tests: pytest.
- No heavy deps (OpenCV, Torch) in core package. Prepare extensibility hooks.

---

## 3) Repository Layout (expected)
```

src/tscv_vision/ **init**.py encoders.py # GAF/RP/Spectrogram features.py # Stats/Hist/Grad/LBP/compose sliding.py # sliding_windows, encode_sliding cli.py # tscv-features entrypoint

samples/ README.md # how to create example data

tests/ test_smoke.py test_sliding.py

README.md ROADMAP.md AGENTS.md pyproject.toml LICENSE .github/workflows/ci.yml .gitignore

````
---

## 4) Definition of Done (DoD)
- ✅ Public APIs type-annotated with docstrings, parameter validation, and clear error messages.
- ✅ Unit tests cover core paths and edge cases (≥90% for `encoders.py`, `sliding.py`, `features.py`).
- ✅ `ruff`, `mypy --strict`, `pytest` pass locally and in CI.
- ✅ README has runnable examples; CLI works for single and sliding modes.
- ✅ No import-time heavy deps; import cost minimal.

---

## 5) Task Queue (v0.1)
1\. **API hardening**
   - Enforce input dtypes/shapes for encoders and features.
   - Improve `spectrogram` short-signal behavior; consistent output shapes when padding.
2\. **Batch feature extraction**
   - Add `features.extract_batch(images, bins=32) -> np.ndarray` (stacked vectors).
   - Add `pipeline.features_for_sliding(x, ...)` thin wrapper returning `(F, starts)`.
3\. **Windowed dataset helper**
   - Provide `dataset.py` with `WindowedDataset` iterator for very long signals (streaming, low memory).
4\. **Docs & Examples**
   - Example notebook (no extra deps): encode + visualize + simple classifier sketch (NumPy-only featurization).
5\. **Testing**
   - Property tests: monotonic effects on hist bins; LBP invariances under constant offsets; RP symmetry.
6\. **CLI polish**
   - `--save-images` (npz stack) and `--save-meta` (starts, win_len, hop) flags with defaults.

---

## 6) API Specifications
### 6.1 Encoders
```python
# encoders.py

gaf(x: Array, method: Literal["summation","difference"] = "summation") -> Array
recurrence_plot(x: Array, metric: Literal["euclidean","manhattan"] = "euclidean", eps: float | None = None) -> Array
spectrogram(x: Array, win: int = 64, hop: int | None = None, window: Literal["hann","rect"] = "hann") -> Array
````

**Contracts:**

- `x` is 1D finite numeric. Raise `ValueError` otherwise.
- Output is `float64` in `[0,1]` or `[-1,1]` as documented; never NaN/inf.

### 6.2 Sliding

```python
# sliding.py
sliding_windows(x: Array, size: int, hop: int | None = None, *, copy: bool = False) -> Array
encode_sliding(x: Array, encoder: Literal["gaf","gadf","rp","spec"] = "gaf", *, size: int, hop: int | None = None, ...) -> tuple[Array, Array]
```

**Semantics:**

- `size >= 2`; default `hop = size//2` if None.
- Returns `(images, starts)` where `starts[i]` is the start index of window `i`.
- For spectrograms, pad time dimension so stacks align.

### 6.3 Features

```python
# features.py
intensity_stats(img: Array) -> Array               # (6,)
histogram(img: Array, bins: int = 32) -> Array     # (bins,)
gradient_histogram(img: Array, bins: int = 16) -> Array  # (bins,)
lbp(img: Array, radius: int = 1) -> Array          # (256,)
extract_feature_vector(img: Array, bins: int = 32) -> Array  # (6+bins+16+256,)
```

**Add (task 2):**

```python
extract_batch(images: Array, bins: int = 32) -> Array  # (N, D)
```

### 6.4 CLI

- **Single:** `tscv-features --encoder {gaf,gadf,rp,spec} --input in.npy --output out.npz --bins 32`
- **Sliding:** `... --sliding --win-len 128 --hop 64 [--rp-*, --spec-*]`
- Saves: `features` (1D or 2D), and for sliding: `window_starts`, `win_len`, `hop`.

--------------------------------------------------------------------------------

## 7) Coding Standards

- Typing: exhaustive annotations; `mypy --strict` clean.
- Docstrings: NumPy-style or Google-style; include shapes/ranges and raised exceptions.
- Validation: early checks for shape, dtype, finiteness; raise `ValueError` with precise messages.
- In-code comments: brief, purposeful; avoid redundancy with docstrings.
- Performance: use vectorized NumPy; prefer stride views; avoid Python loops in hot paths (except small 3×3 kernels / LBP where fine).
- Determinism: no RNG in library code except tests.

--------------------------------------------------------------------------------

## 8) Testing Policy

- Unit tests per function; cover edge cases (constant signals, very short signals, NaNs -> error).
- Property tests (lightweight, no Hypothesis required) using random seeds but fixed.
- Minimum coverage targets (soft):

  - encoders 90%, sliding 90%, features 85%, cli 70%.

Run:

```bash
poetry run ruff check .
poetry run mypy src
poetry run pytest -q
```

--------------------------------------------------------------------------------

## 9) Git & PR Workflow

- Branching: feature branches off `develop` (if present) or `main`.
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `build:`.
- PR Checklist:

  - [ ] Tests added/updated and passing
  - [ ] `ruff`/`mypy` pass
  - [ ] README/CLI help updated if flags changed
  - [ ] No new heavy deps

- CI must be green before merge.

Commands:

```bash
# create repo (local → GitHub)
gh repo create tscv-vision --public --source . --remote origin --push --branch main
# add develop buffer (optional)
git switch -c develop && git push -u origin develop
```

--------------------------------------------------------------------------------

## 10) Release Process

- Bump version in `pyproject.toml` (SemVer; start at 0.1.0).
- Tag `vX.Y.Z` on `main` after CI passes.
- Draft release notes (changes, API surface, breaking changes).

--------------------------------------------------------------------------------

## 11) Extensibility Hooks (for v0.2+)

- Encoder registry pattern (dict name → callable) to allow optional plugins (e.g., OpenCV/Torch) behind extras.
- Feature registry with compatible signature `(img: np.ndarray, **kwargs) -> np.ndarray`.
- Config serialization for pipelines (YAML/JSON) to reproduce features.

--------------------------------------------------------------------------------

## 12) Security & Licensing

- License: MIT (already included).
- Avoid executing untrusted code or loading arbitrary formats; only `.npy` inputs.
- Validate inputs rigorously; fail early and loudly.

--------------------------------------------------------------------------------

## 13) Roadmap Links

- Implement items in `ROADMAP.md` starting at v0.1; keep the file in sync with PRs.

--------------------------------------------------------------------------------

## 14) Acceptance Tests (Manual)

- Single-run example in README executes without errors.
- CLI sliding run saves `features` with expected shape `(n_windows, D)`.
- Visual sanity (dev-only): generate a small PNG from one encoder (not shipped) to eyeball patterns.

--------------------------------------------------------------------------------

## 15) Initial Issues for the Agent

- [ ] Add `features.extract_batch` and tests.
- [ ] Add `pipeline.features_for_sliding` wrapper and tests.
- [ ] Improve spectrogram padding consistency and document shapes.
- [ ] Add CLI flags `--save-images`, `--save-meta` (true by default) with tests.
- [ ] Create an example notebook under `examples/01_quickstart.ipynb` (pure NumPy plots; no seaborn).

--------------------------------------------------------------------------------

_End of AGENTS.md_
