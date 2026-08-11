"""Numerical validation of every scientific encoder against its definition.

Each test re-implements the published formula in the most direct, obviously
correct way — usually an explicit Python loop — and asserts that the optimised
implementation in :mod:`tscv_vision.encoders` agrees. These are *definition*
checks and need no third-party packages, so they run in the default suite;
``tests/test_reference_equivalence.py`` additionally compares against external
reference implementations when those are installed.

References
----------
Wang & Oates (2015), "Imaging Time-Series to Improve Classification and
Imputation", IJCAI (GAF, MTF).  Eckmann et al. (1987) (recurrence plots).
Lin et al. (2007), "Experiencing SAX", DMKD 15:107-144.  Lacasa et al. (2008),
PNAS 105:4972-4975 (visibility graph).  Yeh et al. (2016), ICDM (matrix
profile).  Adams et al. (2017), JMLR 18(8):1-35 (persistence images).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tscv_vision import encoders


def _series(n: int = 24, seed: int = 0) -> np.ndarray:
    return np.asarray(np.random.default_rng(seed).normal(size=n), dtype=float)


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    if span == 0:
        return np.zeros_like(x)
    return (x - x.min()) / span * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Gramian Angular Fields


def test_gasf_matches_definition() -> None:
    """GASF[i, j] = cos(phi_i + phi_j) on the [-1, 1]-scaled series."""

    x = _series()
    phi = np.arccos(_minmax(x))
    expected = np.empty((x.size, x.size))
    for i in range(x.size):
        for j in range(x.size):
            expected[i, j] = math.cos(phi[i] + phi[j])
    np.testing.assert_allclose(encoders.gaf(x), expected, atol=1e-12)


def test_gadf_matches_definition() -> None:
    """GADF[i, j] = sin(phi_i - phi_j)."""

    x = _series(seed=1)
    phi = np.arccos(_minmax(x))
    expected = np.empty((x.size, x.size))
    for i in range(x.size):
        for j in range(x.size):
            expected[i, j] = math.sin(phi[i] - phi[j])
    np.testing.assert_allclose(encoders.gaf(x, method="difference"), expected, atol=1e-12)


def test_gaf_algebraic_identities() -> None:
    """GASF is symmetric with unit diagonal in x^2; GADF is antisymmetric."""

    x = _series(seed=2)
    z = _minmax(x)
    gasf = encoders.gaf(x)
    gadf = encoders.gaf(x, method="difference")
    np.testing.assert_allclose(gasf, gasf.T, atol=1e-12)
    np.testing.assert_allclose(gadf, -gadf.T, atol=1e-12)
    # cos(2 arccos z) = 2 z^2 - 1
    np.testing.assert_allclose(np.diag(gasf), 2 * z**2 - 1.0, atol=1e-12)
    np.testing.assert_allclose(np.diag(gadf), 0.0, atol=1e-12)
    # GASF = z_i z_j - sqrt(1-z_i^2) sqrt(1-z_j^2)
    root = np.sqrt(np.clip(1.0 - z**2, 0.0, None))
    np.testing.assert_allclose(
        gasf, np.outer(z, z) - np.outer(root, root), atol=1e-10
    )


# ---------------------------------------------------------------------------
# Recurrence plot


def test_recurrence_plot_matches_definition() -> None:
    x = _series(seed=3)
    n = x.size
    dist = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = abs(x[i] - x[j])
    dist = dist / (dist.max() + 1e-12)
    np.testing.assert_allclose(encoders.recurrence_plot(x), 1.0 - dist, atol=1e-12)
    np.testing.assert_allclose(
        encoders.recurrence_plot(x, eps=0.3), (dist <= 0.3).astype(float), atol=1e-12
    )


# ---------------------------------------------------------------------------
# Markov Transition Field


def test_mtf_matches_definition() -> None:
    """MTF[i, j] = P(state_j | state_i) with quantile bins."""

    x = _series(48, seed=4)
    bins = 4
    edges = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    states = np.digitize(x, edges)
    trans = np.zeros((bins, bins))
    for i in range(x.size - 1):
        trans[states[i], states[i + 1]] += 1.0
    row_sums = trans.sum(axis=1, keepdims=True)
    trans = trans / np.maximum(row_sums, 1e-12)
    expected = np.empty((x.size, x.size))
    for i in range(x.size):
        for j in range(x.size):
            expected[i, j] = trans[states[i], states[j]]
    np.testing.assert_allclose(encoders.mtf(x, bins=bins), expected, atol=1e-12)


def test_mtf_rows_of_transition_matrix_are_probabilities() -> None:
    x = _series(64, seed=5)
    img = encoders.mtf(x, bins=5)
    assert np.all(img >= 0.0) and np.all(img <= 1.0)


# ---------------------------------------------------------------------------
# DTW


def test_dtw_matrix_matches_definition() -> None:
    """Self-DTW accumulated-cost recursion, normalised and inverted."""

    x = _series(12, seed=6)
    n = x.size
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(x[i] - x[j])
            if i == 0 and j == 0:
                cost[i, j] = 0.0
            elif i == 0:
                cost[i, j] = cost[i, j - 1] + d
            elif j == 0:
                cost[i, j] = cost[i - 1, j] + d
            else:
                cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    expected = 1.0 - cost / (cost.max() + 1e-12)
    np.testing.assert_allclose(encoders.dtw_matrix(x), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# SAX


def test_sax_gaussian_breakpoints_match_published_table() -> None:
    """Breakpoints from Lin et al. (2007), Table 3."""

    published = {
        3: [-0.43, 0.43],
        4: [-0.67, 0.0, 0.67],
        5: [-0.84, -0.25, 0.25, 0.84],
        6: [-0.97, -0.43, 0.0, 0.43, 0.97],
        7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
        8: [-1.15, -0.67, -0.32, 0.0, 0.32, 0.67, 1.15],
    }
    for alphabet, expected in published.items():
        got = encoders._gaussian_breakpoints(alphabet)
        np.testing.assert_allclose(got, expected, atol=5e-3)


def test_sax_symbols_are_equiprobable_on_gaussian_data() -> None:
    x = np.random.default_rng(7).normal(size=20000)
    symbols = encoders.sax_symbols(x, segments=20000, alphabet=4)
    counts = np.bincount(symbols, minlength=4) / symbols.size
    np.testing.assert_allclose(counts, 0.25, atol=0.02)


def test_sax_gaussian_is_invariant_to_affine_rescaling() -> None:
    """Standard SAX z-normalises, so shifting/scaling must not change the word."""

    x = _series(64, seed=8)
    base = encoders.sax_symbols(x, segments=8, alphabet=5)
    np.testing.assert_array_equal(
        base, encoders.sax_symbols(3.0 * x + 7.0, segments=8, alphabet=5)
    )


def test_sax_quantile_variant_is_not_the_standard_one() -> None:
    """The legacy variant rescales to the segment means, standard SAX does not.

    Here a large oscillation dominates the series variance while the segment
    means differ only by ~1e-3. Standard SAX therefore assigns every segment
    the same symbol; the quantile variant re-spreads them over the alphabet
    regardless of how small the differences are.
    """

    oscillation = np.tile([-1.0, 1.0], 32)
    x = oscillation + np.linspace(0.0, 1e-3, 64)
    gaussian = encoders.sax_symbols(x, segments=8, alphabet=3)
    quantile = encoders.sax_symbols(x, segments=8, alphabet=3, breakpoints="quantile")
    assert np.unique(gaussian).size == 1  # every segment mean is within +-0.43 sigma
    assert np.unique(quantile).size == 3


def test_sax_rejects_more_segments_than_samples() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        encoders.sax(np.arange(8.0), segments=9)
    with pytest.raises(ValueError, match="alphabet"):
        encoders.sax(np.arange(8.0), segments=4, alphabet=1)


# ---------------------------------------------------------------------------
# Visibility graph


def test_visibility_graph_matches_definition() -> None:
    """Lacasa's criterion, evaluated by brute force over all intermediate points."""

    x = _series(20, seed=9)
    n = x.size
    expected = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            visible = True
            for k in range(i + 1, j):
                bound = x[j] + (x[i] - x[j]) * (j - k) / (j - i)
                if x[k] >= bound:
                    visible = False
                    break
            if visible:
                expected[i, j] = expected[j, i] = 1.0
    np.testing.assert_array_equal(encoders.visibility_graph(x), expected)


# ---------------------------------------------------------------------------
# Matrix profile


def test_matrix_profile_matches_brute_force() -> None:
    """Closest z-normalised Euclidean neighbour outside the exclusion zone."""

    x = _series(40, seed=10)
    m = 6
    n_sub = x.size - m + 1
    excl = m // 2
    windows = np.array([x[i : i + m] for i in range(n_sub)])
    z = np.array([(w - w.mean()) / (w.std() + 1e-12) for w in windows])
    expected = np.empty(n_sub)
    for i in range(n_sub):
        best = np.inf
        for j in range(n_sub):
            if abs(i - j) <= excl:
                continue
            best = min(best, float(np.linalg.norm(z[i] - z[j])))
        expected[i] = best
    np.testing.assert_allclose(
        encoders.matrix_profile(x, m=m, normalize=False), expected, atol=1e-8
    )
    np.testing.assert_allclose(
        encoders.matrix_profile(x, m=m), expected / expected.max(), atol=1e-8
    )


def test_matrix_profile_rejects_degenerate_lengths() -> None:
    """Previously these returned all-nan instead of raising."""

    x = _series(16, seed=11)
    with pytest.raises(ValueError, match="too short"):
        encoders.matrix_profile(x, m=x.size)
    with pytest.raises(ValueError, match="too short"):
        encoders.matrix_profile(x, m=12)
    prof = encoders.matrix_profile(x, m=12, exclusion=0)
    assert np.all(np.isfinite(prof))


def test_matrix_profile_finds_a_planted_motif() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=200)
    motif = rng.normal(size=20)
    x[20:40] = motif
    x[120:140] = motif
    prof = encoders.matrix_profile(x, m=20, normalize=False)
    # The planted repeats are each other's nearest neighbour: distance ~ 0.
    assert prof[20] < 1e-6
    assert prof[120] < 1e-6
    assert int(np.argmin(prof)) in {20, 120}


# ---------------------------------------------------------------------------
# Window attention


def test_window_attention_matches_softmax_definition() -> None:
    x = _series(20, seed=13)
    w = 5
    wins = np.array([x[i : i + w] for i in range(x.size - w + 1)])
    scores = wins @ wins.T / math.sqrt(w)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    expected = exp / exp.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(encoders.window_attention(x, window=w), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Persistence


def _brute_force_persistence(x: np.ndarray) -> np.ndarray:
    """Sublevel-set 0D persistence by explicit component bookkeeping."""

    n = x.size
    order = sorted(range(n), key=lambda i: (x[i], i))
    comps: dict[int, float] = {}  # representative index -> birth value
    member: dict[int, int] = {}  # point index -> representative
    pairs: list[tuple[float, float]] = []
    for i in order:
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < n and j in member]
        roots = {member[j] for j in neighbours}
        if not roots:
            comps[i] = float(x[i])
            member[i] = i
            continue
        roots_sorted = sorted(roots, key=lambda r: comps[r])
        survivor = roots_sorted[0]
        for dying in roots_sorted[1:]:
            if comps[dying] < x[i]:
                pairs.append((comps[dying], float(x[i])))
            for point, rep in list(member.items()):
                if rep == dying:
                    member[point] = survivor
        member[i] = survivor
    if not pairs:
        return np.zeros((0, 2))
    arr = np.array(pairs, dtype=float)
    return arr[np.lexsort((arr[:, 1], arr[:, 0]))]


def test_persistence_diagram_matches_brute_force() -> None:
    for seed in range(6):
        x = _series(int(np.random.default_rng(seed).integers(10, 60)), seed=seed)
        got = encoders.persistence_diagram(x)
        got = got[np.lexsort((got[:, 1], got[:, 0]))]
        np.testing.assert_allclose(got, _brute_force_persistence(x), atol=1e-12)


def test_persistence_image_is_the_integral_of_the_weighted_surface() -> None:
    """Total pixel mass equals sum of weights when the grid covers the diagram."""

    x = np.sin(np.linspace(0, 12 * np.pi, 200))
    dgm = encoders.persistence_diagram(x)
    pers = dgm[:, 1] - dgm[:, 0]
    wide_b = (dgm[:, 0].min() - 5.0, dgm[:, 0].max() + 5.0)
    wide_p = (pers.min() - 5.0, pers.max() + 5.0)
    img = encoders.persistence_image(
        x, bins=64, sigma=0.05, birth_range=wide_b, pers_range=wide_p
    )
    np.testing.assert_allclose(img.sum(), pers.sum(), rtol=1e-6)


def test_persistence_image_is_stable_under_small_perturbation() -> None:
    """Stability is the point of the representation (Adams et al., Thm. 10)."""

    rng = np.random.default_rng(14)
    x = np.sin(np.linspace(0, 10 * np.pi, 128))
    b_range, p_range = (-1.5, 1.5), (0.0, 2.5)
    base = encoders.persistence_image(x, bins=16, birth_range=b_range, pers_range=p_range)
    noisy = encoders.persistence_image(
        x + rng.normal(scale=1e-4, size=x.size),
        bins=16,
        birth_range=b_range,
        pers_range=p_range,
    )
    assert np.abs(base - noisy).max() < 1e-2


def test_persistence_image_weight_variants() -> None:
    x = np.sin(np.linspace(0, 12 * np.pi, 128))
    kwargs = {"bins": 8, "birth_range": (-1.5, 1.5), "pers_range": (0.0, 2.5)}
    persistence = encoders.persistence_image(x, weight="persistence", **kwargs)
    ramp = encoders.persistence_image(x, weight="ramp", **kwargs)
    uniform = encoders.persistence_image(x, weight="uniform", **kwargs)
    # Same support, different mass: w=p >> w=min(p/pmax,1) here since p_max ~ 2.
    assert persistence.sum() > ramp.sum()
    assert uniform.sum() > ramp.sum()
    for img in (persistence, ramp, uniform):
        assert np.all(img >= 0.0)


# ---------------------------------------------------------------------------
# Spectrogram


def test_spectrogram_matches_framed_rfft() -> None:
    x = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 128))
    win, hop = 32, 16
    spec = encoders.spectrogram(x, win=win, hop=hop, window="rect")
    n_frames = spec.shape[1]
    expected = np.empty((win // 2 + 1, n_frames))
    padded = np.pad(x, (0, max(0, (n_frames - 1) * hop + win - x.size)))
    for f in range(n_frames):
        frame = padded[f * hop : f * hop + win]
        expected[:, f] = np.abs(np.fft.rfft(frame))
    expected = expected / (expected.max() + 1e-12)
    np.testing.assert_allclose(spec, expected, atol=1e-10)


def test_spectrogram_locates_a_pure_tone() -> None:
    fs, freq, win = 256, 32.0, 64
    t = np.arange(fs) / fs
    spec = encoders.spectrogram(np.sin(2 * np.pi * freq * t), win=win, hop=win)
    # bin index = freq * win / fs
    assert int(np.argmax(spec[:, 0])) == int(round(freq * win / fs))
