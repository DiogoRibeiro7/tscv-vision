"""Numerical equivalence against authoritative third-party implementations.

Every test here is skipped unless its reference package is installed, and the
whole module is marked ``optional`` so it runs in the dedicated CI job (see
``.github/workflows``) rather than the default suite. Together with
``tests/test_encoder_definitions.py`` this gives each scientific routine either
a reference-equivalence check or an explicit "this is a new/approximate
variant" label in its docstring.

References
----------
``scikit-image`` for LBP, ``scipy.stats`` for the distribution functions and
tests, ``ripser`` for sublevel-set persistent homology, ``persim`` for
persistence images, ``pyts`` for the imaging transforms.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscv_vision import encoders, features, stats

pytestmark = pytest.mark.optional


# ---------------------------------------------------------------------------
# Local Binary Patterns vs scikit-image


@pytest.fixture
def image() -> np.ndarray:
    return np.asarray(np.random.default_rng(0).random((32, 32)), dtype=float)


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_lbp_codes_match_skimage(image: np.ndarray, radius: int) -> None:
    """LBP_{8,R} codes must equal skimage's, away from the padded border."""

    skfeature = pytest.importorskip("skimage.feature")
    ours = features._lbp_codes(image, radius)
    ref = skfeature.local_binary_pattern(image, 8, radius, method="default").astype(int)
    inner = slice(radius, -radius)
    np.testing.assert_array_equal(ours[inner, inner], ref[inner, inner])


def test_lbp_rotation_invariant_matches_skimage(image: np.ndarray) -> None:
    skfeature = pytest.importorskip("skimage.feature")
    ours = features._LBP_RI_MAP[features._lbp_codes(image, 2)]
    ref = skfeature.local_binary_pattern(image, 8, 2, method="ror").astype(int)
    np.testing.assert_array_equal(ours[2:-2, 2:-2], ref[2:-2, 2:-2])


def test_lbp_uniform_partition_matches_skimage(image: np.ndarray) -> None:
    """Same 59-bin partition as skimage's ``nri_uniform`` (labels may differ)."""

    skfeature = pytest.importorskip("skimage.feature")
    ours = features._LBP_UNI_MAP[features._lbp_codes(image, 1)][1:-1, 1:-1]
    ref = skfeature.local_binary_pattern(image, 8, 1, method="nri_uniform")
    ref = ref.astype(int)[1:-1, 1:-1]
    assert features._LBP_UNI_BINS == 59
    ours_hist = np.bincount(ours.ravel(), minlength=59)
    ref_hist = np.bincount(ref.ravel(), minlength=59)
    np.testing.assert_array_equal(np.sort(ours_hist), np.sort(ref_hist))
    # Both put the non-uniform patterns in the final bin.
    assert ours_hist[58] == ref_hist[58]
    assert ours_hist[58] > 0


def test_lbp_radius_changes_the_histogram(image: np.ndarray) -> None:
    """Regression: radius used to change only the padding, not the sampling."""

    assert not np.allclose(features.lbp(image, radius=1), features.lbp(image, radius=2))
    assert not np.array_equal(
        features._lbp_codes(image, 1), features._lbp_codes(image, 2)
    )


def test_lbp_uniform_histogram_accounts_for_every_pixel(image: np.ndarray) -> None:
    """Regression: non-uniform patterns fell outside the histogram range."""

    hist = features.lbp_uniform(image)
    assert hist.shape == (59,)
    assert hist.sum() == pytest.approx(1.0)  # bin width is 1
    assert hist[-1] > 0.0


# ---------------------------------------------------------------------------
# Statistics vs scipy.stats


def test_special_functions_match_scipy() -> None:
    sps = pytest.importorskip("scipy.special")
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = rng.uniform(0.2, 20.0, size=2)
        x = float(rng.uniform(0.0, 1.0))
        assert stats.betainc(a, b, x) == pytest.approx(sps.betainc(a, b, x), rel=1e-10)
    for _ in range(20):
        a = float(rng.uniform(0.2, 30.0))
        x = float(rng.uniform(0.0, 50.0))
        assert stats.gammainc_upper(a, x) == pytest.approx(sps.gammaincc(a, x), rel=1e-9)


def test_distribution_tails_match_scipy() -> None:
    scistats = pytest.importorskip("scipy.stats")
    for t, df in [(0.1, 1.0), (2.5, 3.0), (-1.7, 12.0), (4.0, 250.0)]:
        assert stats.student_t_sf(t, df) == pytest.approx(scistats.t.sf(t, df), rel=1e-9)
    for x, df in [(0.5, 1.0), (3.2, 2.0), (7.3, 3.0), (25.0, 10.0)]:
        assert stats.chi2_sf(x, df) == pytest.approx(scistats.chi2.sf(x, df), rel=1e-9)
    for z in (-3.0, -0.5, 0.0, 1.2, 4.0):
        assert stats.normal_sf(z) == pytest.approx(scistats.norm.sf(z), rel=1e-12)


def test_welch_ttest_matches_scipy() -> None:
    """Regression: the p-value used to come from the normal distribution."""

    scistats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(2)
    for _ in range(25):
        a = rng.normal(0.0, 1.0, size=int(rng.integers(3, 40)))
        b = rng.normal(0.4, 2.0, size=int(rng.integers(3, 40)))
        got = stats.welch_ttest(a, b)
        ref = scistats.ttest_ind(a, b, equal_var=False)
        assert got.statistic == pytest.approx(ref.statistic, rel=1e-10)
        assert got.pvalue == pytest.approx(ref.pvalue, rel=1e-9)
        assert got.df == pytest.approx(ref.df, rel=1e-10)


def test_welch_ttest_small_sample_differs_from_normal_approximation() -> None:
    """The old z-approximation was anti-conservative for small samples."""

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 3.5, 5.0, 6.5])
    got = stats.welch_ttest(a, b)
    normal_p = 2.0 * stats.normal_sf(abs(got.statistic))
    assert got.pvalue > normal_p * 1.2


def test_wilcoxon_matches_scipy() -> None:
    scistats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    for n in (6, 10, 20, 40):
        x = rng.normal(size=n)
        y = rng.normal(0.5, size=n)
        got = stats.wilcoxon_signed_rank(x, y)
        ref = scistats.wilcoxon(x, y, method="exact" if n <= 25 else "approx")
        assert got.statistic == pytest.approx(ref.statistic)
        assert got.pvalue == pytest.approx(ref.pvalue, rel=1e-9)


def test_friedman_matches_scipy() -> None:
    scistats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(4)
    scores = rng.random((15, 4))
    got = stats.friedman_test(scores)
    ref = scistats.friedmanchisquare(*[-scores[:, j] for j in range(scores.shape[1])])
    assert got.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert got.pvalue == pytest.approx(ref.pvalue, rel=1e-9)
    assert got.ranks.shape == (4,)
    assert got.ranks.mean() == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Persistent homology vs ripser / persim


def _ripser_lower_star(x: np.ndarray) -> np.ndarray:
    """0D sublevel-set diagram via ripser's sparse-distance-matrix recipe."""

    sparse = pytest.importorskip("scipy.sparse")
    ripser_mod = pytest.importorskip("ripser")
    n = x.size
    rows = np.concatenate([np.arange(n), np.arange(n - 1)])
    cols = np.concatenate([np.arange(n), np.arange(1, n)])
    vals = np.concatenate([x, np.maximum(x[:-1], x[1:])])
    mat = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    dgm = ripser_mod.ripser(mat, maxdim=0, distance_matrix=True)["dgms"][0]
    finite = dgm[np.isfinite(dgm[:, 1])]
    finite = finite[finite[:, 1] > finite[:, 0]]
    return np.asarray(finite[np.lexsort((finite[:, 1], finite[:, 0]))])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_persistence_diagram_matches_ripser(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=int(rng.integers(30, 120)))
    got = encoders.persistence_diagram(x)
    got = got[np.lexsort((got[:, 1], got[:, 0]))]
    np.testing.assert_allclose(got, _ripser_lower_star(x), atol=1e-12)


def test_persistence_diagram_of_a_sine_matches_ripser() -> None:
    x = np.sin(np.linspace(0, 8 * np.pi, 200))
    got = encoders.persistence_diagram(x)
    got = got[np.lexsort((got[:, 1], got[:, 0]))]
    np.testing.assert_allclose(got, _ripser_lower_star(x), atol=1e-12)


def test_persistence_image_matches_persim() -> None:
    """Pixel-for-pixel equality with persim's Gaussian persistence imager."""

    persim = pytest.importorskip("persim")
    x = np.asarray(np.random.default_rng(5).normal(size=80))
    diagram = encoders.persistence_diagram(x)
    bins = 8
    birth_range, pers_range = (-2.0, 2.0), (0.0, 4.0)  # equal widths: square pixels
    sigma = (pers_range[1] - pers_range[0]) / bins
    imager = persim.PersistenceImager(
        birth_range=birth_range,
        pers_range=pers_range,
        pixel_size=(birth_range[1] - birth_range[0]) / bins,
        kernel="gaussian",
        kernel_params={"sigma": [[sigma**2, 0.0], [0.0, sigma**2]]},
        weight="persistence",
        weight_params={"n": 1},
    )
    ref = np.asarray(imager.transform(diagram))
    ours = encoders.persistence_image(
        x, bins=bins, sigma=sigma, birth_range=birth_range, pers_range=pers_range
    )
    # Ours is indexed [persistence, birth]; persim uses [birth, persistence].
    np.testing.assert_allclose(ours, ref.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Imaging transforms vs pyts


def test_gaf_matches_pyts() -> None:
    """Equal to pyts up to the conditioning of ``arccos`` at the endpoints.

    Both implementations min-max scale to ``[-1, 1]`` and take ``arccos``, but
    they round the scaling differently (pyts routes it through sklearn's
    ``MinMaxScaler``). Since ``d/dz arccos(z) = -1/sqrt(1-z^2)`` diverges at the
    series minimum and maximum, a 1e-16 discrepancy in the scaled value shows
    up as ~1e-8 in those two rows/columns. Everything else agrees to 1e-15.
    """

    pyts_image = pytest.importorskip("pyts.image")
    rng = np.random.default_rng(6)
    for n in (32, 128):
        x = np.asarray(rng.normal(size=n))
        for method, name in (("summation", "s"), ("difference", "d")):
            ref = pyts_image.GramianAngularField(method=name).fit_transform(x[None, :])[0]
            np.testing.assert_allclose(encoders.gaf(x, method=method), ref, atol=1e-6)


def test_mtf_matches_pyts() -> None:
    pyts_image = pytest.importorskip("pyts.image")
    x = np.asarray(np.random.default_rng(7).normal(size=48))
    ref = pyts_image.MarkovTransitionField(
        n_bins=4, strategy="quantile"
    ).fit_transform(x[None, :])[0]
    np.testing.assert_allclose(encoders.mtf(x, bins=4), ref, atol=1e-10)


def test_sax_breakpoints_match_scipy_norm_ppf() -> None:
    scistats = pytest.importorskip("scipy.stats")
    for alphabet in range(2, 21):
        expected = scistats.norm.ppf(np.arange(1, alphabet) / alphabet)
        np.testing.assert_allclose(
            encoders._gaussian_breakpoints(alphabet), expected, atol=1e-9
        )


def test_matrix_profile_matches_stumpy() -> None:
    stumpy = pytest.importorskip("stumpy")
    rng = np.random.default_rng(8)
    x = rng.normal(size=200)
    m = 20
    ref = np.asarray(stumpy.stump(x, m)[:, 0], dtype=float)
    ours = encoders.matrix_profile(x, m=m, exclusion=int(np.ceil(m / 4)), normalize=False)
    np.testing.assert_allclose(ours, ref, atol=1e-6)
