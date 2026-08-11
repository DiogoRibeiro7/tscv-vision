"""Validation of squared wavelet coherence.

Coherence has a hard numerical bound and a famous degeneracy, so the tests
check both, then verify it recovers coupling that is localised in time and a
known phase lead.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscv_vision.multivariate import WaveletCoherenceResult, wavelet_coherence

FS = 51.2
N = 1024


@pytest.fixture
def time() -> np.ndarray:
    return np.arange(N) / FS


def band_index(frequencies: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(frequencies - target)))


# ---------------------------------------------------------------------------
# Bounds


def test_coherence_is_bounded(time: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    for a, b in [
        (np.sin(2 * np.pi * 3 * time), np.sin(2 * np.pi * 3 * time + 0.7)),
        (rng.normal(size=N), rng.normal(size=N)),
        (np.zeros(N), rng.normal(size=N)),
        (np.sin(2 * np.pi * 3 * time), np.sin(2 * np.pi * 11 * time)),
    ]:
        coherence = wavelet_coherence(a, b, fs=FS)
        assert np.all(coherence >= 0.0)
        assert np.all(coherence <= 1.0)
        assert np.all(np.isfinite(coherence))


def test_a_signal_is_perfectly_coherent_with_itself(time: np.ndarray) -> None:
    coherence = wavelet_coherence(
        np.sin(2 * np.pi * 3 * time), np.sin(2 * np.pi * 3 * time), fs=FS
    )
    np.testing.assert_allclose(coherence, 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# The degeneracy without smoothing


def test_unsmoothed_coherence_is_identically_one(time: np.ndarray) -> None:
    """The well-known collapse: |W_x conj(W_y)|^2 / (|W_x|^2 |W_y|^2) == 1.

    Documented as a warning on the function, and asserted here so nobody
    mistakes `smoothing=False` for a faster approximation.
    """

    rng = np.random.default_rng(1)
    for a, b in [
        (np.sin(2 * np.pi * 3 * time), np.sin(2 * np.pi * 3 * time + 0.7)),
        (rng.normal(size=N), rng.normal(size=N)),
        (np.sin(2 * np.pi * 3 * time), rng.normal(size=N)),
    ]:
        unsmoothed = wavelet_coherence(a, b, fs=FS, smoothing=False)
        np.testing.assert_allclose(unsmoothed, 1.0, atol=1e-9)


def test_smoothing_is_what_makes_it_discriminate(time: np.ndarray) -> None:
    rng = np.random.default_rng(2)
    unrelated = wavelet_coherence(rng.normal(size=N), rng.normal(size=N), fs=FS)
    assert unrelated.mean() < 0.9, "smoothed coherence should not saturate"


# ---------------------------------------------------------------------------
# It detects coupling


def test_coupled_signals_are_more_coherent_than_independent_noise(
    time: np.ndarray,
) -> None:
    rng = np.random.default_rng(3)
    coupled = wavelet_coherence(
        np.sin(2 * np.pi * 3 * time), np.sin(2 * np.pi * 3 * time + 0.7), fs=FS
    )
    unrelated = wavelet_coherence(rng.normal(size=N), rng.normal(size=N), fs=FS)
    assert coupled.mean() > 2.0 * unrelated.mean()


def test_coupling_is_localised_at_the_shared_frequency(time: np.ndarray) -> None:
    result = wavelet_coherence(
        np.sin(2 * np.pi * 3 * time),
        np.sin(2 * np.pi * 3 * time + 0.7),
        fs=FS,
        return_phase=True,
    )
    assert isinstance(result, WaveletCoherenceResult)
    shared = band_index(result.frequencies, 3.0)
    interior = slice(200, -200)
    assert result.coherence[shared, interior].mean() > 0.95


def test_coupling_localised_in_time_shows_up_in_time(time: np.ndarray) -> None:
    """Coupled for the first half, independent for the second."""

    rng = np.random.default_rng(4)
    shared = np.sin(2 * np.pi * 3 * time)
    half = N // 2
    first = np.concatenate([shared[:half], rng.normal(size=N - half)])
    second = np.concatenate([0.8 * shared[:half], rng.normal(size=N - half)])

    result = wavelet_coherence(first, second, fs=FS, return_phase=True)
    assert isinstance(result, WaveletCoherenceResult)
    row = result.coherence[band_index(result.frequencies, 3.0)]
    coupled_half = row[100 : half - 100].mean()
    uncoupled_half = row[half + 100 : -100].mean()
    assert coupled_half > 0.9
    assert uncoupled_half < 0.75
    assert coupled_half > uncoupled_half + 0.2


# ---------------------------------------------------------------------------
# Phase


def test_phase_recovers_a_known_lead(time: np.ndarray) -> None:
    lead = 0.9
    result = wavelet_coherence(
        np.sin(2 * np.pi * 3 * time),
        np.sin(2 * np.pi * 3 * time - lead),
        fs=FS,
        return_phase=True,
    )
    assert isinstance(result, WaveletCoherenceResult)
    row = result.phase[band_index(result.frequencies, 3.0)]
    assert float(np.median(row[200:-200])) == pytest.approx(lead, abs=0.05)


def test_swapping_the_arguments_negates_the_phase(time: np.ndarray) -> None:
    a = np.sin(2 * np.pi * 3 * time)
    b = np.sin(2 * np.pi * 3 * time - 0.9)
    forward = wavelet_coherence(a, b, fs=FS, return_phase=True)
    backward = wavelet_coherence(b, a, fs=FS, return_phase=True)
    assert isinstance(forward, WaveletCoherenceResult)
    assert isinstance(backward, WaveletCoherenceResult)
    np.testing.assert_allclose(forward.coherence, backward.coherence, atol=1e-12)
    np.testing.assert_allclose(forward.phase, -backward.phase, atol=1e-12)


# ---------------------------------------------------------------------------
# The structured result


def test_return_type_follows_the_argument_not_the_data(time: np.ndarray) -> None:
    a = np.sin(2 * np.pi * 3 * time)
    plain = wavelet_coherence(a, a, fs=FS)
    structured = wavelet_coherence(a, a, fs=FS, return_phase=True)
    assert isinstance(plain, np.ndarray)
    assert isinstance(structured, WaveletCoherenceResult)
    np.testing.assert_array_equal(plain, structured.coherence)


def test_structured_result_axes_are_consistent(time: np.ndarray) -> None:
    result = wavelet_coherence(
        np.sin(2 * np.pi * 3 * time), np.sin(2 * np.pi * 5 * time), fs=FS,
        return_phase=True,
    )
    assert isinstance(result, WaveletCoherenceResult)
    n_scales = result.coherence.shape[0]
    assert result.phase.shape == result.coherence.shape
    assert result.scales.shape == (n_scales,)
    assert result.frequencies.shape == (n_scales,)
    assert result.coherence.shape[1] == N
    # Larger scale means lower frequency, monotonically.
    assert np.all(np.diff(result.scales) > 0)
    assert np.all(np.diff(result.frequencies) < 0)


def test_frequency_axis_is_calibrated() -> None:
    """Band-limited coupling raises coherence in *its own* band.

    Stated against neighbouring bands rather than as a global argmax. Two
    pure tones would not test the axis at all — they are perfectly coherent
    wherever both have any energy, since coherence measures consistency, not
    strength — so the shared tone is buried in independent noise. The
    comparison is local because coherence is biased upward at the largest
    scales, where the smoothing window spans much of the record; that is a
    property of the estimator, not of the coupling.
    """

    length = 2048
    grid = np.arange(length) / FS
    rng = np.random.default_rng(5)
    shared = np.sin(2 * np.pi * 5 * grid)
    result = wavelet_coherence(
        shared + 2.0 * rng.normal(size=length),
        shared + 2.0 * rng.normal(size=length),
        fs=FS,
        return_phase=True,
    )
    assert isinstance(result, WaveletCoherenceResult)
    interior = result.coherence[:, 300:-300].mean(axis=1)

    at_shared = interior[band_index(result.frequencies, 5.0)]
    for neighbour in (3.0, 8.0, 12.0):
        assert at_shared > interior[band_index(result.frequencies, neighbour)] + 0.1, (
            f"5 Hz coupling did not stand out against the {neighbour} Hz band"
        )


# ---------------------------------------------------------------------------
# Invariances


def test_invariant_to_rescaling_either_series() -> None:
    """Both auto-spectra divide out, so gain cancels exactly.

    Tested on broadband signals. A pair of pure tones leaves most bands with
    around 1e-60 of energy, where the ratio of two vanishing quantities is
    numerically ill-conditioned and rescaling perturbs it by a few percent —
    a property of the estimator at empty scales, not a failure of the
    invariance, and the reason to read coherence only where there is power.
    """

    rng = np.random.default_rng(6)
    a = rng.normal(size=N)
    b = 0.6 * a + 0.8 * rng.normal(size=N)
    base = wavelet_coherence(a, b, fs=FS)
    np.testing.assert_allclose(wavelet_coherence(100.0 * a, b, fs=FS), base, atol=1e-12)
    np.testing.assert_allclose(wavelet_coherence(a, 0.01 * b, fs=FS), base, atol=1e-12)


def test_deterministic(time: np.ndarray) -> None:
    a = np.sin(2 * np.pi * 3 * time)
    b = np.sin(2 * np.pi * 3 * time + 0.7)
    np.testing.assert_array_equal(
        wavelet_coherence(a, b, fs=FS), wavelet_coherence(a, b, fs=FS)
    )


# ---------------------------------------------------------------------------
# Validation


def test_unequal_lengths_are_rejected(time: np.ndarray) -> None:
    with pytest.raises(ValueError, match="same length"):
        wavelet_coherence(np.sin(time), np.sin(time[:100]), fs=FS)


def test_invalid_arguments(time: np.ndarray) -> None:
    a = np.sin(2 * np.pi * 3 * time)
    for kwargs, match in [
        ({"fs": 0.0}, "fs must be"),
        ({"fs": -3.0}, "fs must be"),
        ({"voices": 0}, "voices"),
        ({"wavelet": "mexh"}, "only the Morlet"),
        ({"scales": np.array([])}, "scales"),
        ({"scales": np.array([1.0, -1.0])}, "scales"),
    ]:
        with pytest.raises(ValueError, match=match):
            wavelet_coherence(a, a, **kwargs)  # type: ignore[arg-type]


def test_custom_scales(time: np.ndarray) -> None:
    a = np.sin(2 * np.pi * 3 * time)
    coherence = wavelet_coherence(a, a, fs=FS, scales=np.geomspace(0.02, 1.0, 12))
    assert coherence.shape == (12, N)


def test_nan_policy(time: np.ndarray) -> None:
    a = np.sin(2 * np.pi * 3 * time)
    dirty = a.copy()
    dirty[50] = np.nan
    with pytest.raises(ValueError):
        wavelet_coherence(dirty, a, fs=FS)
    assert np.all(
        np.isfinite(wavelet_coherence(dirty, a, fs=FS, nan_policy="interpolate"))
    )


def test_metadata_is_recorded() -> None:
    from tscv_vision.representations import get_encoder_metadata, list_encoders

    info = get_encoder_metadata("wavelet_coherence")
    assert info.canonical_method is True
    assert info.input_kind == "bivariate"
    assert "Torrence" in (info.reference or "")
    assert "wavelet_coherence" not in list_encoders()
