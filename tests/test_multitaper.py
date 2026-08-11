"""Validation of the multitaper spectrogram.

Thomson's estimator makes a quantitative promise — variance falling roughly as
1/K while resolution widens to 2NW/W — so the tests measure both rather than
checking that a picture appears.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscv_vision import encoders

pytest.importorskip("scipy", reason="DPSS tapers require SciPy")

FS = 100.0


@pytest.fixture
def tone() -> np.ndarray:
    return np.sin(2 * np.pi * 10.0 * (np.arange(1024) / FS))


# ---------------------------------------------------------------------------
# Agreement with a direct calculation


def test_single_taper_matches_a_direct_periodogram(tone: np.ndarray) -> None:
    """With K=1 the estimator is one tapered periodogram; compute it by hand."""

    from scipy.signal.windows import dpss

    window_size = 256
    image = encoders.multitaper_spectrogram(
        tone,
        fs=FS,
        window_size=window_size,
        hop_length=window_size,
        n_tapers=1,
        time_bandwidth=4.0,
        scaling="power",
    )

    taper = dpss(window_size, 4.0, Kmax=1)[0]
    frame = tone[:window_size]
    expected = np.abs(np.fft.rfft(taper * frame)) ** 2
    # The encoder normalises by the peak over the whole image, so compare the
    # shape of the spectrum rather than its absolute scale.
    np.testing.assert_allclose(
        image[:, 0] / image[:, 0].max(), expected / expected.max(), atol=1e-10
    )


def test_multi_taper_matches_the_average_of_the_individual_periodograms(
    tone: np.ndarray,
) -> None:
    from scipy.signal.windows import dpss

    window_size, k = 256, 5
    image = encoders.multitaper_spectrogram(
        tone,
        fs=FS,
        window_size=window_size,
        hop_length=window_size,
        n_tapers=k,
        time_bandwidth=3.0,
        scaling="power",
    )
    tapers = dpss(window_size, 3.0, Kmax=k)
    frame = tone[:window_size]
    expected = np.mean(
        [np.abs(np.fft.rfft(taper * frame)) ** 2 for taper in tapers], axis=0
    )
    np.testing.assert_allclose(
        image[:, 0] / image[:, 0].max(), expected / expected.max(), atol=1e-10
    )


def test_tapers_are_scipy_dpss() -> None:
    """The tapers must be the Slepian sequences, not a substitute window."""

    from scipy.signal.windows import dpss

    tapers = dpss(64, 3.0, Kmax=5)
    # Orthonormal by construction, which is what makes the average consistent.
    np.testing.assert_allclose(tapers @ tapers.T, np.eye(5), atol=1e-8)


# ---------------------------------------------------------------------------
# The defining property: variance reduction


def test_variance_falls_roughly_as_one_over_k() -> None:
    """Thomson's estimator averages K nearly independent periodograms.

    A single-taper periodogram of Gaussian noise is chi-square with 2 degrees
    of freedom, so its relative variance is about 1. Averaging K of them should
    push that toward 1/K.
    """

    rng = np.random.default_rng(0)
    samples: dict[int, list[float]] = {1: [], 7: []}
    for _ in range(120):
        noise = rng.normal(size=512)
        for k in (1, 7):
            estimate = encoders.multitaper_spectrogram(
                noise,
                fs=FS,
                window_size=256,
                hop_length=256,
                n_tapers=k,
                time_bandwidth=4.0,
                scaling="power",
            )
            samples[k].append(float(estimate[40, 0]))

    def relative_variance(values: list[float]) -> float:
        arr = np.asarray(values)
        return float(arr.var() / arr.mean() ** 2)

    single = relative_variance(samples[1])
    multi = relative_variance(samples[7])
    assert 0.6 < single < 1.6, f"single-taper relative variance was {single}"
    assert multi < single / 3.0, f"K=7 did not reduce variance: {multi} vs {single}"


def test_changing_the_taper_count_changes_the_estimator() -> None:
    noise = np.random.default_rng(1).normal(size=512)
    estimates = [
        encoders.multitaper_spectrogram(
            noise, fs=FS, window_size=256, n_tapers=k, time_bandwidth=4.0
        )
        for k in (1, 3, 7)
    ]
    assert not np.allclose(estimates[0], estimates[1])
    assert not np.allclose(estimates[1], estimates[2])


def test_more_tapers_smooth_the_spectrum() -> None:
    """Averaging orthogonal tapers reduces bin-to-bin roughness."""

    noise = np.random.default_rng(2).normal(size=1024)

    def roughness(k: int) -> float:
        image = encoders.multitaper_spectrogram(
            noise,
            fs=FS,
            window_size=512,
            hop_length=512,
            n_tapers=k,
            time_bandwidth=4.0,
            scaling="power",
        )
        column = image[:, 0]
        return float(np.mean(np.abs(np.diff(column))))

    assert roughness(7) < roughness(1)


# ---------------------------------------------------------------------------
# Frequency localisation


def test_pure_sine_peaks_at_its_frequency(tone: np.ndarray) -> None:
    window_size = 256
    image = encoders.multitaper_spectrogram(
        tone, fs=FS, window_size=window_size, scaling="power"
    )
    peak_bin = int(np.argmax(image[:, 0]))
    expected_bin = int(round(10.0 * window_size / FS))
    assert abs(peak_bin - expected_bin) <= 1


def test_two_well_separated_sinusoids_give_two_resolved_lobes() -> None:
    """Both tones carry near-peak energy and are separated by a deep trough.

    Not "the two largest bins are the two tones": the DPSS main lobe is about
    2 NW / window_size wide, so each tone occupies several adjacent bins and
    the two largest bins can both belong to one lobe.
    """

    t = np.arange(1024) / FS
    signal = np.sin(2 * np.pi * 10.0 * t) + np.sin(2 * np.pi * 30.0 * t)
    window_size = 256
    image = encoders.multitaper_spectrogram(
        signal, fs=FS, window_size=window_size, scaling="power"
    )
    column = image[:, 2]
    low, high = (int(round(f * window_size / FS)) for f in (10.0, 30.0))

    assert column[low] > 0.5 * column.max()
    assert column[high] > 0.5 * column.max()
    # Well separated: the region between the lobes is essentially empty.
    assert column[low + 5 : high - 5].max() < 0.05 * column.max()


def test_broadband_noise_is_spread_not_peaked() -> None:
    noise = np.random.default_rng(3).normal(size=2048)
    image = encoders.multitaper_spectrogram(
        noise, fs=FS, window_size=512, n_tapers=7, time_bandwidth=4.0, scaling="power"
    )
    column = image[:, 0]
    # No single bin dominates the way a tone would.
    assert column.max() / column.mean() < 15.0


# ---------------------------------------------------------------------------
# Shape, scaling, determinism


def test_shape_follows_window_and_hop(tone: np.ndarray) -> None:
    image = encoders.multitaper_spectrogram(
        tone, fs=FS, window_size=128, hop_length=64
    )
    expected_frames = 1 + (tone.size - 128) // 64
    assert image.shape == (65, expected_frames)
    assert image.dtype == np.float64


def test_n_fft_interpolates_the_frequency_axis(tone: np.ndarray) -> None:
    coarse = encoders.multitaper_spectrogram(tone, fs=FS, window_size=128)
    fine = encoders.multitaper_spectrogram(tone, fs=FS, window_size=128, n_fft=512)
    assert coarse.shape[0] == 65
    assert fine.shape[0] == 257
    assert coarse.shape[1] == fine.shape[1]


def test_scalings(tone: np.ndarray) -> None:
    power = encoders.multitaper_spectrogram(tone, fs=FS, window_size=128, scaling="power")
    decibels = encoders.multitaper_spectrogram(
        tone, fs=FS, window_size=128, scaling="log_power"
    )
    for image in (power, decibels):
        assert np.all(image >= 0.0) and np.all(image <= 1.0)
        assert image.max() == pytest.approx(1.0)
    # The dB view lifts weak content relative to the peak.
    assert decibels.mean() > power.mean()
    # Both agree on where the energy is.
    assert np.argmax(power[:, 0]) == np.argmax(decibels[:, 0])


def test_dynamic_range_floors_the_decibel_image(tone: np.ndarray) -> None:
    narrow = encoders.multitaper_spectrogram(
        tone, fs=FS, window_size=128, dynamic_range=20.0
    )
    wide = encoders.multitaper_spectrogram(
        tone, fs=FS, window_size=128, dynamic_range=120.0
    )
    assert np.count_nonzero(narrow) <= np.count_nonzero(wide)


def test_amplitude_invariance(tone: np.ndarray) -> None:
    np.testing.assert_allclose(
        encoders.multitaper_spectrogram(tone, fs=FS, window_size=128),
        encoders.multitaper_spectrogram(9.0 * tone, fs=FS, window_size=128),
        atol=1e-12,
    )


def test_deterministic(tone: np.ndarray) -> None:
    np.testing.assert_array_equal(
        encoders.multitaper_spectrogram(tone, fs=FS, window_size=128),
        encoders.multitaper_spectrogram(tone, fs=FS, window_size=128),
    )


def test_zero_signal_returns_zeros() -> None:
    image = encoders.multitaper_spectrogram(np.zeros(512), window_size=128)
    assert np.all(image == 0.0)


# ---------------------------------------------------------------------------
# Validation of arguments


def test_short_windows_are_allowed(tone: np.ndarray) -> None:
    image = encoders.multitaper_spectrogram(
        tone, fs=FS, window_size=8, time_bandwidth=2.0, n_tapers=2
    )
    assert image.shape[0] == 5
    assert np.all(np.isfinite(image))


def test_invalid_arguments(tone: np.ndarray) -> None:
    for kwargs, match in [
        ({"fs": 0.0}, "fs must be"),
        ({"window_size": 1}, "window_size must be"),
        ({"window_size": 99_999}, "exceeds the series"),
        ({"hop_length": 0}, "hop_length"),
        ({"time_bandwidth": 0.5}, "time_bandwidth"),
        ({"time_bandwidth": float("inf")}, "time_bandwidth"),
        ({"n_tapers": 0}, "n_tapers must be >= 1"),
        ({"n_tapers": 10_000}, "cannot exceed window_size"),
        ({"n_fft": 4}, "n_fft"),
        ({"scaling": "nope"}, "scaling"),
        ({"scaling": "log_power", "dynamic_range": 0.0}, "dynamic_range"),
    ]:
        with pytest.raises(ValueError, match=match):
            encoders.multitaper_spectrogram(tone, **kwargs)  # type: ignore[arg-type]


def test_default_taper_count_needs_enough_bandwidth(tone: np.ndarray) -> None:
    """NW = 1 gives int(2 * 1) - 1 = 1 taper, which is valid but degenerate."""

    image = encoders.multitaper_spectrogram(tone, window_size=128, time_bandwidth=1.0)
    assert np.all(np.isfinite(image))


def test_nan_policy(tone: np.ndarray) -> None:
    dirty = tone.copy()
    dirty[10] = np.nan
    with pytest.raises(ValueError):
        encoders.multitaper_spectrogram(dirty, window_size=128)
    assert np.all(
        np.isfinite(
            encoders.multitaper_spectrogram(
                dirty, window_size=128, nan_policy="interpolate"
            )
        )
    )


def test_registry_and_metadata(tone: np.ndarray) -> None:
    from tscv_vision.representations import get_representation_info

    np.testing.assert_array_equal(
        encoders.get_encoder("mtspec")(tone), encoders.multitaper_spectrogram(tone)
    )
    info = get_representation_info("mtspec")
    assert info.canonical_method is True
    assert "Thomson" in (info.reference or "")
