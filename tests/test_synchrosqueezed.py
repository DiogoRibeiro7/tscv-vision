"""Validation of the synchrosqueezed CWT against analytically known signals.

Synchrosqueezing is only worth having if the reassigned ridge really sits at
the instantaneous frequency, so the tests are built around signals whose
instantaneous frequency is known in closed form, and around the property that
distinguishes it from the plain CWT it is built on: energy concentration.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscv_vision import encoders

FS = 200.0
N = 1024


@pytest.fixture
def time() -> np.ndarray:
    return np.arange(N) / FS


def _freq_grid(n_freq: int) -> np.ndarray:
    return np.linspace(FS / N, FS / 2, n_freq)


def _ridge(image: np.ndarray, n_freq: int) -> np.ndarray:
    """Frequency of the strongest bin at each time step."""
    return _freq_grid(n_freq)[np.argmax(image, axis=0)]


def _interior(values: np.ndarray) -> np.ndarray:
    """Drop the edges, where any wavelet transform suffers boundary effects."""
    return values[N // 4 : 3 * N // 4]


# ---------------------------------------------------------------------------
# Ridge localisation


def test_constant_sinusoid_ridge_is_the_true_frequency(time: np.ndarray) -> None:
    f0 = 20.0
    n_freq = 256
    image = encoders.synchrosqueezed_cwt(
        np.sin(2 * np.pi * f0 * time), fs=FS, frequencies=n_freq
    )
    ridge = _interior(_ridge(image, n_freq))
    step = _freq_grid(n_freq)[1] - _freq_grid(n_freq)[0]
    assert np.abs(np.median(ridge) - f0) <= step
    # The ridge is a constant, not a wandering estimate.
    assert ridge.std() <= step


def test_linear_chirp_ridge_follows_the_analytic_law(time: np.ndarray) -> None:
    """x(t) = sin(2 pi (f0 t + k t^2 / 2)) has instantaneous frequency f0 + k t."""

    f0, k = 10.0, 10.0
    n_freq = 256
    signal = np.sin(2 * np.pi * (f0 * time + 0.5 * k * time**2))
    assert f0 + k * time[-1] < FS / 2, "test signal must stay below Nyquist"

    image = encoders.synchrosqueezed_cwt(signal, fs=FS, frequencies=n_freq)
    ridge = _interior(_ridge(image, n_freq))
    truth = _interior(f0 + k * time)
    step = _freq_grid(n_freq)[1] - _freq_grid(n_freq)[0]
    # Within one frequency bin everywhere, not merely on average.
    assert np.abs(ridge - truth).max() <= step


def test_two_components_are_resolved(time: np.ndarray) -> None:
    low, high = 15.0, 55.0
    n_freq = 256
    signal = np.sin(2 * np.pi * low * time) + np.sin(2 * np.pi * high * time)
    image = encoders.synchrosqueezed_cwt(signal, fs=FS, frequencies=n_freq)

    column = image[:, N // 2]
    grid = _freq_grid(n_freq)
    peaks = np.sort(grid[np.argsort(column)[-2:]])
    step = grid[1] - grid[0]
    assert abs(peaks[0] - low) <= 2 * step
    assert abs(peaks[1] - high) <= 2 * step


def test_ridge_is_unchanged_by_amplitude_scaling(time: np.ndarray) -> None:
    signal = np.sin(2 * np.pi * 25.0 * time)
    base = encoders.synchrosqueezed_cwt(signal, fs=FS, frequencies=128)
    scaled = encoders.synchrosqueezed_cwt(7.5 * signal, fs=FS, frequencies=128)
    np.testing.assert_allclose(base, scaled, atol=1e-10)


def test_time_shift_shifts_the_image(time: np.ndarray) -> None:
    signal = np.sin(2 * np.pi * (10.0 * time + 0.5 * 8.0 * time**2))
    shift = 64
    base = encoders.synchrosqueezed_cwt(signal, fs=FS, frequencies=128)
    rolled = encoders.synchrosqueezed_cwt(np.roll(signal, shift), fs=FS, frequencies=128)
    # Compare the interior only; wrapping corrupts the edges.
    inner = slice(N // 4, 3 * N // 4)
    np.testing.assert_allclose(
        np.roll(base, shift, axis=1)[:, inner], rolled[:, inner], atol=0.05
    )


# ---------------------------------------------------------------------------
# It is not a renamed CWT


def test_energy_is_more_concentrated_than_the_plain_cwt(time: np.ndarray) -> None:
    """The point of synchrosqueezing: the same energy, in fewer bins."""

    signal = np.sin(2 * np.pi * (10.0 * time + 0.5 * 10.0 * time**2))
    sst = encoders.synchrosqueezed_cwt(signal, fs=FS, frequencies=128)
    scales = np.linspace(1.0, 64.0, 128)
    cwt = encoders.cwt(signal, scales)

    def participation(image: np.ndarray) -> float:
        flat = np.abs(image).ravel()
        flat = flat / (flat.sum() + 1e-30)
        return float((flat**2).sum())  # larger = concentrated in fewer bins

    assert participation(sst) > 5 * participation(cwt)


def test_output_is_sparser_than_the_plain_cwt(time: np.ndarray) -> None:
    signal = np.sin(2 * np.pi * 30.0 * time)
    sst = encoders.synchrosqueezed_cwt(signal, fs=FS, frequencies=128)
    cwt = encoders.cwt(signal, np.linspace(1.0, 64.0, 128))
    assert np.mean(sst > 0.01) < np.mean(cwt > 0.01)


# ---------------------------------------------------------------------------
# Shape, dtype, determinism


@pytest.mark.parametrize("n_freq", [2, 16, 64, 200])
def test_shape_and_range(time: np.ndarray, n_freq: int) -> None:
    image = encoders.synchrosqueezed_cwt(
        np.sin(2 * np.pi * 20 * time), fs=FS, frequencies=n_freq
    )
    assert image.shape == (n_freq, N)
    assert image.dtype == np.float64
    assert np.all(image >= 0.0) and np.all(image <= 1.0)
    assert np.all(np.isfinite(image))


def test_deterministic(time: np.ndarray) -> None:
    signal = np.sin(2 * np.pi * 20 * time)
    first = encoders.synchrosqueezed_cwt(signal, fs=FS)
    second = encoders.synchrosqueezed_cwt(signal, fs=FS)
    np.testing.assert_array_equal(first, second)


def test_complex_output_keeps_phase(time: np.ndarray) -> None:
    signal = np.sin(2 * np.pi * 20 * time)
    complex_out = encoders.synchrosqueezed_cwt(signal, fs=FS, magnitude=False)
    assert complex_out.dtype == np.complex128
    magnitude = encoders.synchrosqueezed_cwt(signal, fs=FS, magnitude=True)
    expected = np.abs(complex_out)
    np.testing.assert_allclose(magnitude, expected / expected.max(), atol=1e-12)


def test_log_scale_preserves_the_ridge_and_lifts_weak_bins(time: np.ndarray) -> None:
    signal = np.sin(2 * np.pi * 20 * time) + 0.02 * np.sin(2 * np.pi * 60 * time)
    linear = encoders.synchrosqueezed_cwt(signal, fs=FS, frequencies=128)
    logged = encoders.synchrosqueezed_cwt(
        signal, fs=FS, frequencies=128, log_scale=True
    )
    assert np.argmax(logged[:, N // 2]) == np.argmax(linear[:, N // 2])
    assert logged.max() == pytest.approx(1.0)
    # Weak content is raised relative to the peak.
    assert logged.mean() > linear.mean()


def test_bump_wavelet_also_localises(time: np.ndarray) -> None:
    n_freq = 256
    image = encoders.synchrosqueezed_cwt(
        np.sin(2 * np.pi * 20.0 * time), fs=FS, frequencies=n_freq, wavelet="bump"
    )
    ridge = _interior(_ridge(image, n_freq))
    assert abs(np.median(ridge) - 20.0) <= 2 * (_freq_grid(n_freq)[1] - _freq_grid(n_freq)[0])


# ---------------------------------------------------------------------------
# Edge cases


def test_constant_input_produces_no_ridge() -> None:
    """A DC signal has no oscillation to reassign, so the image is empty."""

    image = encoders.synchrosqueezed_cwt(np.full(256, 3.0), fs=FS)
    assert np.all(np.isfinite(image))
    assert image.max() == 0.0


def test_very_short_input_still_returns_the_documented_shape() -> None:
    image = encoders.synchrosqueezed_cwt(np.array([0.0, 1.0, 0.0, -1.0]), fs=FS)
    assert image.shape == (64, 4)
    assert np.all(np.isfinite(image))


def test_nan_policy(time: np.ndarray) -> None:
    signal = np.sin(2 * np.pi * 20 * time)
    dirty = signal.copy()
    dirty[100] = np.nan
    with pytest.raises(ValueError):
        encoders.synchrosqueezed_cwt(dirty, fs=FS)
    filled = encoders.synchrosqueezed_cwt(dirty, fs=FS, nan_policy="interpolate")
    assert np.all(np.isfinite(filled))


def test_threshold_zero_is_accepted_and_suppression_is_monotone(time: np.ndarray) -> None:
    signal = np.sin(2 * np.pi * 20 * time)
    unfiltered = encoders.synchrosqueezed_cwt(signal, fs=FS, threshold=0.0)
    filtered = encoders.synchrosqueezed_cwt(signal, fs=FS, threshold=1e3)
    assert np.all(np.isfinite(unfiltered))
    # A threshold above every coefficient discards everything.
    assert filtered.max() == 0.0


def test_invalid_arguments(time: np.ndarray) -> None:
    signal = np.sin(2 * np.pi * 20 * time)
    for kwargs, match in [
        ({"fs": 0.0}, "fs must be"),
        ({"fs": -1.0}, "fs must be"),
        ({"fs": float("nan")}, "fs must be"),
        ({"frequencies": 1}, "frequencies"),
        ({"voices": 0}, "voices"),
        ({"threshold": -1.0}, "threshold"),
        ({"wavelet": "haar"}, "wavelet"),
        ({"scales": np.array([1.0, -2.0])}, "scales"),
        ({"scales": np.zeros((2, 2))}, "scales"),
        ({"scales": np.array([])}, "scales"),
    ]:
        with pytest.raises(ValueError, match=match):
            encoders.synchrosqueezed_cwt(signal, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        encoders.synchrosqueezed_cwt(np.array([]), fs=FS)
    with pytest.raises(ValueError):
        encoders.synchrosqueezed_cwt(np.zeros((4, 4)), fs=FS)


def test_custom_scales_are_honoured(time: np.ndarray) -> None:
    signal = np.sin(2 * np.pi * 20 * time)
    image = encoders.synchrosqueezed_cwt(
        signal, fs=FS, scales=np.geomspace(0.01, 0.2, 40), frequencies=128
    )
    assert image.shape == (128, N)
    assert np.all(np.isfinite(image))


def test_registry_and_metadata() -> None:
    from tscv_vision.representations import get_representation_info

    signal = np.sin(np.linspace(0, 40 * np.pi, 256))
    assert encoders.get_encoder("sst")(signal).shape == (64, 256)
    info = get_representation_info("sst")
    assert info.canonical_method is True
    assert "Daubechies" in (info.reference or "")
