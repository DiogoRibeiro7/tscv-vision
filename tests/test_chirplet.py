"""Validation of the chirplet transform.

A chirplet transform earns its name by resolving *chirp rate*, so the tests
recover known sweep rates rather than checking that a time-frequency picture
appears. The distinguishing test is against an STFT: for a stationary tone the
two agree, for a sweeping tone they do not.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscv_vision import encoders

FS = 200.0
N = 1024
WINDOW = 128
RATES = np.linspace(-FS**2 / (4 * WINDOW), FS**2 / (4 * WINDOW), 21)


@pytest.fixture
def time() -> np.ndarray:
    return np.arange(N) / FS


def linear_chirp(t: np.ndarray, f0: float, rate: float) -> np.ndarray:
    """Instantaneous frequency f0 + rate * t."""
    return np.sin(2 * np.pi * (f0 * t + 0.5 * rate * t**2))


def best_rate(tensor: np.ndarray, rates: np.ndarray) -> float:
    """Chirp rate of the strongest atom in the whole tensor."""
    return float(rates[np.unravel_index(np.argmax(tensor), tensor.shape)[0]])


# ---------------------------------------------------------------------------
# It resolves chirp rate


@pytest.mark.parametrize("rate", [-60.0, -30.0, 0.0, 30.0, 60.0])
def test_recovers_the_chirp_rate_of_a_linear_chirp(time: np.ndarray, rate: float) -> None:
    signal = linear_chirp(time, 30.0, rate)
    tensor = encoders.chirplet_transform(
        signal,
        fs=FS,
        window_size=WINDOW,
        chirp_rates=RATES,
        aggregate="none",
        log_scale=False,
    )
    step = RATES[1] - RATES[0]
    assert abs(best_rate(tensor, RATES) - rate) <= step


def test_crossing_chirps_show_both_rates(time: np.ndarray) -> None:
    """One rising and one falling sweep: both rates must be represented."""

    signal = linear_chirp(time, 20.0, 40.0) + linear_chirp(time, 80.0, -40.0)
    tensor = encoders.chirplet_transform(
        signal,
        fs=FS,
        window_size=WINDOW,
        chirp_rates=RATES,
        aggregate="none",
        log_scale=False,
    )
    strength = tensor.max(axis=(1, 2))
    top_two = np.sort(RATES[np.argsort(strength)[-2:]])
    step = RATES[1] - RATES[0]
    assert abs(top_two[0] + 40.0) <= step
    assert abs(top_two[1] - 40.0) <= step


def test_time_reversal_negates_the_recovered_rate(time: np.ndarray) -> None:
    signal = linear_chirp(time, 30.0, 45.0)
    forward = encoders.chirplet_transform(
        signal, fs=FS, window_size=WINDOW, chirp_rates=RATES,
        aggregate="none", log_scale=False,
    )
    backward = encoders.chirplet_transform(
        signal[::-1].copy(), fs=FS, window_size=WINDOW, chirp_rates=RATES,
        aggregate="none", log_scale=False,
    )
    assert best_rate(forward, RATES) == pytest.approx(-best_rate(backward, RATES))


def test_stationary_tone_prefers_zero_chirp_rate(time: np.ndarray) -> None:
    tensor = encoders.chirplet_transform(
        np.sin(2 * np.pi * 40.0 * time),
        fs=FS, window_size=WINDOW, chirp_rates=RATES,
        aggregate="none", log_scale=False,
    )
    assert best_rate(tensor, RATES) == pytest.approx(0.0, abs=RATES[1] - RATES[0])


# ---------------------------------------------------------------------------
# It is not an STFT with a different window


def test_agrees_with_an_stft_only_at_zero_chirp_rate(time: np.ndarray) -> None:
    """The c=0 slice *is* a windowed Fourier transform; other slices are not."""

    signal = linear_chirp(time, 30.0, 45.0)
    tensor = encoders.chirplet_transform(
        signal, fs=FS, window_size=WINDOW, chirp_rates=np.array([0.0, 45.0]),
        aggregate="none", log_scale=False,
    )
    # Compute the c = 0 slice directly as a windowed FFT.
    hop = WINDOW // 4
    n_frames = 1 + (N - WINDOW) // hop
    window = np.hanning(WINDOW)
    expected = np.empty((WINDOW // 2 + 1, n_frames))
    for f in range(n_frames):
        frame = signal[f * hop : f * hop + WINDOW] * window
        expected[:, f] = np.abs(np.fft.fft(frame))[: WINDOW // 2 + 1]
    # The encoder normalises the tensor by its global peak, so compare shapes
    # of the spectrum rather than absolute scale.
    np.testing.assert_allclose(
        tensor[0] / tensor[0].max(), expected / expected.max(), atol=1e-10
    )
    # The matched-rate slice concentrates the sweep better than c = 0 does.
    assert tensor[1].max() > 1.3 * tensor[0].max()


def test_a_sweeping_tone_is_sharper_at_its_own_rate(time: np.ndarray) -> None:
    """A matched chirplet concentrates energy an STFT slice cannot."""

    signal = linear_chirp(time, 30.0, 60.0)

    def concentration(rate: float) -> float:
        slice_ = encoders.chirplet_transform(
            signal, fs=FS, window_size=WINDOW, chirp_rates=np.array([rate]),
            aggregate="none", log_scale=False,
        )[0]
        flat = slice_.ravel() / slice_.sum()
        return float(np.sum(flat**2))

    assert concentration(60.0) > concentration(0.0)


# ---------------------------------------------------------------------------
# Atom correlation


def test_matches_a_direct_atom_correlation(time: np.ndarray) -> None:
    """Compare one cell against the inner product with its chirplet atom."""

    signal = linear_chirp(time, 25.0, 20.0)
    rate, frequency = 20.0, 25.0
    tensor = encoders.chirplet_transform(
        signal,
        fs=FS,
        window_size=WINDOW,
        chirp_rates=np.array([rate]),
        frequencies=np.array([frequency]),
        aggregate="none",
        log_scale=False,
    )

    hop = WINDOW // 4
    window = np.hanning(WINDOW)
    tau = (np.arange(WINDOW) - (WINDOW - 1) / 2.0) / FS
    atom = window * np.exp(2j * np.pi * (frequency * tau + 0.5 * rate * tau**2))
    expected = np.array(
        [
            abs(np.vdot(atom, signal[f * hop : f * hop + WINDOW]))
            for f in range(tensor.shape[2])
        ]
    )
    np.testing.assert_allclose(
        tensor[0, 0] * expected.max() / tensor[0, 0].max(), expected, rtol=1e-9
    )


# ---------------------------------------------------------------------------
# Shape, aggregation, options


def test_default_shape_and_range(time: np.ndarray) -> None:
    image = encoders.chirplet_transform(linear_chirp(time, 30.0, 15.0), fs=FS)
    assert image.shape == (33, 1 + (N - 64) // 16)
    assert np.all(image >= 0.0) and np.all(image <= 1.0)
    assert image.max() == pytest.approx(1.0)


@pytest.mark.parametrize("aggregate", ["max", "mean", "energy"])
def test_aggregations_give_a_two_dimensional_image(
    aggregate: str, time: np.ndarray
) -> None:
    image = encoders.chirplet_transform(
        linear_chirp(time, 30.0, 30.0), fs=FS, window_size=WINDOW,
        chirp_rates=RATES, aggregate=aggregate,  # type: ignore[arg-type]
    )
    assert image.ndim == 2
    assert image.shape[0] == WINDOW // 2 + 1
    assert np.all(np.isfinite(image))


def test_aggregations_are_ordered_as_expected(time: np.ndarray) -> None:
    """max <= energy for non-negative magnitudes, and mean <= max."""

    kwargs = dict(fs=FS, window_size=WINDOW, chirp_rates=RATES, log_scale=False)
    tensor = encoders.chirplet_transform(
        linear_chirp(time, 30.0, 30.0), aggregate="none", **kwargs  # type: ignore[arg-type]
    )
    raw_max = tensor.max(axis=0)
    raw_mean = tensor.mean(axis=0)
    raw_energy = np.sqrt((tensor**2).sum(axis=0))
    assert np.all(raw_mean <= raw_max + 1e-12)
    assert np.all(raw_max <= raw_energy + 1e-12)


def test_none_returns_the_full_tensor(time: np.ndarray) -> None:
    tensor = encoders.chirplet_transform(
        linear_chirp(time, 30.0, 30.0), fs=FS, window_size=WINDOW,
        chirp_rates=RATES, aggregate="none",
    )
    assert tensor.shape[0] == RATES.size
    assert tensor.shape[1] == WINDOW // 2 + 1


def test_explicit_frequencies(time: np.ndarray) -> None:
    wanted = np.array([10.0, 30.0, 55.0])
    image = encoders.chirplet_transform(
        np.sin(2 * np.pi * 30.0 * time), fs=FS, window_size=WINDOW, frequencies=wanted
    )
    assert image.shape[0] == 3
    # The requested 30 Hz row dominates for a 30 Hz tone.
    assert int(np.argmax(image[:, 5])) == 1


def test_log_scale(time: np.ndarray) -> None:
    signal = linear_chirp(time, 30.0, 30.0)
    linear = encoders.chirplet_transform(signal, fs=FS, log_scale=False)
    logged = encoders.chirplet_transform(signal, fs=FS, log_scale=True)
    assert logged.mean() > linear.mean()
    assert logged.max() == pytest.approx(1.0)


def test_amplitude_invariance(time: np.ndarray) -> None:
    signal = linear_chirp(time, 30.0, 30.0)
    np.testing.assert_allclose(
        encoders.chirplet_transform(signal, fs=FS),
        encoders.chirplet_transform(13.0 * signal, fs=FS),
        atol=1e-12,
    )


def test_deterministic(time: np.ndarray) -> None:
    signal = linear_chirp(time, 30.0, 30.0)
    np.testing.assert_array_equal(
        encoders.chirplet_transform(signal, fs=FS),
        encoders.chirplet_transform(signal, fs=FS),
    )


def test_zero_signal(time: np.ndarray) -> None:
    assert np.all(encoders.chirplet_transform(np.zeros(N), fs=FS) == 0.0)


# ---------------------------------------------------------------------------
# Safeguards


def test_memory_guard_refuses_an_oversized_tensor(time: np.ndarray) -> None:
    with pytest.raises(ValueError, match="above max_bytes"):
        encoders.chirplet_transform(
            np.sin(2 * np.pi * 30.0 * time),
            fs=FS,
            chirp_rates=np.linspace(-100.0, 100.0, 5000),
            max_bytes=1000,
        )
    # The message quotes the actual size and the grids responsible.
    with pytest.raises(ValueError, match=r"5000 chirp rates x \d+ frequencies"):
        encoders.chirplet_transform(
            np.sin(2 * np.pi * 30.0 * time),
            fs=FS,
            chirp_rates=np.linspace(-100.0, 100.0, 5000),
            max_bytes=1000,
        )


def test_default_budget_is_documented() -> None:
    assert encoders.MAX_CHIRPLET_BYTES == 512 * 1024**2


def test_invalid_arguments(time: np.ndarray) -> None:
    signal = np.sin(2 * np.pi * 30.0 * time)
    for kwargs, match in [
        ({"fs": 0.0}, "fs must be"),
        ({"window_size": 2}, "window_size must be"),
        ({"window_size": 99_999}, "exceeds the series"),
        ({"hop_length": 0}, "hop_length"),
        ({"aggregate": "nope"}, "aggregate"),
        ({"chirp_rates": np.array([])}, "chirp_rates"),
        ({"chirp_rates": np.array([np.nan])}, "chirp_rates"),
        ({"chirp_rates": np.zeros((2, 2))}, "chirp_rates"),
        ({"frequencies": np.array([])}, "frequencies"),
        ({"frequencies": np.array([np.inf])}, "frequencies"),
    ]:
        with pytest.raises(ValueError, match=match):
            encoders.chirplet_transform(signal, **kwargs)  # type: ignore[arg-type]


def test_nan_policy(time: np.ndarray) -> None:
    signal = np.sin(2 * np.pi * 30.0 * time)
    dirty = signal.copy()
    dirty[100] = np.nan
    with pytest.raises(ValueError):
        encoders.chirplet_transform(dirty, fs=FS)
    assert np.all(
        np.isfinite(encoders.chirplet_transform(dirty, fs=FS, nan_policy="interpolate"))
    )


def test_registry_and_metadata(time: np.ndarray) -> None:
    from tscv_vision.representations import get_representation_info

    signal = linear_chirp(time, 30.0, 30.0)
    np.testing.assert_array_equal(
        encoders.get_encoder("chirplet")(signal), encoders.chirplet_transform(signal)
    )
    info = get_representation_info("chirplet")
    assert info.canonical_method is True
    assert "Mann" in (info.reference or "")
