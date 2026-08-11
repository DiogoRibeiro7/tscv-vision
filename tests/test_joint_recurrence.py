"""Validation of the joint recurrence plot.

Joint recurrence is the AND of per-channel recurrence, and the properties worth
testing follow from that: it must reduce to a plain recurrence plot on one
channel, be blind to channel order, and — because each channel gets its own
threshold — be unmoved by rescaling any single channel.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscv_vision.multivariate import (
    cross_recurrence_plot,
    joint_recurrence_plot,
)


@pytest.fixture
def channels() -> np.ndarray:
    t = np.linspace(0, 12.0, 128)
    return np.column_stack([np.sin(t), np.cos(t)])


def coupled_system(n: int = 300, coupling: float = 0.0, seed: int = 0) -> np.ndarray:
    """Two logistic maps, diffusively coupled with strength ``coupling``."""

    rng = np.random.default_rng(seed)
    a, b = rng.uniform(0.2, 0.8, size=2)
    out = np.empty((n, 2))
    for i in range(n):
        next_a = 3.9 * a * (1.0 - a)
        next_b = 3.8 * b * (1.0 - b)
        a = (1.0 - coupling) * next_a + coupling * next_b
        b = (1.0 - coupling) * next_b + coupling * next_a
        out[i] = (a, b)
    return out


# ---------------------------------------------------------------------------
# Reduction to the single-channel case


def test_one_channel_reduces_to_a_plain_recurrence_plot() -> None:
    series = np.sin(np.linspace(0, 20.0, 120))
    np.testing.assert_array_equal(
        joint_recurrence_plot(series[:, None], recurrence_rate=0.1),
        cross_recurrence_plot(series, series, recurrence_rate=0.1),
    )


def test_and_is_the_product_of_the_per_channel_plots(channels: np.ndarray) -> None:
    """The canonical definition, checked against its own components."""

    joint = joint_recurrence_plot(channels, recurrence_rate=0.15)
    per_channel = [
        cross_recurrence_plot(channels[:, c], channels[:, c], recurrence_rate=0.15)
        for c in range(channels.shape[1])
    ]
    np.testing.assert_array_equal(joint, per_channel[0] * per_channel[1])


# ---------------------------------------------------------------------------
# Structure


def test_symmetry_and_diagonal(channels: np.ndarray) -> None:
    joint = joint_recurrence_plot(channels)
    assert joint.shape == (128, 128)
    np.testing.assert_array_equal(joint, joint.T)
    np.testing.assert_array_equal(np.diag(joint), np.ones(128))
    assert set(np.unique(joint)) <= {0.0, 1.0}


@pytest.mark.parametrize("combination", ["and", "product", "mean"])
def test_channel_order_invariance(combination: str, channels: np.ndarray) -> None:
    """AND, product and mean are all commutative, so order cannot matter."""

    forward = joint_recurrence_plot(channels, combination=combination)  # type: ignore[arg-type]
    reversed_channels = joint_recurrence_plot(
        channels[:, ::-1], combination=combination  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(forward, reversed_channels, atol=1e-12)


def test_joint_rate_is_far_below_the_per_channel_rate() -> None:
    """For independent channels the AND multiplies the rates."""

    rng = np.random.default_rng(0)
    independent = rng.normal(size=(300, 3))
    joint = joint_recurrence_plot(independent, recurrence_rate=0.2)
    assert joint.mean() < 0.2**2, "the AND did not suppress unshared recurrences"
    assert joint.mean() > 0.2**4, "the AND suppressed everything"


def test_coupled_channels_recur_together_more_than_uncoupled_ones() -> None:
    uncoupled = joint_recurrence_plot(coupled_system(coupling=0.0), recurrence_rate=0.2)
    coupled = joint_recurrence_plot(coupled_system(coupling=0.45), recurrence_rate=0.2)
    assert coupled.mean() > 1.5 * uncoupled.mean()


def test_embedding_shrinks_the_plot(channels: np.ndarray) -> None:
    joint = joint_recurrence_plot(channels, dimension=3, delay=4)
    assert joint.shape == (128 - 8, 128 - 8)


# ---------------------------------------------------------------------------
# Per-channel thresholds


def test_invariant_to_rescaling_any_single_channel(channels: np.ndarray) -> None:
    """The point of thresholding per channel rather than globally."""

    base = joint_recurrence_plot(channels)
    for scale in (1e-3, 1e3):
        rescaled = channels.copy()
        rescaled[:, 1] *= scale
        np.testing.assert_array_equal(joint_recurrence_plot(rescaled), base)


def test_a_shared_epsilon_is_not_scale_invariant(channels: np.ndarray) -> None:
    """Which is why an explicit epsilon must be given per channel if needed."""

    rescaled = channels.copy()
    rescaled[:, 1] *= 1000.0
    assert not np.array_equal(
        joint_recurrence_plot(channels, epsilon=0.5),
        joint_recurrence_plot(rescaled, epsilon=0.5),
    )


def test_channel_specific_epsilon(channels: np.ndarray) -> None:
    rescaled = channels.copy()
    rescaled[:, 1] *= 1000.0
    matched = joint_recurrence_plot(rescaled, epsilon=np.array([0.5, 500.0]))
    baseline = joint_recurrence_plot(channels, epsilon=np.array([0.5, 0.5]))
    np.testing.assert_array_equal(matched, baseline)


def test_scalar_epsilon_applies_to_every_channel(channels: np.ndarray) -> None:
    np.testing.assert_array_equal(
        joint_recurrence_plot(channels, epsilon=0.4),
        joint_recurrence_plot(channels, epsilon=np.array([0.4, 0.4])),
    )


def test_epsilon_shape_is_checked(channels: np.ndarray) -> None:
    with pytest.raises(ValueError, match="one entry per channel"):
        joint_recurrence_plot(channels, epsilon=np.array([0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# Combination rules


def test_extensions_are_continuous_and_bounded(channels: np.ndarray) -> None:
    binary = joint_recurrence_plot(channels, combination="and")
    product = joint_recurrence_plot(channels, combination="product")
    mean = joint_recurrence_plot(channels, combination="mean")

    assert set(np.unique(binary)) <= {0.0, 1.0}
    for graded in (product, mean):
        assert np.all(graded >= 0.0) and np.all(graded <= 1.0)
        assert len(np.unique(graded)) > 2, "an extension should not be binary"
        np.testing.assert_allclose(np.diag(graded), 1.0, atol=1e-12)
        np.testing.assert_allclose(graded, graded.T, atol=1e-12)


def test_mean_is_the_average_of_the_channel_similarities(channels: np.ndarray) -> None:
    mean = joint_recurrence_plot(channels, combination="mean")
    per_channel = [
        cross_recurrence_plot(channels[:, c], channels[:, c], binary=False)
        for c in range(channels.shape[1])
    ]
    np.testing.assert_allclose(mean, np.mean(per_channel, axis=0), atol=1e-12)


def test_product_is_the_product_of_the_channel_similarities(
    channels: np.ndarray,
) -> None:
    product = joint_recurrence_plot(channels, combination="product")
    per_channel = [
        cross_recurrence_plot(channels[:, c], channels[:, c], binary=False)
        for c in range(channels.shape[1])
    ]
    np.testing.assert_allclose(product, per_channel[0] * per_channel[1], atol=1e-12)


def test_product_is_never_above_the_mean(channels: np.ndarray) -> None:
    """AM-GM in effect: multiplying similarities in [0,1] cannot exceed averaging."""

    product = joint_recurrence_plot(channels, combination="product")
    mean = joint_recurrence_plot(channels, combination="mean")
    assert np.all(product <= mean + 1e-12)


# ---------------------------------------------------------------------------
# Edge cases


def test_constant_channel_recurs_everywhere() -> None:
    t = np.linspace(0, 12.0, 64)
    with_constant = np.column_stack([np.sin(t), np.full(64, 5.0)])
    only_varying = np.sin(t)[:, None]
    # A constant channel is within any threshold of itself, so the AND is
    # decided entirely by the other channel.
    np.testing.assert_array_equal(
        joint_recurrence_plot(with_constant, recurrence_rate=0.1),
        joint_recurrence_plot(only_varying, recurrence_rate=0.1),
    )


def test_all_constant_channels() -> None:
    joint = joint_recurrence_plot(np.full((16, 3), 2.0))
    np.testing.assert_array_equal(joint, np.ones((16, 16)))


def test_deterministic(channels: np.ndarray) -> None:
    np.testing.assert_array_equal(
        joint_recurrence_plot(channels), joint_recurrence_plot(channels)
    )


def test_invalid_arguments(channels: np.ndarray) -> None:
    with pytest.raises(ValueError, match="2D"):
        joint_recurrence_plot(np.sin(np.linspace(0, 1, 10)))
    with pytest.raises(ValueError, match="at least one channel"):
        joint_recurrence_plot(np.zeros((10, 0)))
    for kwargs, match in [
        ({"dimension": 0}, "dimension"),
        ({"delay": 0}, "delay"),
        ({"combination": "xor"}, "combination"),
        ({"metric": "cosine"}, "metric"),
        ({"recurrence_rate": 0.0}, "recurrence_rate"),
    ]:
        with pytest.raises(ValueError, match=match):
            joint_recurrence_plot(channels, **kwargs)  # type: ignore[arg-type]


def test_nan_policy_that_shortens_channels_unequally_is_rejected() -> None:
    t = np.linspace(0, 12.0, 64)
    dirty = np.column_stack([np.sin(t), np.cos(t)])
    dirty[3, 0] = np.nan
    dirty[7, 1] = np.nan
    dirty[9, 1] = np.nan
    with pytest.raises(ValueError, match="different lengths"):
        joint_recurrence_plot(dirty, nan_policy="omit")
    # Interpolation keeps the sample axis aligned, so it works.
    assert np.all(np.isfinite(joint_recurrence_plot(dirty, nan_policy="interpolate")))


def test_metadata_is_recorded() -> None:
    from tscv_vision.representations import get_encoder_metadata, list_encoders

    info = get_encoder_metadata("joint_recurrence_plot")
    assert info.canonical_method is True
    assert info.input_kind == "multivariate"
    assert "Romano" in (info.reference or "")
    assert "extension" in info.notes.lower()
    assert "joint_recurrence_plot" not in list_encoders()
