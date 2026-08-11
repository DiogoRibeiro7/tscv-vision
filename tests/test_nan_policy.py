"""Tests for NaN/inf handling via the nan_policy parameter."""

from __future__ import annotations

import numpy as np
import pytest

from tscv_vision import encoders


@pytest.fixture
def signal_with_nans() -> np.ndarray:
    """A 16-element sine wave with NaN injected at indices 3, 7, 12."""
    x = np.sin(np.linspace(0, 2 * np.pi, 16))
    x[3] = np.nan
    x[7] = np.nan
    x[12] = np.nan
    return x


@pytest.fixture
def signal_with_inf() -> np.ndarray:
    """A 16-element ramp with +inf at index 5."""
    x = np.linspace(0, 1, 16)
    x[5] = np.inf
    return x


@pytest.fixture
def signal_leading_nans() -> np.ndarray:
    """Series with NaN at the start."""
    x = np.linspace(0, 1, 16)
    x[0] = np.nan
    x[1] = np.nan
    return x


# -- _handle_nans / _validate_series ------------------------------------------


class TestValidateSeriesNanPolicy:
    def test_raise_is_default(self, signal_with_nans: np.ndarray) -> None:
        with pytest.raises(ValueError, match="NaN or infinite"):
            encoders._validate_series(signal_with_nans)

    def test_raise_explicit(self, signal_with_nans: np.ndarray) -> None:
        with pytest.raises(ValueError, match="NaN or infinite"):
            encoders._validate_series(signal_with_nans, nan_policy="raise")

    def test_omit_removes_nans(self, signal_with_nans: np.ndarray) -> None:
        result = encoders._validate_series(signal_with_nans, nan_policy="omit")
        assert np.all(np.isfinite(result))
        assert result.size == 13  # 16 - 3 NaNs

    def test_omit_all_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="empty after removing"):
            encoders._validate_series(
                np.array([np.nan, np.nan]), nan_policy="omit"
            )

    def test_interpolate_fills_gaps(self, signal_with_nans: np.ndarray) -> None:
        result = encoders._validate_series(
            signal_with_nans, nan_policy="interpolate"
        )
        assert result.size == 16
        assert np.all(np.isfinite(result))

    def test_interpolate_all_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="no finite values"):
            encoders._validate_series(
                np.array([np.nan, np.nan]), nan_policy="interpolate"
            )

    def test_forward_fill(self, signal_with_nans: np.ndarray) -> None:
        result = encoders._validate_series(
            signal_with_nans, nan_policy="forward_fill"
        )
        assert result.size == 16
        assert np.all(np.isfinite(result))

    def test_forward_fill_leading_nans(
        self, signal_leading_nans: np.ndarray
    ) -> None:
        result = encoders._validate_series(
            signal_leading_nans, nan_policy="forward_fill"
        )
        assert np.all(np.isfinite(result))
        # leading NaNs should be backward-filled from first valid value
        assert result[0] == result[2]
        assert result[1] == result[2]

    def test_forward_fill_all_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="no finite values"):
            encoders._validate_series(
                np.array([np.nan, np.nan]), nan_policy="forward_fill"
            )

    def test_inf_handled_same_as_nan(self, signal_with_inf: np.ndarray) -> None:
        result = encoders._validate_series(
            signal_with_inf, nan_policy="omit"
        )
        assert np.all(np.isfinite(result))
        assert result.size == 15

    def test_invalid_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="nan_policy must be"):
            encoders._validate_series(
                np.array([1.0, np.nan]), nan_policy="bad"  # type: ignore[arg-type]
            )

    def test_clean_input_unchanged(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        for policy in ("raise", "omit", "interpolate", "forward_fill"):
            result = encoders._validate_series(
                x.copy(), nan_policy=policy  # type: ignore[arg-type]
            )
            np.testing.assert_array_equal(result, x)


# -- Encoder-level nan_policy propagation ------------------------------------


class TestEncoderNanPolicy:
    """Verify that nan_policy is threaded through to each encoder."""

    def test_gaf_interpolate(self, signal_with_nans: np.ndarray) -> None:
        img = encoders.gaf(signal_with_nans, nan_policy="interpolate")
        assert img.shape == (16, 16)
        assert np.all(np.isfinite(img))

    def test_gaf_omit(self, signal_with_nans: np.ndarray) -> None:
        img = encoders.gaf(signal_with_nans, nan_policy="omit")
        assert img.shape == (13, 13)
        assert np.all(np.isfinite(img))

    def test_recurrence_plot_interpolate(
        self, signal_with_nans: np.ndarray
    ) -> None:
        rp = encoders.recurrence_plot(
            signal_with_nans, nan_policy="interpolate"
        )
        assert rp.shape == (16, 16)
        assert np.all(np.isfinite(rp))

    def test_spectrogram_forward_fill(
        self, signal_with_nans: np.ndarray
    ) -> None:
        spec = encoders.spectrogram(
            signal_with_nans, win=8, nan_policy="forward_fill"
        )
        assert np.all(np.isfinite(spec))

    def test_persistence_image_omit(
        self, signal_with_nans: np.ndarray
    ) -> None:
        img = encoders.persistence_image(
            signal_with_nans, nan_policy="omit"
        )
        assert np.all(np.isfinite(img))

    def test_mtf_interpolate(self, signal_with_nans: np.ndarray) -> None:
        img = encoders.mtf(signal_with_nans, nan_policy="interpolate")
        assert img.shape == (16, 16)
        assert np.all(np.isfinite(img))

    def test_gdf_forward_fill(self, signal_with_nans: np.ndarray) -> None:
        img = encoders.gdf(signal_with_nans, nan_policy="forward_fill")
        assert img.shape == (16, 16)
        assert np.all(np.isfinite(img))

    def test_dtw_matrix_omit(self, signal_with_nans: np.ndarray) -> None:
        img = encoders.dtw_matrix(signal_with_nans, nan_policy="omit")
        assert img.shape == (13, 13)
        assert np.all(np.isfinite(img))

    def test_sax_interpolate(self, signal_with_nans: np.ndarray) -> None:
        img = encoders.sax(signal_with_nans, nan_policy="interpolate")
        assert np.all(np.isfinite(img))

    def test_window_attention_interpolate(self, signal_with_nans: np.ndarray) -> None:
        img = encoders.window_attention(signal_with_nans, nan_policy="interpolate")
        assert np.all(np.isfinite(img))

    def test_visibility_graph_forward_fill(
        self, signal_with_nans: np.ndarray
    ) -> None:
        img = encoders.visibility_graph(
            signal_with_nans, nan_policy="forward_fill"
        )
        assert img.shape == (16, 16)
        assert np.all(np.isfinite(img))

    def test_multi_scale_conv_interpolate(
        self, signal_with_nans: np.ndarray
    ) -> None:
        img = encoders.multi_scale_conv(
            signal_with_nans, nan_policy="interpolate"
        )
        assert np.all(np.isfinite(img))

    def test_matrix_profile_omit(self, signal_with_nans: np.ndarray) -> None:
        mp = encoders.matrix_profile(
            signal_with_nans, m=3, nan_policy="omit"
        )
        assert np.all(np.isfinite(mp))

    def test_random_projection_interpolate(
        self, signal_with_nans: np.ndarray
    ) -> None:
        img = encoders.random_projection_image(
            signal_with_nans, nan_policy="interpolate"
        )
        assert img.shape == (32, 32)
        assert np.all(np.isfinite(img))

    def test_ensemble_interpolate(
        self, signal_with_nans: np.ndarray
    ) -> None:
        imgs = encoders.ensemble(
            signal_with_nans,
            names=["gaf", "rp"],
            nan_policy="interpolate",
        )
        assert np.all(np.isfinite(imgs))
