"""Utilities for integrating :mod:`tscv-vision` into ML pipelines."""
# mypy: ignore-errors

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .sliding import features_for_sliding

Array = NDArray[np.float64]

if TYPE_CHECKING:  # pragma: no cover - import for type check only
    import tensorflow as tf  # type: ignore
    from onnx import TensorProto  # type: ignore
    from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
else:  # pragma: no cover - runtime fallback
    class BaseEstimator:  # type: ignore[too-many-ancestors]
        pass

    class TransformerMixin:  # type: ignore[too-many-ancestors]
        pass

    tf = None  # type: ignore
    TensorProto = Any  # type: ignore


class SklearnFeatureTransformer(TransformerMixin, BaseEstimator):
    """scikit-learn transformer that extracts features from time series.

    Parameters
    ----------
    encoder:
        Name of the encoder registered in :mod:`tscv_vision.encoders`.
    bins:
        Number of histogram bins.
    feature_names:
        Optional subset of feature extractor names.
    """

    def __init__(
        self,
        encoder: str = "gaf",
        *,
        bins: int = 32,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.encoder = encoder
        self.bins = bins
        self.feature_names = feature_names

    def fit(self, X: Array, y: Any | None = None) -> SklearnFeatureTransformer:  # noqa: D401 - standard sklearn signature
        """No-op fit method for compatibility."""
        return self

    def transform(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_points)")
        feats = []
        for row in X:
            f, _ = features_for_sliding(
                row,
                encoder=self.encoder,
                size=row.shape[-1],
                hop=row.shape[-1],
                bins=self.bins,
                feature_names=self.feature_names,
            )
            feats.append(f[0])
        return np.stack(feats)


class TorchFeatureDataset:  # pragma: no cover - small wrapper
    """PyTorch dataset producing feature vectors from time series."""

    def __init__(
        self,
        series: Array,
        *,
        encoder: str = "gaf",
        bins: int = 32,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        try:
            import torch.utils.data as data  # type: ignore
        except Exception as exc:  # pragma: no cover - import error
            raise ImportError("torch is required for TorchFeatureDataset") from exc
        self._data_module = data
        self.series = np.asarray(series, dtype=float)
        self.encoder = encoder
        self.bins = bins
        self.feature_names = feature_names

    def __len__(self) -> int:
        return int(self.series.shape[0])

    def __getitem__(self, idx: int) -> Array:
        row = self.series[idx]
        f, _ = features_for_sliding(
            row,
            encoder=self.encoder,
            size=row.shape[-1],
            hop=row.shape[-1],
            bins=self.bins,
            feature_names=self.feature_names,
        )
        return f[0]


def tf_feature_dataset(
    series: Array,
    *,
    encoder: str = "gaf",
    bins: int = 32,
    feature_names: Sequence[str] | None = None,
) -> Any:  # pragma: no cover - requires tensorflow
    """Build a ``tf.data.Dataset`` yielding feature vectors."""
    try:
        import tensorflow as _tf  # type: ignore
    except Exception as exc:  # pragma: no cover - import error
        raise ImportError("tensorflow is required for tf_feature_dataset") from exc

    series_arr = np.asarray(series, dtype=float)

    def gen() -> Iterator[Array]:
        for row in series_arr:
            f, _ = features_for_sliding(
                row,
                encoder=encoder,
                size=row.shape[-1],
                hop=row.shape[-1],
                bins=bins,
                feature_names=feature_names,
            )
            yield f[0]

    first, _ = features_for_sliding(
        series_arr[0],
        encoder=encoder,
        size=series_arr[0].shape[-1],
        hop=series_arr[0].shape[-1],
        bins=bins,
        feature_names=feature_names,
    )
    output_shape = (first[0].shape[0],)
    return _tf.data.Dataset.from_generator(
        gen, output_types=_tf.float64, output_shapes=output_shape
    )


def to_onnx_tensor(
    array: Array,
    name: str = "features",
) -> TensorProto:  # pragma: no cover - requires onnx
    """Convert ``array`` into an ONNX ``TensorProto``."""
    try:
        from onnx import numpy_helper
    except Exception as exc:  # pragma: no cover - import error
        raise ImportError("onnx is required for ONNX export") from exc
    return numpy_helper.from_array(array.astype(np.float32), name)


def save_onnx(
    array: Array,
    path: str,
    name: str = "features",
) -> None:  # pragma: no cover - requires onnx
    """Save ``array`` as an ONNX ``TensorProto`` to ``path``."""
    tensor = to_onnx_tensor(array, name)
    with open(path, "wb") as f:
        f.write(tensor.SerializeToString())


__all__ = [
    "SklearnFeatureTransformer",
    "TorchFeatureDataset",
    "tf_feature_dataset",
    "to_onnx_tensor",
    "save_onnx",
]
