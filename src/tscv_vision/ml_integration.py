"""Utilities for integrating :mod:`tscv-vision` into ML pipelines."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ._sklearn_compat import BaseEstimator, TransformerMixin
from .sliding import features_for_sliding

Array = NDArray[np.float64]

if TYPE_CHECKING:  # pragma: no cover - import for type check only
    from onnx import TensorProto
else:
    TensorProto = Any


class SklearnFeatureTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """scikit-learn transformer that extracts features from time series.

    When scikit-learn is installed this subclasses the real
    :class:`~sklearn.base.BaseEstimator`/:class:`~sklearn.base.TransformerMixin`,
    so it composes with :class:`~sklearn.pipeline.Pipeline`,
    :func:`~sklearn.model_selection.cross_val_score` and clone/grid-search.
    Without scikit-learn it falls back to minimal stubs that still provide
    ``get_params``/``set_params``/``fit_transform``.

    Parameters
    ----------
    encoder:
        Name of the encoder registered in :mod:`tscv_vision.encoders`.
    bins:
        Number of histogram bins.
    feature_names:
        Optional subset of feature extractor names.

    Examples
    --------
    >>> import numpy as np
    >>> tr = SklearnFeatureTransformer(bins=8)
    >>> out = tr.fit_transform(np.random.default_rng(0).random((3, 16)))
    >>> out.shape[0]
    3
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

    @staticmethod
    def _check_X(X: Array) -> Array:
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_points)")
        return arr

    def fit(self, X: Array, y: Any | None = None) -> SklearnFeatureTransformer:
        """Validate ``X`` and record its shape; no statistics are learned."""
        arr = self._check_X(X)
        self.n_features_in_ = int(arr.shape[1])
        return self

    def transform(self, X: Array) -> Array:
        """Encode every row of ``X`` and return its image feature vector."""
        arr = self._check_X(X)
        feats = []
        for row in arr:
            f, _ = features_for_sliding(
                row,
                encoder=self.encoder,
                size=row.shape[-1],
                hop=row.shape[-1],
                bins=self.bins,
                feature_names=self.feature_names,
            )
            feats.append(f[0])
        out: Array = np.stack(feats)
        self.n_features_out_ = int(out.shape[1])
        return out

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[np.str_]:
        """Return generated feature names, requires a prior ``transform`` call."""
        n_out = getattr(self, "n_features_out_", None)
        if n_out is None:
            raise AttributeError(
                "call transform() before get_feature_names_out(); the output "
                "dimensionality depends on the encoder and the installed "
                "optional feature extractors"
            )
        prefix = f"{self.encoder}_"
        return np.asarray([f"{prefix}{i}" for i in range(n_out)], dtype=object).astype(str)


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
            import torch.utils.data as data
        except Exception as exc:  # pragma: no cover - import error
            raise ImportError("torch is required for TorchFeatureDataset") from exc
        self._data_module = data
        self.series = np.asarray(series, dtype=float)
        self.encoder = encoder
        self.bins = bins
        self.feature_names = feature_names

    def __len__(self) -> int:
        return int(self.series.shape[0])

    def __getitem__(self, idx: int) -> Any:
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
        import tensorflow as _tf
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
