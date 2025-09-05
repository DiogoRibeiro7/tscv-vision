from __future__ import annotations

import numpy as np
import pytest

from tscv_vision.ml_integration import (
    SklearnFeatureTransformer,
    TorchFeatureDataset,
    tf_feature_dataset,
    to_onnx_tensor,
)

pytestmark = pytest.mark.optional


def _sample_series(n: int = 32) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    return np.sin(t)


def test_sklearn_transformer() -> None:
    pytest.importorskip("sklearn")
    X = np.stack([_sample_series() for _ in range(2)])
    trans = SklearnFeatureTransformer("gaf")
    out = trans.fit_transform(X)
    assert out.shape[0] == 2


def test_torch_dataset() -> None:
    pytest.importorskip("torch")
    series = np.stack([_sample_series() for _ in range(3)])
    ds = TorchFeatureDataset(series, encoder="gaf")
    item = ds[0]
    assert item.ndim == 1


def test_tf_dataset() -> None:
    pytest.importorskip("tensorflow")
    series = np.stack([_sample_series() for _ in range(2)])
    ds = tf_feature_dataset(series)
    first = next(iter(ds.as_numpy_iterator()))
    assert first.ndim == 1


def test_onnx_export() -> None:
    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(0)
    arr = rng.random((4, 5))
    tensor = to_onnx_tensor(arr, name="const")
    from onnx import TensorProto, helper  # type: ignore

    graph = helper.make_graph(
        [helper.make_node("Identity", ["const"], ["out"])],
        "g",
        [],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(arr.shape))],
        initializer=[tensor],
    )
    model = helper.make_model(graph)
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    result = sess.run(None, {})[0]
    assert np.allclose(result, arr.astype(np.float32))
