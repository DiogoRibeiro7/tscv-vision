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


def test_sklearn_transformer_is_a_real_estimator() -> None:
    """The class must inherit sklearn's bases, not local stubs."""

    sklearn_base = pytest.importorskip("sklearn.base")
    assert issubclass(SklearnFeatureTransformer, sklearn_base.BaseEstimator)
    assert issubclass(SklearnFeatureTransformer, sklearn_base.TransformerMixin)
    trans = SklearnFeatureTransformer("gaf", bins=8)
    # fit_transform comes from TransformerMixin, not from our own definition.
    assert "fit_transform" not in vars(SklearnFeatureTransformer)
    assert trans.get_params()["bins"] == 8
    assert sklearn_base.clone(trans).get_params() == trans.get_params()


def test_sklearn_transformer_composes_in_a_pipeline() -> None:
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(0)
    X = np.stack([_sample_series() * s for s in rng.uniform(0.5, 2.0, size=12)])
    y = (X.std(axis=1) > np.median(X.std(axis=1))).astype(int)
    pipe = Pipeline(
        [
            ("feat", SklearnFeatureTransformer("gaf", bins=8)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    scores = cross_val_score(pipe, X, y, cv=3)
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_sklearn_transformer_validates_input() -> None:
    pytest.importorskip("sklearn")
    trans = SklearnFeatureTransformer("gaf", bins=8)
    with pytest.raises(ValueError, match="2D"):
        trans.fit(_sample_series())
    with pytest.raises(AttributeError, match="transform"):
        trans.get_feature_names_out()
    out = trans.fit_transform(np.stack([_sample_series() for _ in range(2)]))
    assert trans.get_feature_names_out().shape == (out.shape[1],)


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
    # Pin opset/IR instead of taking onnx's newest defaults: a newer `onnx`
    # paired with an older `onnxruntime` otherwise fails with
    # "Unsupported model IR version".
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    try:
        sess = onnxruntime.InferenceSession(model.SerializeToString())
    except Exception as exc:  # pragma: no cover - depends on runtime build
        pytest.skip(f"onnxruntime cannot load the test model: {exc}")
    result = sess.run(None, {})[0]
    assert np.allclose(result, arr.astype(np.float32))
