import numpy as np
import pytest

from tscv_vision.analytics import (
    TSHAPExplainer,
    gaf_attribution,
    generate_counterfactual,
    plot_importance,
    rp_attribution,
    spectrogram_attribution,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

pytestmark = pytest.mark.optional


def test_tshap_explainer_shapes_and_errors() -> None:
    def model(x: np.ndarray) -> float:
        return float(x.mean())

    explainer = TSHAPExplainer(model)
    t_imp, f_imp = explainer.explain(np.arange(8.0), window=4)
    assert t_imp.shape == (8,)
    assert f_imp.ndim == 1
    with pytest.raises(ValueError):
        explainer.explain(np.arange(5.0), window=0)


def test_attribution_mappings() -> None:
    mat = np.arange(9.0).reshape(3, 3)
    gaf = gaf_attribution(mat)
    rp = rp_attribution(mat)
    assert np.allclose(gaf, rp)
    spec = spectrogram_attribution(np.ones((4, 5)))
    assert spec.shape == (5,)
    with pytest.raises(ValueError):
        gaf_attribution(np.ones((2, 3)))


def test_plot_importance_returns_figure() -> None:
    fig = plot_importance(np.ones(5), np.zeros(5))
    assert hasattr(fig, "canvas")


def test_generate_counterfactual_changes_prediction() -> None:
    def model(x: np.ndarray) -> float:
        return float(x.mean())

    series = np.zeros(4)
    cf = generate_counterfactual(series, model, target=1.0, step=0.5, max_iter=50)
    assert model(cf) == pytest.approx(1.0, abs=1e-2)
