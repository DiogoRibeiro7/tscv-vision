import numpy as np

from tscv_vision import analysis


def test_variance_and_topk():
    feats = np.array([[1, 2, 3], [1, 4, 9], [1, 8, 27]], dtype=float)
    sel, mask = analysis.variance_threshold(feats, threshold=10)
    assert sel.shape[1] == 1 and mask.tolist() == [False, False, True]
    top, idx = analysis.topk_variance(feats, 2)
    assert top.shape[1] == 2 and idx.tolist() == [1, 2]


def test_feature_importance_corr():
    feats = np.array([[0, 1], [1, 0], [2, 1]], dtype=float)
    target = np.array([0, 1, 2], dtype=float)
    imp = analysis.feature_importance_corr(feats, target)
    assert imp.shape == (2,)
    assert np.all(imp >= 0)

