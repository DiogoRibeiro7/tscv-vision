import numpy as np
import pytest

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


def test_representation_alignment_linear_cka_invariances():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(16, 5))
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))

    assert analysis.representation_alignment(z, z) == pytest.approx(1.0)
    assert analysis.representation_alignment(z, 7.0 * z @ rotation) == pytest.approx(
        1.0
    )


def test_representation_similarity_and_redundancy():
    rng = np.random.default_rng(1)
    first = rng.normal(size=(20, 4))
    second = first + 0.01 * rng.normal(size=(20, 4))
    unrelated = rng.normal(size=(20, 4))

    matrix = analysis.representation_similarity(
        {"first": first, "second": second, "unrelated": unrelated}
    )

    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    assert matrix[0, 1] > matrix[0, 2]
    assert 0.0 <= analysis.representation_redundancy([first, unrelated]) <= 1.0


def test_representation_effective_rank_detects_collapse():
    x = np.arange(12.0)
    collapsed = np.column_stack([x, 2.0 * x, -x])
    full = np.eye(6)

    assert analysis.representation_effective_rank(np.ones((6, 3))) == 0.0
    assert analysis.representation_effective_rank(collapsed) == pytest.approx(1.0)
    assert analysis.representation_effective_rank(full) > 1.0


def test_representation_complementarity_reports_pairwise_gain():
    rows = analysis.representation_complementarity(
        {"gaf": 0.72, "cwt": 0.75},
        {("gaf", "cwt"): 0.81},
    )

    assert rows == [
        {
            "left": "gaf",
            "right": "cwt",
            "fused_score": 0.81,
            "best_individual_score": 0.75,
            "improvement": pytest.approx(0.06),
        }
    ]

