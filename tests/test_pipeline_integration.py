from __future__ import annotations

import numpy as np

from tscv_vision import features
from tscv_vision.sliding import encode_sliding, features_for_sliding


def test_pipeline_raw_to_features() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=256)
    imgs, starts = encode_sliding(x, encoder="gaf", size=64, hop=32)
    sel = ["intensity", "hist", "gradient", "lbp"]
    vecs = features.extract_batch(imgs, bins=8, selected=sel)
    feats2, starts2 = features_for_sliding(
        x, encoder="gaf", size=64, hop=32, bins=8, feature_names=sel
    )
    np.testing.assert_allclose(vecs, feats2)
    np.testing.assert_array_equal(starts, starts2)
    assert vecs.shape[1] == 6 + 8 + 16 + 256
