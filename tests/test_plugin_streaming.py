import numpy as np

from tscv_vision import encoders, sliding, streaming


def test_custom_encoder_registration():
    def my_enc(x: np.ndarray) -> np.ndarray:
        return np.outer(x, x)

    encoders.register_encoder("outer", my_enc)
    x = np.arange(5.0)
    imgs, _ = sliding.encode_sliding(x, encoder="outer", size=5, hop=5)
    assert imgs.shape == (1, 5, 5)
    assert np.allclose(imgs[0], np.outer(x, x))


def test_online_encode_matches_batch():
    x = np.arange(10.0)
    batch, _ = sliding.encode_sliding(x, size=4, hop=2)
    streamed = list(streaming.online_encode(x, size=4, hop=2))
    assert len(streamed) == batch.shape[0]
    for a, b in zip(batch, streamed, strict=True):
        assert np.allclose(a, b)

