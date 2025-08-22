import numpy as np
import pytest

from tscv_vision.neural import (
    VAE,
    AttentionFusion,
    TorchCNNEncoder,
    ViTAdapter,
    contrastive_loss,
    nas_random_search,
    style_transfer,
)

torch = pytest.importorskip("torch")


def test_cnn_encoder_outputs_vector() -> None:
    enc = TorchCNNEncoder()
    rng = np.random.default_rng(0)
    img = rng.random((32, 32), dtype=np.float32)
    vec = enc.encode(img)
    assert vec.shape == (64,)


def test_vit_adapter_outputs_vector() -> None:
    enc = ViTAdapter(dim=32)
    rng = np.random.default_rng(1)
    img = rng.random((32, 32), dtype=np.float32)
    vec = enc.encode(img)
    assert vec.shape == (32,)


def test_attention_fusion() -> None:
    fusion = AttentionFusion(n_inputs=2, feat_dim=16)
    t1 = torch.randn(1, 16)
    t2 = torch.randn(1, 16)
    out = fusion([t1, t2])
    assert out.shape == (1, 16)


def test_style_transfer_shape() -> None:
    rng = np.random.default_rng(2)
    img = rng.random((16, 16), dtype=np.float32)
    out = style_transfer(img, img, steps=1)
    assert out.shape == (16, 16)


def test_vae_sampling() -> None:
    vae = VAE(latent_dim=4)
    samples = vae.sample(2)
    assert samples.shape == (2, 1, 32, 32)


def test_contrastive_loss_positive() -> None:
    z1 = torch.randn(4, 8)
    z2 = torch.randn(4, 8)
    loss = contrastive_loss(z1, z2)
    assert float(loss) > 0


def test_nas_random_search() -> None:
    space = [1, 2, 3]
    def eval_fn(x: int) -> float:  # simple decreasing function
        return float(4 - x)
    best = nas_random_search(space, eval_fn)
    assert best == 3
