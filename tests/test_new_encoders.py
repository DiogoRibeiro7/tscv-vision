import numpy as np

from tscv_vision import encoders


def test_mtf_shape_and_bounds() -> None:
    x = np.linspace(0, 1, 16)
    img = encoders.mtf(x, bins=4)
    assert img.shape == (16, 16)
    assert np.all(img >= 0) and np.all(img <= 1)


def test_multi_scale_rp_stack() -> None:
    x = np.sin(np.linspace(0, 2 * np.pi, 32))
    imgs = encoders.multi_scale_rp(x, scales=[1, 2, 4])
    assert imgs.shape == (3, 32, 32)


def test_dtw_matrix_properties() -> None:
    x = np.random.RandomState(0).randn(10)
    img = encoders.dtw_matrix(x)
    assert img.shape == (10, 10)
    assert np.allclose(img, img.T)
    assert np.all(img >= 0) and np.all(img <= 1)


def test_sax_shape() -> None:
    x = np.sin(np.linspace(0, 1, 32))
    img = encoders.sax(x, segments=8, alphabet=5)
    assert img.shape == (8, 8)


def test_ensemble_stack() -> None:
    x = np.sin(np.linspace(0, 1, 16))
    img = encoders.ensemble(x, names=["gaf", "rp"])
    assert img.shape == (2, 16, 16)


def test_persistence_image_shape() -> None:
    x = np.sin(np.linspace(0, 2 * np.pi, 64))
    img = encoders.persistence_image(x, bins=16)
    assert img.shape == (16, 16)


def test_randproj_deterministic() -> None:
    x = np.linspace(0, 1, 16)
    img1 = encoders.random_projection_image(x, size=4, seed=0)
    img2 = encoders.random_projection_image(x, size=4, seed=0)
    assert img1.shape == (4, 4)
    assert np.allclose(img1, img2)

