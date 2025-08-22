import pytest

from tscv_vision.neural import MambaEncoder, RetNetEncoder


def test_mamba_requires_torch() -> None:
    with pytest.raises(ImportError):
        MambaEncoder()


def test_retnet_requires_torch() -> None:
    with pytest.raises(ImportError):
        RetNetEncoder()
