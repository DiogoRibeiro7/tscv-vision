import numpy as np
import pytest

from tscv_vision import encoders
from tscv_vision.gpu import encoders as gpu_enc

cupy = pytest.importorskip("cupy")

pytestmark = [pytest.mark.gpu, pytest.mark.optional]


def test_gaf_gpu_matches_cpu():
    x = np.linspace(-1, 1, 32)
    cpu = encoders.gaf(x)
    gpu = gpu_enc.gaf(x)
    assert np.allclose(cpu, gpu, atol=1e-6)


def test_spectrogram_gpu_matches_cpu():
    x = np.sin(np.linspace(0, 8 * np.pi, 128))
    cpu = encoders.spectrogram(x, win=32, hop=16)
    gpu = gpu_enc.spectrogram(x, win=32, hop=16)
    assert np.allclose(cpu, gpu, atol=1e-6)
