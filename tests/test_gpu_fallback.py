import importlib
import sys
import types

import numpy as np
import pytest

# Bypass heavy package __init__ by installing a lightweight placeholder
pkg = types.ModuleType("tscv_vision")
pkg.__path__ = ["src/tscv_vision"]
sys.modules.setdefault("tscv_vision", pkg)

encoders = importlib.import_module("tscv_vision.encoders")
streaming = importlib.import_module("tscv_vision.streaming")
StreamingEncoder = streaming.StreamingEncoder


class FakeDevice:
    def __init__(self, device: int | None = None) -> None:  # pragma: no cover - trivial
        self.device = device

    def __enter__(self) -> None:  # pragma: no cover - trivial
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - trivial
        return False


class FakeCP:
    float64 = np.float64

    def __init__(self) -> None:
        from types import SimpleNamespace

        runtime = SimpleNamespace(memGetInfo=lambda: (0, 0))
        self.cuda = SimpleNamespace(Device=FakeDevice, runtime=runtime)
        self.cos_calls = 0

    def asarray(self, arr, dtype=None):
        return np.asarray(arr, dtype=dtype)

    def arccos(self, x):
        return np.arccos(x)

    def clip(self, x, a, b):
        return np.clip(x, a, b)

    def empty(self, shape, dtype=None):
        return np.empty(shape, dtype=dtype)

    def cos(self, x):
        self.cos_calls += 1
        return np.cos(x)

    def sin(self, x):
        self.cos_calls += 1
        return np.sin(x)

    def asnumpy(self, x):
        return np.asarray(x)


def test_gpu_module_requires_cupy(monkeypatch):
    gpu_enc = importlib.import_module("tscv_vision.gpu.encoders")
    monkeypatch.setattr(gpu_enc, "cp", None)
    x = np.linspace(-1, 1, 8)
    with pytest.raises(RuntimeError):
        gpu_enc.gaf(x)


def test_encoders_fallback_without_cupy(monkeypatch):
    gpu_enc = importlib.import_module("tscv_vision.gpu.encoders")
    monkeypatch.setattr(gpu_enc, "cp", None)
    x = np.linspace(-1, 1, 8)
    cpu = encoders.gaf(x)
    gpu_flag = encoders.gaf(x, use_gpu=True)
    np.testing.assert_allclose(cpu, gpu_flag)

    x2 = np.sin(np.linspace(0, np.pi, 32))
    cpu_spec = encoders.spectrogram(x2, win=16, hop=8)
    gpu_spec = encoders.spectrogram(x2, win=16, hop=8, use_gpu=True)
    np.testing.assert_allclose(cpu_spec, gpu_spec)


def test_streaming_encoder_fallback(monkeypatch):
    gpu_enc = importlib.import_module("tscv_vision.gpu.encoders")
    monkeypatch.setattr(gpu_enc, "cp", None)
    data = np.linspace(-1, 1, 8)
    se_gpu = StreamingEncoder(encoder="gaf", size=8, use_gpu=True)
    se_cpu = StreamingEncoder(encoder="gaf", size=8)
    out_gpu: list[np.ndarray] = []
    out_cpu: list[np.ndarray] = []
    for val in data:
        out_gpu.extend(se_gpu.push(float(val)))
        out_cpu.extend(se_cpu.push(float(val)))
    assert len(out_gpu) == 1 and len(out_cpu) == 1
    np.testing.assert_allclose(out_gpu[0], out_cpu[0])


def test_gaf_mem_limit_enforced(monkeypatch):
    gpu_enc = importlib.import_module("tscv_vision.gpu.encoders")
    fake_cp = FakeCP()
    monkeypatch.setattr(gpu_enc, "cp", fake_cp)
    monkeypatch.setattr(gpu_enc, "_require_cupy", lambda: None)
    x = np.linspace(-1, 1, 16)
    out = gpu_enc.gaf(x, mem_limit=64)
    exp = encoders.gaf(x)
    np.testing.assert_allclose(out, exp, atol=1e-12)
    assert fake_cp.cos_calls > 1
