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


def _scale_rows(X: np.ndarray) -> np.ndarray:
    """Per-row min-max scale to [-1, 1] (matches CPU gaf's internal scaling)."""
    xmin = X.min(axis=1, keepdims=True)
    xmax = X.max(axis=1, keepdims=True)
    span = xmax - xmin
    safe = np.where(span == 0, 1.0, span)
    out = (X - xmin) / safe * 2.0 - 1.0
    return np.where(span == 0, 0.0, out)


def test_gaf_batch_matches_per_window(monkeypatch):
    gpu_enc = importlib.import_module("tscv_vision.gpu.encoders")
    monkeypatch.setattr(gpu_enc, "cp", FakeCP())
    monkeypatch.setattr(gpu_enc, "_require_cupy", lambda: None)
    rng = np.random.default_rng(0)
    X = _scale_rows(rng.uniform(-1.0, 1.0, size=(5, 16)))
    out = gpu_enc.gaf_batch(X)
    expected = np.stack([encoders.gaf(row) for row in X])
    np.testing.assert_allclose(out, expected, atol=1e-12)
    diff = gpu_enc.gaf_batch(X, method="difference")
    expected_diff = np.stack(
        [encoders.gaf(row, method="difference") for row in X]
    )
    np.testing.assert_allclose(diff, expected_diff, atol=1e-12)


def test_gaf_batch_validates_shape_and_method(monkeypatch):
    gpu_enc = importlib.import_module("tscv_vision.gpu.encoders")
    monkeypatch.setattr(gpu_enc, "cp", FakeCP())
    monkeypatch.setattr(gpu_enc, "_require_cupy", lambda: None)
    with pytest.raises(ValueError):
        gpu_enc.gaf_batch(np.zeros(8))
    with pytest.raises(ValueError):
        gpu_enc.gaf_batch(np.zeros((2, 4)), method="bogus")
    with pytest.raises(ValueError):
        gpu_enc.gaf_batch(np.zeros((2, 4)), mem_limit=0)


def test_gaf_batch_mem_limit_chunks(monkeypatch):
    gpu_enc = importlib.import_module("tscv_vision.gpu.encoders")
    fake_cp = FakeCP()
    monkeypatch.setattr(gpu_enc, "cp", fake_cp)
    monkeypatch.setattr(gpu_enc, "_require_cupy", lambda: None)
    X = _scale_rows(np.linspace(-1.0, 1.0, 32).reshape(8, 4))
    mem_limit = 4 * 4 * 8  # fits one window per chunk
    out = gpu_enc.gaf_batch(X, mem_limit=mem_limit)
    expected = np.stack([encoders.gaf(row) for row in X])
    np.testing.assert_allclose(out, expected, atol=1e-12)
    assert fake_cp.cos_calls == X.shape[0]


def test_encode_sliding_use_gpu_true(monkeypatch):
    gpu_enc = importlib.import_module("tscv_vision.gpu.encoders")
    monkeypatch.setattr(gpu_enc, "cp", FakeCP())
    monkeypatch.setattr(gpu_enc, "_require_cupy", lambda: None)
    sliding = importlib.import_module("tscv_vision.sliding")
    x = np.sin(np.linspace(0.0, 4 * np.pi, 128))
    cpu_imgs, cpu_starts = sliding.encode_sliding(x, size=32, hop=16)
    gpu_imgs, gpu_starts = sliding.encode_sliding(
        x, size=32, hop=16, use_gpu=True
    )
    np.testing.assert_allclose(gpu_imgs, cpu_imgs, atol=1e-12)
    np.testing.assert_array_equal(gpu_starts, cpu_starts)


def test_encode_sliding_use_gpu_auto_skips_small(monkeypatch):
    gpu_enc = importlib.import_module("tscv_vision.gpu.encoders")

    def _should_not_be_called(*args, **kwargs):  # pragma: no cover - assertion
        raise AssertionError("gaf_batch should not be called for small inputs")

    monkeypatch.setattr(gpu_enc, "gaf_batch", _should_not_be_called)
    sliding = importlib.import_module("tscv_vision.sliding")
    x = np.sin(np.linspace(0.0, 4 * np.pi, 128))
    imgs, _ = sliding.encode_sliding(x, size=16, hop=8, use_gpu="auto")
    assert imgs.shape[0] > 0  # CPU path ran


def test_encode_sliding_use_gpu_auto_takes_large(monkeypatch):
    gpu_enc = importlib.import_module("tscv_vision.gpu.encoders")
    calls = {"n": 0}

    def _tracking_batch(X, **kwargs):
        calls["n"] += 1
        return np.stack([encoders.gaf(row, method=kwargs.get("method", "summation"))
                         for row in X])

    monkeypatch.setattr(gpu_enc, "gaf_batch", _tracking_batch)
    sliding = importlib.import_module("tscv_vision.sliding")
    # threshold is 10 MiB = n_windows * size * size * 8
    # size=128 -> n_windows needed >= 10*1024*1024/(128*128*8) ~= 80
    n = 128 * 100  # plenty of windows at hop=128
    x = np.sin(np.linspace(0.0, 20 * np.pi, n))
    imgs, _ = sliding.encode_sliding(x, size=128, hop=128, use_gpu="auto")
    assert calls["n"] == 1
    assert imgs.shape[1:] == (128, 128)


def test_encode_sliding_use_gpu_falls_back_on_runtimeerror(monkeypatch):
    gpu_enc = importlib.import_module("tscv_vision.gpu.encoders")

    def _raising(*args, **kwargs):
        raise RuntimeError("simulated GPU OOM")

    monkeypatch.setattr(gpu_enc, "gaf_batch", _raising)
    sliding = importlib.import_module("tscv_vision.sliding")
    x = np.sin(np.linspace(0.0, 4 * np.pi, 128))
    imgs_gpu, _ = sliding.encode_sliding(x, size=32, hop=16, use_gpu=True)
    imgs_cpu, _ = sliding.encode_sliding(x, size=32, hop=16)
    np.testing.assert_allclose(imgs_gpu, imgs_cpu, atol=1e-12)


def test_encode_sliding_use_gpu_multichannel_falls_back(monkeypatch):
    gpu_enc = importlib.import_module("tscv_vision.gpu.encoders")

    def _should_not_be_called(*args, **kwargs):  # pragma: no cover - assertion
        raise AssertionError("gaf_batch should not run for multichannel input")

    monkeypatch.setattr(gpu_enc, "gaf_batch", _should_not_be_called)
    sliding = importlib.import_module("tscv_vision.sliding")
    t = np.linspace(0.0, 2 * np.pi, 128)
    x = np.column_stack([np.sin(t), np.cos(t)])
    imgs, _ = sliding.encode_sliding(x, size=32, hop=32, use_gpu=True)
    assert imgs.shape[0] > 0  # CPU path produced output


def test_encode_sliding_use_gpu_non_gaf_encoder_falls_back(monkeypatch):
    gpu_enc = importlib.import_module("tscv_vision.gpu.encoders")

    def _should_not_be_called(*args, **kwargs):  # pragma: no cover - assertion
        raise AssertionError("gaf_batch should not run for non-gaf encoder")

    monkeypatch.setattr(gpu_enc, "gaf_batch", _should_not_be_called)
    sliding = importlib.import_module("tscv_vision.sliding")
    x = np.sin(np.linspace(0.0, 4 * np.pi, 128))
    imgs, _ = sliding.encode_sliding(
        x, encoder="rp", size=32, hop=16, use_gpu=True
    )
    assert imgs.shape[0] > 0
