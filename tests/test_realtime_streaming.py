import numpy as np
from numpy.typing import NDArray

from tscv_vision.streaming import StreamingEncoder


def test_streaming_encoder_basic() -> None:
    enc = StreamingEncoder(size=4, hop=2)
    outs: list[NDArray[np.float64]] = []
    for s in range(8):
        outs.extend(enc.push(float(s)))
    assert len(outs) == 3
    assert outs[0].dtype == np.float64


def test_streaming_encoder_anomaly() -> None:
    triggered: list[NDArray[np.float64]] = []

    def on_anom(img: NDArray[np.float64]) -> None:
        triggered.append(img)

    enc = StreamingEncoder(size=4, anomaly_threshold=-1.1, on_anomaly=on_anom)
    for _ in range(4):
        enc.push(1.0)
    assert triggered, "anomaly callback was not invoked"


def test_streaming_encoder_adaptive() -> None:
    def adapt(buf: NDArray[np.float64]) -> int:
        return 2 if np.std(buf) < 0.1 else 4

    enc = StreamingEncoder(size=4, hop=2, adaptive=adapt)
    for _ in range(6):
        enc.push(0.0)
    assert enc.size == 2


def test_incremental_update() -> None:
    calls = {"full": 0, "inc": 0}

    def full(win: NDArray[np.float64]) -> NDArray[np.float64]:
        calls["full"] += 1
        return np.array([win.sum()])

    def inc(prev: NDArray[np.float64], old: float, new: float) -> NDArray[np.float64]:
        calls["inc"] += 1
        return np.array([prev[0] - old + new])

    enc = StreamingEncoder(size=3, hop=1, encode_fn=full, incremental=inc)
    outs: list[NDArray[np.float64]] = []
    for s in [1.0, 2.0, 3.0, 4.0]:
        outs.extend(enc.push(s))
    assert calls["full"] == 1
    assert calls["inc"] == 1
    assert outs[0][0] == 6.0
    assert outs[1][0] == 9.0


def test_gpu_missing() -> None:
    enc = StreamingEncoder(size=4, use_gpu=True)
    for i in range(4):
        enc.push(float(i))
    assert not enc.use_gpu


def test_adaptive_precision() -> None:
    enc = StreamingEncoder(size=2, precision="adaptive", latency_threshold=0.0)
    for s in [1.0, 2.0, 3.0]:
        enc.push(s)
    assert enc.dtype == np.float16

