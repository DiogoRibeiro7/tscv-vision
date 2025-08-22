import numpy as np
from numpy.typing import NDArray

from tscv_vision.streaming import StreamingEncoder


def test_streaming_encoder_basic() -> None:
    enc = StreamingEncoder(size=4, hop=2)
    outs: list[NDArray[np.float64]] = []
    for s in range(8):
        outs.extend(enc.push(float(s)))
    assert len(outs) == 3
    assert outs[0].dtype == np.float32


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

