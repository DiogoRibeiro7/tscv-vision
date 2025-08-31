"""Streaming utilities for online feature extraction."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from time import perf_counter
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .sliding import EncoderLike, _encode_window_static

Array = NDArray[np.float64]


def stream_windows(samples: Iterable[float], size: int, hop: int | None = None) -> Iterator[Array]:
    """Yield sliding windows from a sample stream."""

    real_hop = hop if hop is not None else max(1, size // 2)
    buf: list[float] = []
    for s in samples:
        buf.append(float(s))
        if len(buf) >= size:
            win = np.array(buf[:size], dtype=float)
            yield win
            del buf[:real_hop]


def online_encode(
    samples: Iterable[float],
    *,
    encoder: EncoderLike = "gaf",
    size: int,
    hop: int | None = None,
    metric: str = "euclidean",
    eps: float | None = None,
    spec_win: int | None = None,
    spec_hop: int | None = None,
    spec_window: str = "hann",
    channel_fusion: str = "stack",
    cwt_scales: Array | None = None,
) -> Iterator[Array]:
    """Stream-encode windows from ``samples`` as data arrives."""

    for w in stream_windows(samples, size, hop):
        yield _encode_window_static(
            w,
            encoder=encoder,
            size=size,
            metric=metric,  # type: ignore[arg-type]
            eps=eps,
            spec_win=spec_win,
            spec_hop=spec_hop,
            spec_window=spec_window,  # type: ignore[arg-type]
            channel_fusion=channel_fusion,  # type: ignore[arg-type]
            cwt_scales=cwt_scales,
        )


class StreamingEncoder:
    """Low-latency encoder with adaptive buffering and precision control.

    Parameters
    ----------
    encoder:
        Encoder to apply to each completed window.
    size:
        Initial window size.
    hop:
        Step between windows; defaults to ``size // 2``.
    adaptive:
        Optional function ``adaptive(buf) -> int`` returning a new window size
        based on the current buffer. The size is clamped to ``[8, max_size]``.
    anomaly_threshold:
        If set, an anomaly is triggered when the mean pixel value of an encoded
        window exceeds this threshold. ``on_anomaly`` is then invoked.
    on_anomaly:
        Callback receiving the anomalous image.
    dtype:
        Output dtype used for encoded images. Use ``np.float16`` for
        quantization on edge devices.
    max_size:
        Maximum allowed window size when adapting.
    incremental:
        Optional function ``update(prev, old, new)`` returning an updated
        encoded image given the previous image and the outgoing and incoming
        samples. When provided and ``hop == 1`` the encoder avoids full
        recomputation.
    use_gpu:
        If ``True`` and ``cupy`` is installed, selected encoders run on the GPU.
    precision:
        ``"high"`` (``float64``), ``"medium"`` (``float32``), ``"low``"
        (``float16``), or ``"adaptive"`` which dynamically scales between them
        based on ``latency_threshold``.
    latency_threshold:
        Target seconds-per-window for ``"adaptive"`` precision. Exceeding this
        degrades precision; consistently staying below half upgrades precision.
    """

    def __init__(
        self,
        *,
        encoder: EncoderLike = "gaf",
        size: int,
        hop: int | None = None,
        adaptive: Callable[[Array], int] | None = None,
        anomaly_threshold: float | None = None,
        on_anomaly: Callable[[Array], None] | None = None,
        dtype: np.dtype[Any] | type[np.floating[Any]] | None = None,
        max_size: int | None = None,
        incremental: Callable[[Array, float, float], Array] | None = None,
        use_gpu: bool = False,
        precision: Literal["high", "medium", "low", "adaptive"] = "high",
        latency_threshold: float = 0.01,
        encode_fn: Callable[[Array], Array] | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.encoder = encoder
        self.size = size
        self.hop = hop if hop is not None else max(1, size // 2)
        self.adaptive = adaptive
        self.anomaly_threshold = anomaly_threshold
        self.on_anomaly = on_anomaly
        level_map = {"high": 0, "medium": 1, "low": 2}
        dtypes: list[np.dtype[np.floating[Any]]] = [
            np.dtype(np.float64),
            np.dtype(np.float32),
            np.dtype(np.float16),
        ]
        if dtype is not None:
            self.dtype = np.dtype(dtype)
            self._precision_level = level_map.get(precision, 0)
        else:
            if precision == "adaptive":
                self.dtype = np.dtype(np.float64)
                self._precision_level = 0
            else:
                self._precision_level = level_map[precision]
                self.dtype = dtypes[self._precision_level]
        self.max_size = max_size or size * 4
        self._buf: deque[float] = deque()
        self.incremental = incremental
        self._last_encoded: Array | None = None
        self.use_gpu = use_gpu
        self.precision = precision
        self.latency_threshold = latency_threshold
        self.encode_fn = encode_fn
        self._prev_first: float | None = None

    def _maybe_adapt(self) -> None:
        if self.adaptive is None:
            return
        arr = np.fromiter(self._buf, dtype=float)
        if arr.size == 0:
            return
        new_size = int(self.adaptive(arr))
        new_size = int(np.clip(new_size, 2, self.max_size))
        if new_size != self.size:
            self.size = new_size
            self.hop = min(self.hop, self.size)

    def _adapt_precision(self, latency: float) -> None:
        if self.precision != "adaptive":
            return
        levels: list[np.dtype[np.floating[Any]]] = [
            np.dtype(np.float64),
            np.dtype(np.float32),
            np.dtype(np.float16),
        ]
        if latency > self.latency_threshold and self._precision_level < 2:
            self._precision_level += 1
        elif latency < self.latency_threshold / 2 and self._precision_level > 0:
            self._precision_level -= 1
        self.dtype = levels[self._precision_level]

    def _encode(self, win: Array) -> Array:
        if self.use_gpu:
            try:
                import cupy as cp
            except Exception as exc:  # pragma: no cover - optional path
                raise RuntimeError("CuPy is required for GPU acceleration") from exc
            if self.encoder == "gaf":
                w_cp = cp.asarray(win)
                phi = cp.arccos(cp.clip(w_cp, -1.0, 1.0))
                img = cp.cos(phi[:, None] + phi[None, :])
                return np.asarray(cp.asnumpy(img), dtype=float)
            # fallback to CPU for unsupported encoders
        if self.encode_fn is not None:
            return self.encode_fn(win)
        return _encode_window_static(
            win,
            encoder=self.encoder,
            size=self.size,
            metric="euclidean",
            eps=None,
            spec_win=None,
            spec_hop=None,
            spec_window="hann",
            channel_fusion="stack",
            cwt_scales=None,
        )

    def push(self, sample: float) -> list[Array]:
        """Process a new sample and return encoded windows if available."""

        self._buf.append(float(sample))
        outputs: list[Array] = []
        while len(self._buf) >= self.size:
            start = perf_counter()
            current_first = float(self._buf[0])
            if (
                self.incremental is not None
                and self._last_encoded is not None
                and self.hop == 1
                and self._prev_first is not None
            ):
                new = float(self._buf[self.size - 1])
                encoded = self.incremental(self._last_encoded, self._prev_first, new)
            else:
                win = np.fromiter(islice(self._buf, 0, self.size), dtype=float)
                encoded = self._encode(win)
            latency = perf_counter() - start
            self._adapt_precision(latency)
            img = encoded.astype(self.dtype, copy=False)
            self._last_encoded = img
            self._prev_first = current_first
            outputs.append(img)
            if self.anomaly_threshold is not None and float(np.mean(img)) > self.anomaly_threshold:
                if self.on_anomaly is not None:
                    self.on_anomaly(img)
                self.size = max(8, self.size // 2)
                self.hop = max(1, self.hop // 2)
            for _ in range(self.hop):
                if self._buf:
                    self._buf.popleft()
            self._maybe_adapt()
        return outputs


def kafka_stream(
    topic: str, bootstrap_servers: str, group_id: str | None = None
) -> Iterator[float]:
    """Yield samples from a Kafka topic.

    Requires ``kafka-python``; raises ``RuntimeError`` if not installed.
    """

    try:
        from kafka import KafkaConsumer
    except Exception as exc:  # pragma: no cover - optional path
        raise RuntimeError("kafka-python is required for kafka_stream") from exc
    consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers, group_id=group_id)
    for msg in consumer:
        yield float(msg.value)


def redis_stream(key: str, host: str = "localhost", port: int = 6379) -> Iterator[float]:
    """Yield samples from a Redis stream.

    Requires ``redis``; raises ``RuntimeError`` if not installed.
    """

    try:
        import redis
    except Exception as exc:  # pragma: no cover - optional path
        raise RuntimeError("redis-py is required for redis_stream") from exc
    client = redis.Redis(host=host, port=port)
    last_id = "$"
    while True:  # pragma: no cover - infinite generator
        resp = client.xread({key: last_id}, block=0)
        if resp:
            _, messages = resp[0]
            for msg_id, fields in messages:
                last_id = msg_id.decode()
                yield float(fields[b"value"])


def serve_websocket(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - stub
    """Start a WebSocket server for real-time feature serving.

    This is a lightweight placeholder; a full implementation requires the
    ``websockets`` package and asyncio event loop management.
    """
    raise NotImplementedError("WebSocket serving requires optional dependencies")


def serve_grpc(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - stub
    """Start a gRPC server for real-time feature serving.

    A full implementation would depend on ``grpcio`` and protocol definitions.
    """
    raise NotImplementedError("gRPC serving requires optional dependencies")


__all__ = [
    "stream_windows",
    "online_encode",
    "StreamingEncoder",
    "kafka_stream",
    "redis_stream",
    "serve_websocket",
    "serve_grpc",
]

