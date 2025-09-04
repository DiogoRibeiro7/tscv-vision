"""CuPy-based GPU implementations of select encoders and kernels."""
from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray

try:  # optional dependency
    import cupy as cp
except Exception:  # pragma: no cover - optional path
    cp = None

Array = NDArray[np.float64]


def _require_cupy() -> None:
    if cp is None:  # pragma: no cover - checked at runtime
        raise RuntimeError("CuPy is required for GPU acceleration")


def memory_usage(device: int | None = None) -> tuple[int, int]:
    """Return the current and total GPU memory in bytes."""

    _require_cupy()
    with cp.cuda.Device(device if device is not None else 0):
        free, total = cp.cuda.runtime.memGetInfo()
    used = total - free
    return int(used), int(total)


def gaf(
    x: Array,
    *,
    method: str = "summation",
    device: int | None = None,
    mem_limit: int | None = None,
) -> Array:
    """GPU Gramian Angular Field encoder."""

    _require_cupy()
    summation = method == "summation"
    with cp.cuda.Device(device if device is not None else 0):
        x_cp = cp.asarray(x, dtype=cp.float64)
        phi = cp.arccos(cp.clip(x_cp, -1.0, 1.0))
        n = x_cp.size
        out = cp.empty((n, n), dtype=cp.float64)
        if mem_limit is not None:
            if mem_limit <= 0:
                raise ValueError("mem_limit must be positive")
            row_bytes = n * 8
            rows = max(1, mem_limit // row_bytes)
            for start in range(0, n, rows):
                end = min(n, start + rows)
                if summation:
                    out[start:end] = cp.cos(phi[start:end, None] + phi[None, :])
                else:
                    out[start:end] = cp.sin(phi[start:end, None] - phi[None, :])
        else:
            if summation:
                out = cp.cos(phi[:, None] + phi[None, :])
            else:
                out = cp.sin(phi[:, None] - phi[None, :])
        return cast(Array, cp.asnumpy(out))


def spectrogram(
    x: Array,
    win: int,
    hop: int,
    *,
    window: str = "hann",
    device: int | None = None,
) -> Array:
    """GPU STFT-based magnitude spectrogram."""

    _require_cupy()
    if hop <= 0 or win < 8 or hop > win:
        raise ValueError("invalid window or hop length")
    with cp.cuda.Device(device if device is not None else 0):
        x_cp = cp.asarray(x, dtype=cp.float64)
        if window == "hann":
            w = 0.5 - 0.5 * cp.cos(2 * cp.pi * cp.arange(win) / win)
        elif window == "rect":
            w = cp.ones(win)
        else:
            raise ValueError("window must be 'hann' or 'rect'")
        n_frames = 1 + math.ceil((x_cp.size - win) / hop)
        total_len = (n_frames - 1) * hop + win
        if x_cp.size < total_len:
            x_cp = cp.pad(x_cp, (0, total_len - x_cp.size))
        view = cp.lib.stride_tricks.as_strided(
            x_cp,
            shape=(n_frames, win),
            strides=(x_cp.strides[0] * hop, x_cp.strides[0]),
            writeable=False,
        )
        frames = view * w[None, :]
        fft = cp.fft.rfft(frames, n=win, axis=1)
        mag = cp.abs(fft).T
        mag = mag / (cp.max(mag) + 1e-12)
        return cast(Array, cp.asnumpy(mag))


def convolve2d(
    img: Array,
    kernel: Array,
    *,
    device: int | None = None,
) -> Array:
    """2D convolution on the GPU using ``cupyx.scipy.ndimage``."""

    _require_cupy()
    from cupyx.scipy import ndimage

    with cp.cuda.Device(device if device is not None else 0):
        img_cp = cp.asarray(img, dtype=cp.float64)
        ker_cp = cp.asarray(kernel, dtype=cp.float64)
        out = ndimage.convolve(img_cp, ker_cp, mode="constant", cval=0.0)
        return cast(Array, cp.asnumpy(out))


def lbp(img: Array, *, device: int | None = None) -> Array:
    """Local Binary Pattern on the GPU."""

    _require_cupy()
    with cp.cuda.Device(device if device is not None else 0):
        img_cp = cp.asarray(img, dtype=cp.float64)
        h, w = img_cp.shape
        out = cp.zeros((h - 2, w - 2), dtype=cp.float64)
        center = img_cp[1:-1, 1:-1]
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
        ]
        for i, (dy, dx) in enumerate(offsets):
            neigh = img_cp[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
            out += ((neigh >= center) << i).astype(cp.float64)
        return cast(Array, cp.asnumpy(out))


__all__ = ["gaf", "spectrogram", "convolve2d", "lbp", "memory_usage"]
