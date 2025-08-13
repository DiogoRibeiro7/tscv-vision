from __future__ import annotations
from typing import Literal, Tuple
import numpy as np
from . import encoders as _enc

Array = np.ndarray


def sliding_windows(x: Array, size: int, hop: int | None = None, *, copy: bool = False) -> Array:
    """Return a 2D view of 1D series as overlapping windows.

    Uses stride tricks for O(1) creation when ``copy=False``.

    Args:
        x: Input 1D series of length ``N``.
        size: Window length (``size >= 2``).
        hop: Step between starts (defaults to ``size // 2``).
        copy: If True, returns a compact copy; if False, a read-only view.

    Returns:
        Array with shape (n_windows, size).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("sliding_windows expects 1D input")
    if size < 2:
        raise ValueError("size must be >= 2")
    if hop is None:
        hop = max(1, size // 2)
    n = x.shape[0]
    n_win = 1 + max(0, (n - size) // hop)
    if n_win <= 0:
        return np.empty((0, size), dtype=float)
    view = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_win, size),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    )
    return view.copy() if copy else view


def encode_sliding(
    x: Array,
    encoder: Literal["gaf", "gadf", "rp", "spec"] = "gaf",
    *,
    size: int,
    hop: int | None = None,
    metric: Literal["euclidean", "manhattan"] = "euclidean",
    eps: float | None = None,
    spec_win: int | None = None,
    spec_hop: int | None = None,
    spec_window: Literal["hann", "rect"] = "hann",
) -> Tuple[Array, Array]:
    """Encode overlapping windows from a 1D series into a stack of images.

    For GAF/GADF/RP: each window of length ``size`` -> (size,size) image.
    For spectrogram: per-window STFT with ``spec_win`` (defaults to ``size``).

    Returns (images, starts).
    """
    x = np.asarray(x, dtype=float)
    win_view = sliding_windows(x, size=size, hop=hop, copy=False)
    n_windows = win_view.shape[0]
    if n_windows == 0:
        return np.zeros((0, size, size), dtype=float), np.zeros((0,), dtype=int)

    real_hop = hop if hop is not None else max(1, size // 2)
    starts = np.arange(n_windows, dtype=int) * real_hop

    imgs: list[Array] = []
    if encoder in {"gaf", "gadf"}:
        method = "summation" if encoder == "gaf" else "difference"
        for w in win_view:
            imgs.append(_enc.gaf(w, method=method))
        return np.stack(imgs, axis=0), starts

    if encoder == "rp":
        for w in win_view:
            imgs.append(_enc.recurrence_plot(w, metric=metric, eps=eps))
        return np.stack(imgs, axis=0), starts

    # spectrogram per window
    fwin = size if spec_win is None else spec_win
    for w in win_view:
        imgs.append(_enc.spectrogram(w, win=fwin, hop=spec_hop, window=spec_window))
    # Spectrograms may differ in T if very short; pad to the max T for stacking
    maxF = max(im.shape[0] for im in imgs)
    maxT = max(im.shape[1] for im in imgs)
    padded = []
    for im in imgs:
        padF = maxF - im.shape[0]
        padT = maxT - im.shape[1]
        padded.append(np.pad(im, ((0, padF), (0, padT)), mode="constant"))
    return np.stack(padded, axis=0), starts
