"""Sliding-window utilities and batch encoders."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from . import encoders as _enc
from . import features
from .parallel import map_parallel

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]


def _encode_window_static(
    w: Array,
    *,
    encoder: Literal["gaf", "gadf", "rp", "spec"],
    size: int,
    metric: Literal["euclidean", "manhattan"],
    eps: float | None,
    spec_win: int | None,
    spec_hop: int | None,
    spec_window: Literal["hann", "rect"],
    channel_fusion: Literal["stack", "mean", "concat"],
) -> Array:
    series = [w] if w.ndim == 1 else [w[:, c] for c in range(w.shape[1])]
    ch_imgs: list[Array] = []
    if encoder in {"gaf", "gadf"}:
        method: Literal["summation", "difference"] = (
            "summation" if encoder == "gaf" else "difference"
        )
        for s in series:
            ch_imgs.append(_enc.gaf(s, method=method))
    elif encoder == "rp":
        for s in series:
            ch_imgs.append(_enc.recurrence_plot(s, metric=metric, eps=eps))
    else:
        fwin = size if spec_win is None else spec_win
        for s in series:
            ch_imgs.append(
                _enc.spectrogram(s, win=fwin, hop=spec_hop, window=spec_window)
            )

    if len(ch_imgs) == 1:
        fused = ch_imgs[0]
    elif channel_fusion == "stack":
        fused = np.stack(ch_imgs, axis=-1)
    elif channel_fusion == "mean":
        fused = np.mean(np.stack(ch_imgs, axis=0), axis=0)
    else:
        fused = np.concatenate(ch_imgs, axis=1)
    return fused


def sliding_windows(
    x: Array, size: int, hop: int | None = None, *, copy: bool = False
) -> Array:
    """Return overlapping windows of ``x`` using stride tricks.

    Parameters
    ----------
    x:
        Input 1D or 2D series of length ``N``.
    size:
        Window length ``>=2`` and ``<= N``.
    hop:
        Step between starts (defaults to ``size//2``).
    copy:
        If ``True`` return a compact copy; otherwise a read-only view.

    Returns
    -------
    Array
        View with shape ``(n_windows, size[, C])``.
    """

    x = np.asarray(x, dtype=float)
    if x.ndim not in {1, 2}:
        raise ValueError("sliding_windows expects 1D or 2D input")
    if size < 2:
        raise ValueError("size must be >= 2")
    if hop is None:
        hop = max(1, size // 2)
    n = x.shape[0]
    if size > n:
        raise ValueError("size cannot exceed length of x")
    n_win = 1 + max(0, (n - size) // hop)
    if x.ndim == 1:
        if n_win <= 0:
            return np.empty((0, size), dtype=float)
        view = np.lib.stride_tricks.as_strided(
            x,
            shape=(n_win, size),
            strides=(x.strides[0] * hop, x.strides[0]),
            writeable=False,
        )
        return view.copy() if copy else view
    # multichannel
    C = x.shape[1]
    if n_win <= 0:
        return np.empty((0, size, C), dtype=float)
    view = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_win, size, C),
        strides=(x.strides[0] * hop, x.strides[0], x.strides[1]),
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
    channel_fusion: Literal["stack", "mean", "concat"] = "stack",
    workers: int | None = None,
) -> tuple[Array, IntArray]:
    """Encode sliding windows from ``x`` into a stack of images.

    Returns
    -------
    images, starts:
        ``images`` has shape ``(n_windows, H, W[, C])``; ``starts`` is the start
        index of each window. If ``workers`` is greater than one, encoding is
        performed in parallel processes.
    """

    x = np.asarray(x, dtype=float)
    win_view = sliding_windows(x, size=size, hop=hop, copy=False)
    n_windows = win_view.shape[0]
    if n_windows == 0:
        return (
            np.zeros((0, size, size), dtype=float),
            np.zeros((0,), dtype=int),
        )

    real_hop = hop if hop is not None else max(1, size // 2)
    starts = np.arange(n_windows, dtype=int) * real_hop

    func = partial(
        _encode_window_static,
        encoder=encoder,
        size=size,
        metric=metric,
        eps=eps,
        spec_win=spec_win,
        spec_hop=spec_hop,
        spec_window=spec_window,
        channel_fusion=channel_fusion,
    )

    imgs = map_parallel(func, cast(Iterable[Array], win_view), workers)

    if encoder == "spec":
        maxF = max(im.shape[0] for im in imgs)
        maxT = max(im.shape[1] for im in imgs)
        padded: list[Array] = []
        for im in imgs:
            padF = maxF - im.shape[0]
            padT = maxT - im.shape[1]
            if im.ndim == 3:
                padded.append(np.pad(im, ((0, padF), (0, padT), (0, 0)), mode="constant"))
            else:
                padded.append(np.pad(im, ((0, padF), (0, padT)), mode="constant"))
        imgs = padded
    return np.stack(imgs, axis=0), starts


def features_for_sliding(
    x: Array,
    *,
    encoder: Literal["gaf", "gadf", "rp", "spec"] = "gaf",
    size: int,
    hop: int | None = None,
    bins: int = 32,
    metric: Literal["euclidean", "manhattan"] = "euclidean",
    eps: float | None = None,
    spec_win: int | None = None,
    spec_hop: int | None = None,
    spec_window: Literal["hann", "rect"] = "hann",
    channel_fusion: Literal["stack", "mean", "concat"] = "stack",
    feature_names: Iterable[str] | None = None,
    workers: int | None = None,
) -> tuple[Array, IntArray]:
    """Encode windows from ``x`` and extract feature vectors.

    Parameters
    ----------
    x:
        1D input series of length ``N``.
    encoder:
        Encoder name as accepted by :func:`encode_sliding`.
    size:
        Sliding window length.
    hop:
        Step between windows. Defaults to ``size//2``.
    bins:
        Histogram bins forwarded to :func:`features.extract_batch`.
    metric, eps, spec_win, spec_hop, spec_window:
        Additional parameters forwarded to :func:`encode_sliding`.
    channel_fusion:
        Channel fusion strategy for multichannel series.
    feature_names:
        Names of feature extractors from :mod:`tscv_vision.features`.
    workers:
        Number of parallel worker processes for encoding. ``None`` or ``1``
        runs sequentially.

    Returns
    -------
    features, starts:
        ``features`` has shape ``(n_windows, D)`` where ``D`` is the feature
        dimension. ``starts`` is an integer array of window start indices.
    """

    x = np.asarray(x, dtype=float)
    images, starts = encode_sliding(
        x,
        encoder=encoder,
        size=size,
        hop=hop,
        metric=metric,
        eps=eps,
        spec_win=spec_win,
        spec_hop=spec_hop,
        spec_window=spec_window,
        channel_fusion=channel_fusion,
        workers=workers,
    )
    feats = features.extract_batch(images, bins=bins, selected=feature_names)
    return feats, starts


__all__ = [
    "sliding_windows",
    "encode_sliding",
    "features_for_sliding",
]

