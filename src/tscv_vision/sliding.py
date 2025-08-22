"""Sliding-window utilities and batch encoders."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import partial
from typing import Callable, Literal, cast, overload

import numpy as np
from numpy.typing import NDArray

from . import encoders as _enc
from . import features
from .parallel import map_parallel

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]


EncoderLike = str | Callable[[Array], Array]


def _encode_window_static(
    w: Array,
    *,
    encoder: EncoderLike,
    size: int,
    metric: Literal["euclidean", "manhattan"],
    eps: float | None,
    spec_win: int | None,
    spec_hop: int | None,
    spec_window: Literal["hann", "rect"],
    channel_fusion: Literal["stack", "mean", "concat"],
    cwt_scales: Array | None,
) -> Array:
    series = [w] if w.ndim == 1 else [w[:, c] for c in range(w.shape[1])]
    ch_imgs: list[Array] = []
    if isinstance(encoder, str):
        if encoder in {"gaf", "gadf"}:
            method: Literal["summation", "difference"] = (
                "summation" if encoder == "gaf" else "difference"
            )
            for s in series:
                ch_imgs.append(_enc.gaf(s, method=method))
        elif encoder == "rp":
            for s in series:
                ch_imgs.append(_enc.recurrence_plot(s, metric=metric, eps=eps))
        elif encoder == "spec":
            fwin = size if spec_win is None else spec_win
            for s in series:
                ch_imgs.append(
                    _enc.spectrogram(s, win=fwin, hop=spec_hop, window=spec_window)
                )
        elif encoder == "cwt":
            scales = (
                np.arange(1, min(size, 32), dtype=float)
                if cwt_scales is None
                else np.asarray(cwt_scales, dtype=float)
            )
            for s in series:
                ch_imgs.append(_enc.cwt(s, scales=scales))
        else:
            func = _enc.get_encoder(encoder)
            for s in series:
                ch_imgs.append(func(s))
    else:
        func = encoder
        for s in series:
            ch_imgs.append(func(s))

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
        View with shape ``(n_windows, size[, C])``. The view is O(1) to create
        but can be large; use :func:`encode_sliding` with ``lazy=True`` for
        memory-constrained workloads.
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


@overload
def encode_sliding(
    x: Array,
    encoder: EncoderLike = "gaf",
    *,
    size: int,
    hop: int | None = None,
    metric: Literal["euclidean", "manhattan"] = "euclidean",
    eps: float | None = None,
    spec_win: int | None = None,
    spec_hop: int | None = None,
    spec_window: Literal["hann", "rect"] = "hann",
    channel_fusion: Literal["stack", "mean", "concat"] = "stack",
    cwt_scales: Array | None = None,
    workers: int | None = None,
    lazy: Literal[False] = False,
) -> tuple[Array, IntArray]:
    ...


@overload
def encode_sliding(
    x: Array,
    encoder: EncoderLike = "gaf",
    *,
    size: int,
    hop: int | None = None,
    metric: Literal["euclidean", "manhattan"] = "euclidean",
    eps: float | None = None,
    spec_win: int | None = None,
    spec_hop: int | None = None,
    spec_window: Literal["hann", "rect"] = "hann",
    channel_fusion: Literal["stack", "mean", "concat"] = "stack",
    cwt_scales: Array | None = None,
    workers: int | None = None,
    lazy: Literal[True],
) -> Iterator[tuple[Array, int]]:
    ...


def encode_sliding(
    x: Array,
    encoder: EncoderLike = "gaf",
    *,
    size: int,
    hop: int | None = None,
    metric: Literal["euclidean", "manhattan"] = "euclidean",
    eps: float | None = None,
    spec_win: int | None = None,
    spec_hop: int | None = None,
    spec_window: Literal["hann", "rect"] = "hann",
    channel_fusion: Literal["stack", "mean", "concat"] = "stack",
    cwt_scales: Array | None = None,
    workers: int | None = None,
    lazy: bool = False,
) -> tuple[Array, IntArray] | Iterator[tuple[Array, int]]:
    """Encode sliding windows from ``x`` into images.

    Parameters
    ----------
    x:
        Input series ``(N,)`` or ``(N, C)``.
    encoder:
        Encoder name.
    size, hop, metric, eps, spec_win, spec_hop, spec_window, channel_fusion:
        Same as :func:`sliding_windows` and encoders.
    workers:
        Number of worker processes for parallel encoding.
    lazy:
        If ``True`` yield ``(image, start)`` pairs instead of stacking all
        images into memory. This trades some speed for much lower memory use.

    Returns
    -------
    tuple or iterator
        When ``lazy`` is ``False`` (default) returns ``(images, starts)`` where
        ``images`` has shape ``(n_windows, H, W[, C])``. When ``lazy`` is
        ``True`` returns an iterator yielding ``(image, start)`` tuples.
    """

    x = np.asarray(x, dtype=float)
    real_hop = hop if hop is not None else max(1, size // 2)

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
        cwt_scales=cwt_scales,
    )

    if lazy:
        def gen() -> Iterator[tuple[Array, int]]:
            n = x.shape[0]
            if n < size:
                return
            if x.ndim == 1:
                for start in range(0, n - size + 1, real_hop):
                    w = x[start : start + size]
                    yield func(w), start
            else:
                for start in range(0, n - size + 1, real_hop):
                    w = x[start : start + size, :]
                    yield func(w), start

        return gen()

    win_view = sliding_windows(x, size=size, hop=real_hop, copy=False)
    n_windows = win_view.shape[0]
    if n_windows == 0:
        return (
            np.zeros((0, size, size), dtype=float),
            np.zeros((0,), dtype=int),
        )
    starts = np.arange(n_windows, dtype=int) * real_hop
    imgs_list = list(map_parallel(func, cast(Iterable[Array], win_view), workers))

    if isinstance(encoder, str) and encoder == "spec":
        maxF = max(im.shape[0] for im in imgs_list)
        maxT = max(im.shape[1] for im in imgs_list)
        if imgs_list[0].ndim == 3:
            out = np.zeros((n_windows, maxF, maxT, imgs_list[0].shape[2]), dtype=float)
        else:
            out = np.zeros((n_windows, maxF, maxT), dtype=float)
        for i, im in enumerate(imgs_list):
            out[i, : im.shape[0], : im.shape[1], ...] = im
        return out, starts

    return np.stack(imgs_list, axis=0), starts


@overload
def features_for_sliding(
    x: Array,
    *,
    encoder: EncoderLike = "gaf",
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
    cwt_scales: Array | None = None,
    workers: int | None = None,
    lazy: Literal[False] = False,
) -> tuple[Array, IntArray]:
    ...


@overload
def features_for_sliding(
    x: Array,
    *,
    encoder: EncoderLike = "gaf",
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
    cwt_scales: Array | None = None,
    workers: int | None = None,
    lazy: Literal[True],
) -> Iterator[tuple[Array, int]]:
    ...


def features_for_sliding(
    x: Array,
    *,
    encoder: EncoderLike = "gaf",
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
    cwt_scales: Array | None = None,
    workers: int | None = None,
    lazy: bool = False,
) -> tuple[Array, IntArray] | Iterator[tuple[Array, int]]:
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
    lazy:
        If ``True`` yield ``(feat, start)`` tuples instead of stacking the
        feature matrix into memory.

    Returns
    -------
    tuple or iterator
        When ``lazy`` is ``False`` returns ``(features, starts)`` with feature
        matrix shape ``(n_windows, D)``. When ``True`` returns an iterator of
        ``(feature, start)`` pairs.
    """

    x = np.asarray(x, dtype=float)
    if lazy:
        def gen() -> Iterator[tuple[Array, int]]:
            for img, st in encode_sliding(
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
                cwt_scales=cwt_scales,
                workers=workers,
                lazy=True,
            ):
                feat = features.extract_feature_vector(
                    img, bins=bins, selected=feature_names
                )
                yield feat, st

        return gen()

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
        cwt_scales=cwt_scales,
        workers=workers,
    )
    feats = features.extract_batch(images, bins=bins, selected=feature_names)
    return feats, starts


__all__ = [
    "sliding_windows",
    "encode_sliding",
    "features_for_sliding",
]

