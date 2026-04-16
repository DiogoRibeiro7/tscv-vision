"""Dataset utilities for streaming windowed features."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from . import features
from .sliding import encode_sliding

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]


class WindowedDataset:
    """Iterate over sliding windows and yield features with metadata.

    Parameters
    ----------
    x:
        Input array or path to a ``.npy`` file for memory-mapped loading.
    size, hop, encoder, bins, features_sel, cache, channel_fusion, metric,
    eps, spec_win, spec_hop, spec_window, workers:
        Forwarded to :func:`encode_sliding` and feature extraction utilities.
    """

    def __init__(
        self,
        x: Array | str,
        size: int,
        hop: int | None = None,
        *,
        encoder: Literal["gaf", "gadf", "rp", "spec"] = "gaf",
        bins: int = 32,
        features_sel: Iterable[str] | None = None,
        cache: str | None = None,
        channel_fusion: Literal["stack", "mean", "concat"] = "stack",
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        eps: float | None = None,
        spec_win: int | None = None,
        spec_hop: int | None = None,
        spec_window: Literal["hann", "rect"] = "hann",
        workers: int | None = None,
    ) -> None:
        if isinstance(x, str):
            if x.endswith(".npy"):
                self._x = np.load(x, mmap_mode="r")
            else:
                raise ValueError("Only .npy files are supported for memory-mapped input")
        else:
            self._x = np.asarray(x, dtype=float)
        self.size = size
        self.hop = hop
        self.encoder = encoder
        self.bins = bins
        self.features_sel = features_sel
        self.cache = cache
        self.fusion = channel_fusion
        self.metric = metric
        self.eps = eps
        self.spec_win = spec_win
        self.spec_hop = spec_hop
        self.spec_window = spec_window
        self.workers = workers
        self._features: Array | None = None
        self._starts: IntArray | None = None

    def _ensure(self) -> None:
        if self._features is not None:
            return
        if self.cache and os.path.exists(self.cache):
            data = np.load(self.cache)
            self._features = data["features"]
            self._starts = data["starts"].astype(int)
            return
        images, starts = encode_sliding(
            self._x,
            encoder=self.encoder,
            size=self.size,
            hop=self.hop,
            metric=self.metric,
            eps=self.eps,
            spec_win=self.spec_win,
            spec_hop=self.spec_hop,
            spec_window=self.spec_window,
            channel_fusion=self.fusion,
            workers=self.workers,
        )
        feats = features.extract_batch(images, bins=self.bins, selected=self.features_sel)
        self._features = feats
        self._starts = starts
        if self.cache:
            np.savez(self.cache, features=feats, starts=starts)

    def __iter__(self) -> Iterator[tuple[Array, dict[str, int]]]:
        self._ensure()
        if self._features is None or self._starts is None:
            raise RuntimeError("dataset not materialised")
        for feat, start in zip(self._features, self._starts, strict=True):
            yield feat, {"start": int(start)}

    def __len__(self) -> int:  # pragma: no cover - trivial
        self._ensure()
        if self._starts is None:
            raise RuntimeError("dataset not materialised")
        return int(self._starts.shape[0])


def iter_npz(path: str) -> Iterator[Array]:
    """Yield arrays stored in a ``.npz`` archive one by one."""

    with np.load(path) as data:
        for name in data.files:
            yield np.asarray(data[name], dtype=float)


def iter_parquet(path: str, batch_size: int = 1024) -> Iterator[Array]:
    """Yield batches from a Parquet file.

    Requires :mod:`pyarrow` to be installed.
    """

    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - import error
        raise ImportError("pyarrow is required for Parquet streaming") from exc
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield np.asarray(batch.column(0).to_numpy(), dtype=float)


def stream_directory(path: str) -> Iterator[Array]:
    """Yield memory-mapped ``.npy`` arrays from ``path`` one by one."""

    for name in sorted(os.listdir(path)):
        if name.endswith(".npy"):
            yield np.load(os.path.join(path, name), mmap_mode="r")


__all__ = ["WindowedDataset", "iter_npz", "iter_parquet", "stream_directory"]

