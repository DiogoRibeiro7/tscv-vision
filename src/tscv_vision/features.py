"""Feature extraction utilities for time-series images."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Callable, Literal, cast, overload

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


# Sobel-like kernels for gradients
_GX = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=float)
_GY = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=float)


def _pad_reflect(img: Array, k: int) -> Array:
    return np.pad(img, ((k, k), (k, k)), mode="reflect")


def _conv2(img: Array, kernel: Array) -> Array:
    k_h, k_w = kernel.shape
    if k_h != k_w or k_h % 2 == 0:
        raise ValueError("kernel must be square with odd size")
    k = k_h // 2
    p = _pad_reflect(img, k)
    H, W = img.shape
    view = np.lib.stride_tricks.as_strided(
        p,
        shape=(H, W, k_h, k_w),
        strides=(p.strides[0], p.strides[1], p.strides[0], p.strides[1]),
        writeable=False,
    )
    return cast(Array, np.tensordot(view, kernel, axes=((2, 3), (0, 1))))


def intensity_stats(img: Array) -> Array:
    """Return mean, std, min, max, skewness and kurtosis of ``img``.

    Parameters
    ----------
    img:
        2D image array.

    Returns
    -------
    Array of shape ``(6,)``.
    """

    if img.ndim != 2:
        raise ValueError("img must be 2D")
    x = img.astype(float).ravel()
    mu = float(np.mean(x))
    sigma = float(np.std(x) + 1e-12)
    mn = float(np.min(x))
    mx = float(np.max(x))
    z = (x - mu) / sigma
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4) - 3.0)
    arr: Array = np.array([mu, sigma, mn, mx, skew, kurt], dtype=np.float64)
    return arr


def histogram(img: Array, bins: int = 32) -> Array:
    """Normalized intensity histogram.

    Parameters
    ----------
    img:
        2D image array.
    bins:
        Number of histogram bins.

    Returns
    -------
    Array of shape ``(bins,)``.
    """

    if img.ndim != 2:
        raise ValueError("img must be 2D")
    x = img.astype(float).ravel()
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx == mn:
        h = np.zeros(bins, dtype=float)
        h[0] = 1.0
        return h
    x01 = (x - mn) / (mx - mn)
    h, _ = np.histogram(x01, bins=bins, range=(0.0, 1.0), density=True)
    return h.astype(float)


def gradient_histogram(img: Array, bins: int = 16) -> Array:
    """Histogram of gradient magnitudes using Sobel filters."""

    if img.ndim != 2:
        raise ValueError("img must be 2D")
    gx = _conv2(img, _GX)
    gy = _conv2(img, _GY)
    mag = np.sqrt(gx * gx + gy * gy)
    if np.allclose(mag.max(), 0.0):
        h = np.zeros(bins, dtype=float)
        h[0] = 1.0
        return h
    m = mag / (mag.max() + 1e-12)
    h, _ = np.histogram(m.ravel(), bins=bins, range=(0.0, 1.0), density=True)
    return h.astype(float)


def lbp(img: Array, radius: int = 1) -> Array:
    """Local Binary Pattern histogram (256 bins).

    Parameters
    ----------
    img:
        2D image array.
    radius:
        Neighborhood radius (default ``1``).

    Returns
    -------
    Array of shape ``(256,)``.
    """

    if img.ndim != 2:
        raise ValueError("img must be 2D")
    padded = np.pad(img, radius, mode="reflect")
    H, W = img.shape
    codes = np.zeros((H, W), dtype=np.uint16)
    offs = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]
    center = padded[radius : radius + H, radius : radius + W]
    for bit, (dy, dx) in enumerate(offs):
        nbr = padded[radius + dy : radius + dy + H, radius + dx : radius + dx + W]
        incr = np.left_shift((nbr >= center).astype(np.uint16), np.uint16(bit))
        codes |= incr
    h, _ = np.histogram(codes.ravel(), bins=256, range=(0, 256), density=True)
    return h.astype(float)


def _wrap_no_bins(fn: Callable[[Array], Array]) -> Callable[[Array, int], Array]:
    def inner(img: Array, bins: int = 32) -> Array:  # noqa: ANN001, ARG001
        return fn(img)

    return inner

FeatureFunc = Callable[[Array, int], Array]


FEATURES_REGISTRY: dict[str, FeatureFunc] = {
    "intensity": _wrap_no_bins(intensity_stats),
    "hist": histogram,
    "gradient": _wrap_no_bins(lambda img: gradient_histogram(img, bins=16)),
    "lbp": _wrap_no_bins(lbp),
}


def extract_feature_vector(
    img: Array, bins: int = 32, selected: Iterable[str] | None = None
) -> Array:
    """Compose a feature vector from registered descriptors.

    Parameters
    ----------
    img:
        2D image or stack ``(H, W[, C])``.
    bins:
        Histogram bins for features that support it.
    selected:
        Feature names to include; defaults to all registered.

    Returns
    -------
    Array
        Concatenated feature vector.
    """

    if selected is None:
        names = list(FEATURES_REGISTRY)
    else:
        names = list(selected)
        for name in names:
            if name not in FEATURES_REGISTRY:
                raise KeyError(name)

    if img.ndim == 2:
        channels = [img]
    elif img.ndim == 3:
        channels = [img[..., c] for c in range(img.shape[2])]
    else:
        raise ValueError("img must be 2D or 3D")

    parts: list[Array] = []
    for ch in channels:
        for name in names:
            parts.append(FEATURES_REGISTRY[name](ch, bins))
    return np.concatenate(parts, dtype=float)


@overload
def extract_batch(
    images: Array,
    bins: int = 32,
    selected: Iterable[str] | None = None,
    *,
    lazy: Literal[False] = False,
) -> Array:
    ...


@overload
def extract_batch(
    images: Array,
    bins: int = 32,
    selected: Iterable[str] | None = None,
    *,
    lazy: Literal[True],
) -> Iterator[Array]:
    ...


def extract_batch(
    images: Array,
    bins: int = 32,
    selected: Iterable[str] | None = None,
    *,
    lazy: bool = False,
) -> Array | Iterator[Array]:
    """Extract feature vectors for a batch of images.

    Parameters
    ----------
    images:
        Array of shape ``(N, H, W)`` or ``(N, H, W, C)``.
    bins:
        Histogram bins for :func:`extract_feature_vector`.
    selected:
        Feature names to compute. ``None`` uses all registered features.
    lazy:
        If ``True`` yield feature vectors one by one instead of allocating a
        full ``(N, D)`` matrix. This reduces peak memory for very large
        batches at the cost of increased iteration overhead.

    Returns
    -------
    Array or Iterator[Array]
        Matrix of shape ``(N, D)`` or an iterator over feature vectors when
        ``lazy`` is ``True``.
    """

    if images.ndim not in {3, 4}:
        raise ValueError("images must have shape (N, H, W[, C])")

    if lazy:
        def gen() -> Iterator[Array]:
            for im in images:
                yield extract_feature_vector(im, bins=bins, selected=selected)

        return gen()

    n = images.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    first = extract_feature_vector(images[0], bins=bins, selected=selected)
    out = np.empty((n, first.size), dtype=float)
    out[0] = first
    for i in range(1, n):
        out[i] = extract_feature_vector(images[i], bins=bins, selected=selected)
    return out


__all__ = [
    "intensity_stats",
    "histogram",
    "gradient_histogram",
    "lbp",
    "extract_feature_vector",
    "extract_batch",
    "FEATURES_REGISTRY",
]

