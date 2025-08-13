from __future__ import annotations
import numpy as np
from typing import Tuple

Array = np.ndarray

# Small convolution helpers for gradients (Sobel-like 3x3)
_GX = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=float)
_GY = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]], dtype=float)


def _pad_reflect(img: Array, k: int) -> Array:
    return np.pad(img, ((k, k), (k, k)), mode="reflect")


def _conv2(img: Array, kernel: Array) -> Array:
    k = kernel.shape[0] // 2
    p = _pad_reflect(img, k)
    out = np.zeros_like(img, dtype=float)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            region = p[i:i+2*k+1, j:j+2*k+1]
            out[i, j] = float(np.sum(region * kernel))
    return out


def intensity_stats(img: Array) -> Array:
    """Basic intensity statistics: mean, std, min, max, skewness, kurtosis."""
    x = img.astype(float).ravel()
    mu = float(np.mean(x))
    sigma = float(np.std(x) + 1e-12)
    mn = float(np.min(x))
    mx = float(np.max(x))
    z = (x - mu) / sigma
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4) - 3.0)
    return np.array([mu, sigma, mn, mx, skew, kurt], dtype=float)


def histogram(img: Array, bins: int = 32) -> Array:
    x = img.astype(float).ravel()
    # Assume image in arbitrary range; robustly scale to [0,1]
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx == mn:
        h = np.zeros(bins, dtype=float)
        h[0] = 1.0
        return h
    x01 = (x - mn) / (mx - mn)
    h, _ = np.histogram(x01, bins=bins, range=(0.0, 1.0), density=True)
    return h.astype(float)


def gradient_histogram(img: Array, bins: int = 16) -> Array:
    gx = _conv2(img, _GX)
    gy = _conv2(img, _GY)
    mag = np.sqrt(gx * gx + gy * gy)
    # magnitude-only histogram
    if np.allclose(mag.max(), 0.0):
        h = np.zeros(bins, dtype=float)
        h[0] = 1.0
        return h
    m = mag / (mag.max() + 1e-12)
    h, _ = np.histogram(m.ravel(), bins=bins, range=(0.0, 1.0), density=True)
    return h.astype(float)


def lbp(img: Array, radius: int = 1) -> Array:
    """Local Binary Pattern histogram (8-neighborhood) with given radius.

    Returns a 256-bin histogram (uniformity not enforced for simplicity).
    """
    padded = np.pad(img, radius, mode="reflect")
    H, W = img.shape
    codes = np.zeros((H, W), dtype=np.uint16)
    # Offsets for 8 neighbors (clockwise)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    center = padded[radius:radius+H, radius:radius+W]
    for bit, (dy, dx) in enumerate(offs):
        nbr = padded[radius+dy:radius+dy+H, radius+dx:radius+dx+W]
        codes |= ((nbr >= center).astype(np.uint16) << bit)
    h, _ = np.histogram(codes.ravel(), bins=256, range=(0, 256), density=True)
    return h.astype(float)


def extract_feature_vector(img: Array, bins: int = 32) -> Array:
    """Compose a single 1D feature vector from multiple descriptors.

    Vector = [intensity_stats(6) | histogram(bins) | grad_hist(16) | lbp(256)]
    """
    img = img.astype(float)
    parts = [
        intensity_stats(img),
        histogram(img, bins=bins),
        gradient_histogram(img, bins=16),
        lbp(img, radius=1),
    ]
    return np.concatenate(parts, dtype=float)