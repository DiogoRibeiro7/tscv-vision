"""Feature extraction utilities for time-series images."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Literal, cast, overload

import numpy as np
from numpy.typing import NDArray

from .fusion import fuse as _fuse

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
    ndarray
        Array of shape ``(6,)``.

    Raises
    ------
    ValueError
        If ``img`` is not 2D.
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
    ndarray
        Array of shape ``(bins,)``.

    Raises
    ------
    ValueError
        If ``img`` is not 2D.
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
    """Histogram of gradient magnitudes using Sobel filters.

    Parameters
    ----------
    img:
        2D image array.
    bins:
        Number of histogram bins.

    Returns
    -------
    ndarray
        Normalized histogram of shape ``(bins,)``.

    Raises
    ------
    ValueError
        If ``img`` is not 2D.
    """

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


def _lbp_codes(img: Array, radius: int) -> np.ndarray:
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
    return codes


def lbp(img: Array, radius: int = 1) -> Array:
    """Local Binary Pattern histogram (256 bins).

    Parameters
    ----------
    img:
        2D image array.
    radius:
        Neighbourhood radius for the LBP operator.

    Returns
    -------
    ndarray
        Normalized histogram of shape ``(256,)``.

    Raises
    ------
    ValueError
        If ``img`` is not 2D.
    """

    codes = _lbp_codes(img, radius)
    h, _ = np.histogram(codes.ravel(), bins=256, range=(0, 256), density=True)
    return h.astype(float)


def _lbp_rotation_map() -> np.ndarray:
    mapping = np.empty(256, dtype=np.uint8)
    for i in range(256):
        bits = [(i >> r) & 1 for r in range(8)]
        rotations = [
            sum(bits[(k + shift) % 8] << (7 - k) for k in range(8)) for shift in range(8)
        ]
        mapping[i] = min(rotations)
    return mapping


_LBP_RI_MAP = _lbp_rotation_map()


def lbp_ri(img: Array, radius: int = 1) -> Array:
    """Rotation-invariant Local Binary Pattern histogram (256 bins).

    Parameters
    ----------
    img:
        2D image array.
    radius:
        Neighbourhood radius for the LBP operator.

    Returns
    -------
    ndarray
        Normalized histogram of shape ``(256,)``.

    Raises
    ------
    ValueError
        If ``img`` is not 2D.
    """

    codes = _lbp_codes(img, radius)
    ri = _LBP_RI_MAP[codes]
    h, _ = np.histogram(ri.ravel(), bins=256, range=(0, 256), density=True)
    return h.astype(float)


def _lbp_uniform_map() -> tuple[np.ndarray, int]:
    mapping = np.full(256, 255, dtype=np.uint8)
    idx = 0
    for i in range(256):
        bits = [(i >> r) & 1 for r in range(8)]
        transitions = sum(bits[r] != bits[(r + 1) % 8] for r in range(8))
        if transitions <= 2:
            mapping[i] = idx
            idx += 1
    return mapping, idx + 1  # extra bin for non-uniform patterns


_LBP_UNI_MAP, _LBP_UNI_BINS = _lbp_uniform_map()


def lbp_uniform(img: Array, radius: int = 1) -> Array:
    """Uniform LBP histogram.

    Only patterns with at most two 0-1 transitions are counted as
    individual bins; all others share a single non-uniform bin.

    Parameters
    ----------
    img:
        2D image array.
    radius:
        Neighbourhood radius for the LBP operator.

    Returns
    -------
    ndarray
        Normalized histogram of shape ``(_LBP_UNI_BINS,)``.

    Raises
    ------
    ValueError
        If ``img`` is not 2D.
    """

    codes = _lbp_codes(img, radius)
    uni = _LBP_UNI_MAP[codes]
    h, _ = np.histogram(uni.ravel(), bins=_LBP_UNI_BINS, range=(0, _LBP_UNI_BINS), density=True)
    return h.astype(float)


def glcm_features(
    img: Array,
    *,
    levels: int = 8,
    distances: Sequence[int] = (1,),
    angles: Sequence[float] = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4),
) -> Array:
    """Gray-Level Co-occurrence Matrix statistics.

    Computes contrast, homogeneity and energy for each distance/angle pair and
    returns a flattened feature vector.
    """

    if img.ndim != 2:
        raise ValueError("img must be 2D")
    if levels <= 0:
        raise ValueError("levels must be positive")
    x = img.astype(float)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx == mn:
        quant = np.zeros_like(x, dtype=np.int64)
    else:
        quant = np.floor((x - mn) / (mx - mn + 1e-12) * (levels - 1)).astype(np.int64)
    feats: list[float] = []
    H, W = quant.shape
    for d in distances:
        for ang in angles:
            dx = int(round(math.cos(ang) * d))
            dy = int(round(math.sin(ang) * d))
            if dx == 0 and dy == 0:
                continue
            y0 = max(0, dy)
            y1 = H + min(0, dy)
            x0 = max(0, dx)
            x1 = W + min(0, dx)
            s = quant[y0:y1, x0:x1]
            t = quant[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            pair = s * levels + t
            glcm = np.bincount(pair.ravel(), minlength=levels * levels).reshape(levels, levels)
            if glcm.sum() == 0:
                glcm = glcm.astype(float)
            else:
                glcm = glcm.astype(float) / glcm.sum()
            i = np.arange(levels)
            j = i[:, None]
            diff = i - j
            contrast = float(np.sum((diff**2) * glcm))
            homogeneity = float(np.sum(glcm / (1.0 + diff**2)))
            energy = float(np.sum(glcm**2))
            feats.extend([contrast, homogeneity, energy])
    return np.array(feats, dtype=float)


def _gabor_kernel(
    frequency: float,
    theta: float,
    sigma: float = 1.0,
    gamma: float = 0.5,
    n_stds: int = 3,
) -> Array:
    radius = int(np.ceil(n_stds * sigma))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    rotx = x * math.cos(theta) + y * math.sin(theta)
    roty = -x * math.sin(theta) + y * math.cos(theta)
    g = np.exp(-0.5 * (rotx**2 + (gamma * roty) ** 2) / (sigma**2))
    g *= np.cos(2 * math.pi * frequency * rotx)
    return cast(Array, g.astype(float))


def gabor_features(
    img: Array,
    *,
    frequencies: Sequence[float] = (0.2, 0.4),
    thetas: Sequence[float] = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4),
) -> Array:
    """Gabor filter responses summarised by mean and std of magnitudes."""

    if img.ndim != 2:
        raise ValueError("img must be 2D")
    feats: list[float] = []
    for freq in frequencies:
        for theta in thetas:
            kernel = _gabor_kernel(freq, theta)
            resp = _conv2(img, kernel)
            mag = np.abs(resp)
            feats.append(float(np.mean(mag)))
            feats.append(float(np.std(mag)))
    return np.array(feats, dtype=float)


def edge_density(img: Array, threshold: float | None = None) -> Array:
    """Proportion of edge pixels based on gradient magnitude.

    Parameters
    ----------
    img:
        2D image array.
    threshold:
        Gradient magnitude above which a pixel is counted as an edge.
        Defaults to the mean magnitude.

    Returns
    -------
    ndarray
        Scalar array of shape ``(1,)`` with the edge ratio in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``img`` is not 2D.
    """

    if img.ndim != 2:
        raise ValueError("img must be 2D")
    gx = _conv2(img, _GX)
    gy = _conv2(img, _GY)
    mag = np.sqrt(gx * gx + gy * gy)
    if threshold is None:
        threshold = float(np.mean(mag))
    ratio = float(np.count_nonzero(mag > threshold) / mag.size)
    return np.array([ratio], dtype=float)


def orientation_histogram(img: Array, bins: int = 16) -> Array:
    """Histogram of gradient orientations in ``[0, 2pi)``.

    Parameters
    ----------
    img:
        2D image array.
    bins:
        Number of orientation bins.

    Returns
    -------
    ndarray
        Normalized histogram of shape ``(bins,)``.

    Raises
    ------
    ValueError
        If ``img`` is not 2D.
    """

    if img.ndim != 2:
        raise ValueError("img must be 2D")
    gx = _conv2(img, _GX)
    gy = _conv2(img, _GY)
    ang = np.mod(np.arctan2(gy, gx) + 2 * math.pi, 2 * math.pi)
    h, _ = np.histogram(ang.ravel(), bins=bins, range=(0.0, 2 * math.pi), density=True)
    return h.astype(float)


def contour_ratio(img: Array, threshold: float | None = None) -> Array:
    """Edge area ratio within its bounding box as a simple shape descriptor.

    Parameters
    ----------
    img:
        2D image array.
    threshold:
        Gradient magnitude threshold for edge detection.
        Defaults to the mean magnitude.

    Returns
    -------
    ndarray
        Scalar array of shape ``(1,)`` with the ratio in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``img`` is not 2D.
    """

    if img.ndim != 2:
        raise ValueError("img must be 2D")
    gx = _conv2(img, _GX)
    gy = _conv2(img, _GY)
    mag = np.sqrt(gx * gx + gy * gy)
    if threshold is None:
        threshold = float(np.mean(mag))
    edges = mag > threshold
    coords = np.argwhere(edges)
    if coords.size == 0:
        return np.array([0.0], dtype=float)
    area = float(coords.shape[0])
    y0, x0 = np.min(coords, axis=0)
    y1, x1 = np.max(coords, axis=0)
    bbox_area = float((y1 - y0 + 1) * (x1 - x0 + 1))
    ratio = area / bbox_area
    return np.array([ratio], dtype=float)


def fractal_dimension(img: Array, threshold: float = 0.5) -> Array:
    """Estimate fractal dimension via box counting.

    Parameters
    ----------
    img:
        2D image array.
    threshold:
        Binarisation threshold (applied after min-max scaling to
        ``[0, 1]``).

    Returns
    -------
    ndarray
        Scalar array of shape ``(1,)`` with the estimated dimension.

    Raises
    ------
    ValueError
        If ``img`` is not 2D.
    """

    if img.ndim != 2:
        raise ValueError("img must be 2D")
    x = img.astype(float)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx == mn:
        return np.array([0.0], dtype=float)
    x01 = (x - mn) / (mx - mn)
    binary = x01 > threshold
    H, W = binary.shape
    max_scale = int(np.floor(np.log2(min(H, W))))
    if max_scale == 0:
        return np.array([0.0], dtype=float)
    sizes = 2 ** np.arange(1, max_scale + 1)
    counts = []
    for size in sizes:
        h_blocks = H // size
        w_blocks = W // size
        if h_blocks == 0 or w_blocks == 0:
            break
        view = binary[: h_blocks * size, : w_blocks * size].reshape(h_blocks, size, w_blocks, size)
        blocks = np.any(np.any(view, axis=3), axis=1)
        counts.append(np.count_nonzero(blocks))
    if len(counts) < 2:
        return np.array([0.0], dtype=float)
    coeffs = np.polyfit(np.log(sizes[: len(counts)]), np.log(counts), 1)
    fd = -coeffs[0]
    return np.array([float(fd)], dtype=float)


def fft_features(img: Array) -> Array:
    """Mean and std of 2D FFT magnitude and phase.

    Parameters
    ----------
    img:
        2D image array.

    Returns
    -------
    ndarray
        Array of shape ``(4,)`` containing ``[mag_mean, mag_std,
        phase_mean, phase_std]``.

    Raises
    ------
    ValueError
        If ``img`` is not 2D.
    """

    if img.ndim != 2:
        raise ValueError("img must be 2D")
    fft = np.fft.fft2(img)
    mag = np.abs(fft)
    phase = np.angle(fft)
    return np.array(
        [float(np.mean(mag)), float(np.std(mag)), float(np.mean(phase)), float(np.std(phase))],
        dtype=float,
    )


try:
    import pywt
except Exception:  # pragma: no cover - optional dependency
    pywt = None


def wavelet_stats(img: Array, wavelet: str = "db1") -> Array:
    """Mean and std of wavelet detail coefficients.

    Requires ``pywt``.
    """

    if img.ndim != 2:
        raise ValueError("img must be 2D")
    if pywt is None:
        raise ImportError("pywt is required for wavelet_stats")
    coeffs2 = pywt.dwt2(img, wavelet)
    _cA, (cH, cV, cD) = coeffs2
    feats: list[float] = []
    for c in (cH, cV, cD):
        feats.append(float(np.mean(c)))
        feats.append(float(np.std(c)))
    return np.array(feats, dtype=float)


def power_spectral_density(img: Array) -> Array:
    """Mean and std of power spectral density.

    Parameters
    ----------
    img:
        2D image array.

    Returns
    -------
    ndarray
        Array of shape ``(2,)`` containing ``[psd_mean, psd_std]``.

    Raises
    ------
    ValueError
        If ``img`` is not 2D.
    """

    if img.ndim != 2:
        raise ValueError("img must be 2D")
    fft = np.fft.fft2(img)
    psd = np.abs(fft) ** 2
    return np.array([float(np.mean(psd)), float(np.std(psd))], dtype=float)


def cnn_features(
    img: Array,
    *,
    model: str = "resnet18",
    layer: str = "avgpool",
    device: str = "cpu",
) -> Array:
    """Extract features from a pre-trained CNN.

    Requires ``torch`` and ``torchvision``; uses the output of ``layer`` as the
    feature vector.
    """

    try:
        import torch
        from torchvision import models, transforms
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError("torch and torchvision are required for cnn_features") from e

    net = getattr(models, model)(weights="DEFAULT").to(device)
    net.eval()

    feats: list[np.ndarray] = []

    def hook(_: Any, __: Any, output: Any) -> None:  # noqa: ANN001
        feats.append(output.detach().cpu().numpy().ravel())

    handle = dict(net.named_modules())[layer].register_forward_hook(hook)
    with torch.no_grad():
        arr = np.repeat(img[..., None], 3, axis=2) if img.ndim == 2 else img
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
        preprocess = transforms.Resize((224, 224))
        tensor = preprocess(tensor)
        tensor = tensor.to(device)
        net(tensor)
    handle.remove()
    if not feats:
        raise RuntimeError("Layer hook did not capture features")
    return feats[0].astype(float)


def autoencoder_features(img: Array, model: Any) -> Array:
    """Return latent representation from a provided autoencoder ``model``."""

    try:
        import torch
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError("torch is required for autoencoder_features") from e

    tensor = torch.from_numpy(img.astype(np.float32))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError("img must be 2D or 3D")
    with torch.no_grad():
        if hasattr(model, "encode"):
            latent = model.encode(tensor)
        elif hasattr(model, "encoder"):
            latent = model.encoder(tensor)
        else:  # pragma: no cover - user error
            raise AttributeError("model must have encode or encoder method")
    latent_arr = latent.detach().cpu().numpy().ravel().astype(float)
    return cast(Array, latent_arr)


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
    "lbp_ri": _wrap_no_bins(lbp_ri),
    "lbp_uniform": _wrap_no_bins(lbp_uniform),
    "glcm": _wrap_no_bins(glcm_features),
    "gabor": _wrap_no_bins(gabor_features),
    "edge_density": _wrap_no_bins(edge_density),
    "orientation": orientation_histogram,
    "contour": _wrap_no_bins(contour_ratio),
    "fractal": _wrap_no_bins(fractal_dimension),
    "fft": _wrap_no_bins(fft_features),
    "psd": _wrap_no_bins(power_spectral_density),
    "wavelet": _wrap_no_bins(wavelet_stats),
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
            try:
                parts.append(FEATURES_REGISTRY[name](ch, bins))
            except ImportError:
                if selected is not None and name in selected:
                    raise
                continue
    if not parts:
        raise RuntimeError(
            "No feature extractors produced output; install optional "
            "dependencies (pywavelets, etc.) or pass a narrower 'selected' list."
        )
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


def scale_features(X: Array, method: Literal["zscore", "minmax"] = "zscore") -> Array:
    """Normalize features with ``zscore`` or ``minmax`` scaling.

    Parameters
    ----------
    X:
        2D feature matrix of shape ``(n_samples, n_features)``.
    method:
        ``"zscore"`` subtracts the mean and divides by std;
        ``"minmax"`` scales each column to ``[0, 1]``.

    Returns
    -------
    ndarray
        Scaled array with the same shape as ``X``.

    Raises
    ------
    ValueError
        If ``X`` is not 2D or ``method`` is unknown.
    """

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if method == "zscore":
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0) + 1e-12
        return cast(Array, (X - mu) / sigma)
    if method == "minmax":
        mn = np.min(X, axis=0)
        mx = np.max(X, axis=0)
        denom = mx - mn + 1e-12
        return cast(Array, (X - mn) / denom)
    raise ValueError("Unknown method")


def rank_features(
    X: Array,
    y: Array | None = None,
    method: Literal["variance", "mutual_info"] = "variance",
) -> np.ndarray:
    """Return feature indices sorted by descending importance.

    Parameters
    ----------
    X:
        2D feature matrix of shape ``(n_samples, n_features)``.
    y:
        Target labels, required when ``method="mutual_info"``.
    method:
        ``"variance"`` ranks by column variance; ``"mutual_info"``
        uses sklearn's mutual information classifier.

    Returns
    -------
    ndarray
        Integer index array of shape ``(n_features,)`` ordered from
        most to least important.

    Raises
    ------
    ValueError
        If ``X`` is not 2D, ``method`` is unknown, or ``y`` is missing
        for ``"mutual_info"``.
    """

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if method == "variance":
        scores = np.var(X, axis=0)
    elif method == "mutual_info":
        if y is None:
            raise ValueError("y required for mutual_info")
        try:
            from sklearn.feature_selection import mutual_info_classif
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError("scikit-learn required for mutual_info") from e
        scores = mutual_info_classif(X, y)
    else:
        raise ValueError("Unknown method")
    return np.argsort(-scores)


def select_top_k(
    X: Array,
    k: int,
    y: Array | None = None,
    method: Literal["variance", "mutual_info"] = "variance",
) -> Array:
    """Select the top-``k`` features according to ``method``.

    Parameters
    ----------
    X:
        2D feature matrix of shape ``(n_samples, n_features)``.
    k:
        Number of features to keep (must be positive and at most
        ``n_features``).
    y:
        Target labels, required when ``method="mutual_info"``.
    method:
        Ranking criterion forwarded to :func:`rank_features`.

    Returns
    -------
    ndarray
        Reduced matrix of shape ``(n_samples, k)``.

    Raises
    ------
    ValueError
        If ``k`` is non-positive or exceeds the number of features.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if k > X.shape[1]:
        raise ValueError("k must not exceed number of features")
    idx = rank_features(X, y=y, method=method)[:k]
    return X[:, idx]


def fuse_features(
    features_list: Sequence[Array],
    mode: Literal["concat", "mean", "median", "weighted"] = "concat",
    weights: Sequence[float] | None = None,
) -> Array:
    """Wrapper around :func:`tscv_vision.fusion.fuse` for convenience."""

    return _fuse(features_list, mode=mode, weights=weights)


__all__ = [
    "intensity_stats",
    "histogram",
    "gradient_histogram",
    "lbp",
    "lbp_ri",
    "lbp_uniform",
    "glcm_features",
    "gabor_features",
    "edge_density",
    "orientation_histogram",
    "contour_ratio",
    "fractal_dimension",
    "fft_features",
    "wavelet_stats",
    "power_spectral_density",
    "cnn_features",
    "autoencoder_features",
    "extract_feature_vector",
    "extract_batch",
    "scale_features",
    "rank_features",
    "select_top_k",
    "fuse_features",
    "FEATURES_REGISTRY",
]

