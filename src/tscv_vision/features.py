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
    x = np.asarray(img, dtype=float).ravel()
    mu = float(np.mean(x))
    centered = x - mu
    sigma = float(np.sqrt(np.mean(centered * centered)) + 1e-12)
    mn = float(np.min(x))
    mx = float(np.max(x))
    z = centered / sigma
    z2 = z * z
    skew = float(np.mean(z2 * z))
    kurt = float(np.mean(z2 * z2) - 3.0)
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
    _, _, mag = _gradient_components(img)
    return _gradient_histogram_from_mag(mag, bins)


def _gradient_components(img: Array) -> tuple[Array, Array, Array]:
    gx = _conv2(img, _GX)
    gy = _conv2(img, _GY)
    mag = np.sqrt(gx * gx + gy * gy)
    return gx, gy, cast(Array, mag)


def _gradient_histogram_from_mag(mag: Array, bins: int = 16) -> Array:
    if np.allclose(mag.max(), 0.0):
        h = np.zeros(bins, dtype=float)
        h[0] = 1.0
        return h
    m = mag / (mag.max() + 1e-12)
    h, _ = np.histogram(m.ravel(), bins=bins, range=(0.0, 1.0), density=True)
    return h.astype(float)


#: Number of sampling points used by the LBP operators in this module.
_LBP_POINTS = 8


def _lbp_offsets(radius: float, points: int = _LBP_POINTS) -> Array:
    """Return ``(points, 2)`` ``(row, col)`` offsets on a circle of ``radius``.

    Sampling follows the standard :math:`LBP_{P,R}` convention of Ojala et al.
    (2002): point ``p`` sits at angle :math:`2\\pi p / P` measured
    counter-clockwise from the positive column axis, so bit ``0`` is the
    neighbour at ``(0, +R)``. This matches ``skimage.feature.
    local_binary_pattern``.
    """

    angles = 2.0 * np.pi * np.arange(points, dtype=float) / points
    rows = -radius * np.sin(angles)
    cols = radius * np.cos(angles)
    offs = np.stack([rows, cols], axis=1)
    # Snap near-integer offsets (axis-aligned points) so that they are sampled
    # exactly instead of interpolated between two rows/columns 1e-16 apart.
    rounded = np.round(offs)
    return cast(Array, np.where(np.abs(offs - rounded) < 1e-9, rounded, offs))


def _bilinear_sample(padded: Array, rows: Array, cols: Array) -> Array:
    """Bilinearly sample ``padded`` on the grid ``rows`` x ``cols``."""

    h, w = padded.shape
    r0 = np.floor(rows).astype(np.intp)
    c0 = np.floor(cols).astype(np.intp)
    wr = (rows - r0)[:, None]
    wc = (cols - c0)[None, :]
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    v00 = padded[np.ix_(r0, c0)]
    v01 = padded[np.ix_(r0, c1)]
    v10 = padded[np.ix_(r1, c0)]
    v11 = padded[np.ix_(r1, c1)]
    top = v00 * (1.0 - wc) + v01 * wc
    bottom = v10 * (1.0 - wc) + v11 * wc
    return cast(Array, top * (1.0 - wr) + bottom * wr)


def _lbp_codes(img: Array, radius: int, points: int = _LBP_POINTS) -> np.ndarray:
    """Return the :math:`LBP_{P,R}` code of every pixel of ``img``.

    Neighbours are sampled on a circle of ``radius`` using bilinear
    interpolation, so ``radius`` genuinely changes the sampling geometry.
    Out-of-image neighbours use reflected padding.
    """

    if img.ndim != 2:
        raise ValueError("img must be 2D")
    if radius < 1:
        raise ValueError("radius must be >= 1")
    if points < 1 or points > 16:
        raise ValueError("points must be in [1, 16]")
    pad = int(math.ceil(radius))
    padded = np.pad(np.asarray(img, dtype=float), pad, mode="reflect")
    H, W = img.shape
    rows = np.arange(H, dtype=float) + pad
    cols = np.arange(W, dtype=float) + pad
    center = padded[pad : pad + H, pad : pad + W]
    codes = np.zeros((H, W), dtype=np.uint16)
    for bit, (dr, dc) in enumerate(_lbp_offsets(float(radius), points)):
        nbr = _bilinear_sample(padded, rows + dr, cols + dc)
        incr = np.left_shift((nbr >= center).astype(np.uint16), np.uint16(bit))
        codes |= incr
    return codes


def lbp(img: Array, radius: int = 1) -> Array:
    """Local Binary Pattern histogram (256 bins).

    Implements :math:`LBP_{8,R}` with circular, bilinearly interpolated
    sampling (Ojala et al., 2002). Increasing ``radius`` samples neighbours
    further from the centre pixel and therefore yields different codes.

    Parameters
    ----------
    img:
        2D image array.
    radius:
        Neighbourhood radius for the LBP operator; must be ``>= 1``.

    Returns
    -------
    ndarray
        Normalized histogram of shape ``(256,)``.

    Raises
    ------
    ValueError
        If ``img`` is not 2D or ``radius < 1``.
    """

    codes = _lbp_codes(img, radius)
    return _lbp_histogram(codes)


def _lbp_rotation_map(points: int = _LBP_POINTS) -> np.ndarray:
    """Map each code to the smallest code in its bitwise-rotation orbit."""

    n_codes = 1 << points
    mask = n_codes - 1
    mapping = np.empty(n_codes, dtype=np.uint16)
    for i in range(n_codes):
        rotations = [((i >> s) | (i << (points - s))) & mask for s in range(points)]
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
    return _lbp_histogram(ri)


def _lbp_uniform_map(points: int = _LBP_POINTS) -> tuple[np.ndarray, int]:
    """Map uniform codes to consecutive bins and all others to a shared bin.

    Returns the lookup table plus the total number of bins. For ``points=8``
    there are 58 uniform patterns, so bin ``58`` collects the non-uniform ones
    and the histogram has 59 bins.
    """

    n_codes = 1 << points
    uniform: list[int] = []
    for i in range(n_codes):
        bits = [(i >> r) & 1 for r in range(points)]
        transitions = sum(bits[r] != bits[(r + 1) % points] for r in range(points))
        if transitions <= 2:
            uniform.append(i)
    non_uniform_bin = len(uniform)
    mapping = np.full(n_codes, non_uniform_bin, dtype=np.uint16)
    for bin_idx, code in enumerate(uniform):
        mapping[code] = bin_idx
    return mapping, non_uniform_bin + 1  # extra bin for non-uniform patterns


_LBP_UNI_MAP, _LBP_UNI_BINS = _lbp_uniform_map()


def lbp_uniform(img: Array, radius: int = 1) -> Array:
    """Uniform LBP histogram.

    Only patterns with at most two 0-1 transitions are counted as
    individual bins; all others share the final non-uniform bin
    (index ``_LBP_UNI_BINS - 1``).

    Parameters
    ----------
    img:
        2D image array.
    radius:
        Neighbourhood radius for the LBP operator.

    Returns
    -------
    ndarray
        Normalized histogram of shape ``(_LBP_UNI_BINS,)``. The histogram
        covers every pixel, so ``h.sum() == 1`` when scaled by the bin width.

    Raises
    ------
    ValueError
        If ``img`` is not 2D.
    """

    codes = _lbp_codes(img, radius)
    uni = _LBP_UNI_MAP[codes]
    return _lbp_histogram(uni, bins=_LBP_UNI_BINS)


def _lbp_histogram(codes: np.ndarray, bins: int = 256) -> Array:
    counts = np.bincount(codes.ravel(), minlength=bins)[:bins].astype(float)
    total = float(codes.size)
    return counts / total


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
    _, _, mag = _gradient_components(img)
    return _edge_density_from_mag(mag, threshold=threshold)


def _edge_density_from_mag(mag: Array, threshold: float | None = None) -> Array:
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
    gx, gy, _ = _gradient_components(img)
    return _orientation_histogram_from_components(gx, gy, bins)


def _orientation_histogram_from_components(gx: Array, gy: Array, bins: int = 16) -> Array:
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
    _, _, mag = _gradient_components(img)
    return _contour_ratio_from_mag(mag, threshold=threshold)


def _contour_ratio_from_mag(mag: Array, threshold: float | None = None) -> Array:
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
    counts_arr = np.asarray(counts, dtype=float)
    positive = counts_arr > 0.0
    if int(np.count_nonzero(positive)) < 2:
        return np.array([0.0], dtype=float)
    used_sizes = sizes[: len(counts_arr)][positive]
    coeffs = np.polyfit(np.log(used_sizes), np.log(counts_arr[positive]), 1)
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
_DEFAULT_FEATURES_REGISTRY = dict(FEATURES_REGISTRY)
_LBP_FEATURES = frozenset({"lbp", "lbp_ri", "lbp_uniform"})
_GRADIENT_FEATURES = frozenset({"gradient", "edge_density", "orientation", "contour"})


def _is_default_feature(name: str) -> bool:
    return FEATURES_REGISTRY.get(name) is _DEFAULT_FEATURES_REGISTRY.get(name)


def feature_layout(
    bins: int = 32, selected: Iterable[str] | None = None
) -> dict[str, int]:
    """Return the number of values each registered extractor contributes.

    The total dimensionality of :func:`extract_feature_vector` depends on
    ``bins`` **and on which optional dependencies are installed** (for example
    ``wavelet`` needs PyWavelets and is skipped otherwise). Rather than
    hard-coding a number, query it:

    >>> layout = feature_layout(bins=16)
    >>> layout["intensity"], layout["hist"]
    (6, 16)
    >>> sum(layout.values()) == feature_vector_length(bins=16)
    True

    Parameters
    ----------
    bins:
        Histogram bins for the features that accept them.
    selected:
        Feature names to include; defaults to every extractor that is usable
        in the current environment.

    Returns
    -------
    dict
        Mapping of feature name to output size, in concatenation order.
        Extractors whose optional dependency is missing are omitted unless
        they were explicitly requested (in which case the ImportError
        propagates).

    Raises
    ------
    KeyError
        If a requested name is not registered.
    """

    probe = np.zeros((8, 8), dtype=float)
    names = _resolve_feature_names(selected)
    layout: dict[str, int] = {}
    for name in names:
        try:
            layout[name] = int(FEATURES_REGISTRY[name](probe, bins).size)
        except ImportError:
            if selected is not None:
                raise
    return layout


def feature_vector_length(bins: int = 32, selected: Iterable[str] | None = None) -> int:
    """Length of :func:`extract_feature_vector` output for one channel."""

    return int(sum(feature_layout(bins, selected).values()))


def _resolve_feature_names(selected: Iterable[str] | None) -> list[str]:
    if selected is None:
        return list(FEATURES_REGISTRY)
    names = list(selected)
    for name in names:
        if name not in FEATURES_REGISTRY:
            raise KeyError(name)
    return names


def _lbp_feature_from_codes(name: str, codes: np.ndarray) -> Array:
    if name == "lbp":
        return _lbp_histogram(codes)
    if name == "lbp_ri":
        return _lbp_histogram(_LBP_RI_MAP[codes])
    if name == "lbp_uniform":
        return _lbp_histogram(_LBP_UNI_MAP[codes], bins=_LBP_UNI_BINS)
    raise KeyError(name)


def _gradient_feature_from_cache(name: str, gx: Array, gy: Array, mag: Array, bins: int) -> Array:
    if name == "gradient":
        return _gradient_histogram_from_mag(mag, bins=16)
    if name == "edge_density":
        return _edge_density_from_mag(mag)
    if name == "orientation":
        return _orientation_histogram_from_components(gx, gy, bins=bins)
    if name == "contour":
        return _contour_ratio_from_mag(mag)
    raise KeyError(name)


def _extract_channel_parts(
    ch: Array,
    names: Sequence[str],
    bins: int,
    *,
    explicit: bool,
) -> list[Array]:
    parts: list[Array] = []
    lbp_codes_cache: np.ndarray | None = None
    gradient_cache: tuple[Array, Array, Array] | None = None
    for name in names:
        try:
            if name in _LBP_FEATURES and _is_default_feature(name):
                if lbp_codes_cache is None:
                    lbp_codes_cache = _lbp_codes(ch, radius=1)
                parts.append(_lbp_feature_from_codes(name, lbp_codes_cache))
            elif name in _GRADIENT_FEATURES and _is_default_feature(name):
                if gradient_cache is None:
                    gradient_cache = _gradient_components(ch)
                parts.append(_gradient_feature_from_cache(name, *gradient_cache, bins))
            else:
                parts.append(FEATURES_REGISTRY[name](ch, bins))
        except ImportError:
            if explicit:
                raise
            continue
    return parts


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
        Concatenated feature vector of length
        ``feature_vector_length(bins, selected)`` per channel. Use
        :func:`feature_layout` to see the breakdown — the size depends on
        ``bins`` and on the installed optional dependencies, so do not assume
        a fixed number.

    Raises
    ------
    KeyError
        If a requested feature name is not registered.
    ValueError
        If ``img`` is neither 2D nor 3D.
    RuntimeError
        If no extractor produced output.
    """

    explicit = selected is not None
    names = _resolve_feature_names(selected)

    if img.ndim == 2:
        channels = [img]
    elif img.ndim == 3:
        channels = [img[..., c] for c in range(img.shape[2])]
    else:
        raise ValueError("img must be 2D or 3D")

    parts: list[Array] = []
    for ch in channels:
        parts.extend(_extract_channel_parts(ch, names, bins, explicit=explicit))
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
    "feature_layout",
    "feature_vector_length",
    "scale_features",
    "rank_features",
    "select_top_k",
    "fuse_features",
    "FEATURES_REGISTRY",
]

